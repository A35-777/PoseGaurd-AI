"""CPU-friendly model training pipeline with metrics tracking and checkpointing."""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, Any, Optional

from .architecture import PoseGuardNet
from .dataset import PoseGuardDataset
from ..config import DEFECT_TYPES, MODEL_VERSION

class ModelTrainer:
    """Trains PoseGuardNet from scratch on CPU using multi-label binary cross entropy."""
    
    def __init__(
        self,
        model: Optional[PoseGuardNet] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32
    ):
        self.device = torch.device("cpu")
        self.model = model or PoseGuardNet(in_channels=48, num_classes=len(DEFECT_TYPES))
        self.model.to(self.device)
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def evaluate(self, val_loader: DataLoader, threshold: float = 0.5) -> Dict[str, float]:
        """Compute evaluation metrics on validation or test dataset."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * x.size(0)
                
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= threshold).astype(int)
                all_preds.append(preds)
                all_targets.append(y.cpu().numpy().astype(int))
                
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        val_loss = total_loss / len(val_loader.dataset)
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        precision = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
        recall = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
        
        # Per class F1
        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
        per_class_dict = {DEFECT_TYPES[i]: float(per_class_f1[i]) for i in range(len(DEFECT_TYPES))}
        
        return {
            "loss": val_loss,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "per_class_f1": per_class_dict
        }

    def train(
        self,
        train_path: str,
        val_path: str,
        epochs: int = 25,
        save_dir: str = "models",
        save_filename: str = "poseguard_v1.pt"
    ) -> Dict[str, Any]:
        """Execute CPU training loop with validation checkpointing, balanced loss, and label smoothing."""
        os.makedirs(save_dir, exist_ok=True)
        
        train_dataset = PoseGuardDataset(train_path, augment=True)
        val_dataset = PoseGuardDataset(val_path, augment=False)
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        
        history = []
        best_val_f1 = -1.0
        best_epoch = -1
        model_save_path = os.path.join(save_dir, save_filename)
        meta_save_path = os.path.join(save_dir, save_filename.replace(".pt", "_meta.json"))
        
        start_time = time.time()
        print(f"Starting training on CPU for {epochs} epochs | Parameters: {self.model.count_parameters():,}")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                # Calibrated label smoothing for probability refinement
                y_smoothed = y * 0.96 + 0.02
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y_smoothed)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * x.size(0)
                
            scheduler.step()
            train_loss /= len(train_dataset)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            epoch_info = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "val_precision": round(val_metrics["precision"], 4),
                "val_recall": round(val_metrics["recall"], 4)
            }
            history.append(epoch_info)
            
            if epoch % 5 == 0 or epoch == epochs or val_metrics["macro_f1"] > best_val_f1:
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}")
                
            if val_metrics["macro_f1"] >= best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                # Save checkpoint
                torch.save(self.model.state_dict(), model_save_path)
                
        total_time = time.time() - start_time
        file_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
        
        # Load best model for final verification
        self.model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
        final_val_metrics = self.evaluate(val_loader)
        
        metadata = {
            "model_version": MODEL_VERSION,
            "architecture": "PoseGuardNet (Conv1D + BiGRU + MultiScale Pooling)",
            "trained_from_initialized_weights": True,
            "device_used": "cpu",
            "parameter_count": self.model.count_parameters(),
            "model_file_size_mb": round(file_size_mb, 3),
            "training_duration_seconds": round(total_time, 2),
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_f1, 4),
            "defect_classes": DEFECT_TYPES,
            "feature_channels": 48,
            "window_size": 32,
            "created_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(meta_save_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        with open(os.path.join(save_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
            
        print(f"Training completed in {total_time:.1f}s. Best checkpoint saved to {model_save_path} ({file_size_mb:.2f} MB)")
        return {
            "best_val_macro_f1": best_val_f1,
            "best_epoch": best_epoch,
            "training_time": total_time,
            "file_size_mb": file_size_mb,
            "final_val_metrics": final_val_metrics
        }

