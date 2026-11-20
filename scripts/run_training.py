"""
Script to train PoseGuardNet independently from initialized weights on CPU.
Saves model weights to models/poseguard_v1.pt.
"""

import os
import sys

# Ensure local package path is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from poseguard.model.trainer import ModelTrainer

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(root_dir, "data", "processed", "train_data.npz")
    val_path = os.path.join(root_dir, "data", "processed", "val_data.npz")
    save_dir = os.path.join(root_dir, "models")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Processed datasets not found. Run scripts/generate_dataset.py first.")
        
    trainer = ModelTrainer(lr=1e-3, weight_decay=1e-4, batch_size=32)
    results = trainer.train(
        train_path=train_path,
        val_path=val_path,
        epochs=25,
        save_dir=save_dir,
        save_filename="poseguard_v1.pt"
    )
    
    print("\n" + "=" * 60)
    print("TRAINING RUN COMPLETE")
    print(f"Best Val Macro F1: {results['best_val_macro_f1']:.4f} (Epoch {results['best_epoch']})")
    print(f"Model File Size: {results['file_size_mb']:.2f} MB")
    print(f"Total CPU Training Time: {results['training_time']:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()

