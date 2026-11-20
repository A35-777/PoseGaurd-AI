"""CPU inference engine for PoseGuardNet."""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .architecture import PoseGuardNet
from .feature_extractor import KinematicFeatureExtractor
from ..parser.skeleton import AnimationClip
from ..config import DEFECT_TYPES

class ModelInferenceEngine:
    """Loads trained PoseGuardNet checkpoint and executes CPU inference on animations."""
    
    def __init__(
        self,
        model_path: str = "models/poseguard_v1.pt",
        meta_path: Optional[str] = None
    ):
        self.device = torch.device("cpu")
        self.model = PoseGuardNet(in_channels=48, num_classes=len(DEFECT_TYPES))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.metadata = {}
        if meta_path is None:
            meta_candidate = model_path.replace(".pt", "_meta.json")
            if os.path.exists(meta_candidate):
                meta_path = meta_candidate
                
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                
        self.feature_extractor = KinematicFeatureExtractor(window_size=32, window_hop=8)
        
    def predict_windows(self, window_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on pre-extracted windows [N, 48, 32].
        Returns sigmoid probabilities [N, 5].
        """
        if window_tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor [N, 48, 32], got shape {window_tensor.shape}")
            
        x = torch.from_numpy(window_tensor).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict_clip(
        self,
        clip: AnimationClip,
        ground_y: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run full neural analysis on an AnimationClip.
        Returns window predictions, frame-level probability curves, and detected neural regions.
        """
        T = clip.num_frames
        windows, frame_ranges = self.feature_extractor.extract_windows(clip, ground_y=ground_y)
        
        window_probs = self.predict_windows(windows) # [N, 5]
        
        # Aggregate to frame-level probability curve [T, 5]
        frame_probs = np.zeros((T, len(DEFECT_TYPES)), dtype=np.float32)
        frame_counts = np.zeros(T, dtype=np.float32)
        
        for (w_start, w_end), probs in zip(frame_ranges, window_probs):
            w_end_clamped = min(T, w_end + 1)
            frame_probs[w_start:w_end_clamped] += probs
            frame_counts[w_start:w_end_clamped] += 1.0
            
        # Normalize overlapping frame averages
        valid_frames = frame_counts > 0
        frame_probs[valid_frames] /= frame_counts[valid_frames, np.newaxis]
        
        # Neural candidates
        neural_candidates = []
        for c_idx, defect_type in enumerate(DEFECT_TYPES):
            curve = frame_probs[:, c_idx]
            threshold = 0.50
            active = curve >= threshold
            
            t = 0
            while t < T:
                if active[t]:
                    start_f = t
                    while t < T and active[t]:
                        t += 1
                    end_f = t - 1
                    duration = end_f - start_f + 1
                    
                    if duration >= 4:
                        peak_prob = float(np.max(curve[start_f:end_f + 1]))
                        mean_prob = float(np.mean(curve[start_f:end_f + 1]))
                        neural_candidates.append({
                            "type": defect_type,
                            "start_frame": int(start_f),
                            "end_frame": int(end_f),
                            "duration_frames": int(duration),
                            "confidence": round(peak_prob, 3),
                            "mean_probability": round(mean_prob, 3),
                            "source": "neural_model"
                        })
                else:
                    t += 1
                    
        return {
            "window_probabilities": window_probs,
            "frame_probabilities": frame_probs,
            "frame_ranges": frame_ranges,
            "neural_candidates": neural_candidates
        }

