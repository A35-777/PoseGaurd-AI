"""PyTorch Dataset implementation for PoseGuard AI kinematic motion windows."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional

class PoseGuardDataset(Dataset):
    """Dataset of windowed kinematic features [48, 32] and multi-label defect targets [5]."""
    
    def __init__(
        self,
        npz_path: str,
        augment: bool = False
    ):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")
            
        data = np.load(npz_path, allow_pickle=True)
        self.windows = data["windows"].astype(np.float32) # [N, 48, 32]
        self.labels = data["labels"].astype(np.float32)   # [N, 5]
        self.seq_ids = data["sequence_ids"]
        self.augment = augment
        
    def __len__(self) -> int:
        return self.windows.shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        win = np.copy(self.windows[idx])
        label = np.copy(self.labels[idx])
        
        if self.augment:
            # Kinematic augmentation: subtle additive noise (std=0.002)
            noise = np.random.normal(0.0, 0.002, size=win.shape).astype(np.float32)
            win += noise
            # Subtle random scale (+-3%)
            scale = np.random.uniform(0.97, 1.03)
            win[:12, :] *= scale # Scale positions
            
        return torch.from_numpy(win), torch.from_numpy(label)

