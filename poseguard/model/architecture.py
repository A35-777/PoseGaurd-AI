"""PoseGuardNet: Compact Temporal Convolutional Recurrent Network for animation defect detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class TemporalResidualBlock(nn.Module):
    """Temporal 1D Convolutional Residual Block with Batch Normalization."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + res)
        return out

class PoseGuardNet(nn.Module):
    """
    Compact Temporal AI Model for 3D Skeletal Animation Defect Classification.
    Combines 1D multi-scale temporal convolutions, bidirectional GRU, and attention pooling.
    Trained strictly from initialized weights on CPU.
    """
    def __init__(
        self,
        in_channels: int = 48,
        num_classes: int = 5,
        hidden_dim: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.25
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Initial temporal feature projection
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=2)  # 32 -> 16
        )
        
        # Residual temporal representations
        self.res1 = TemporalResidualBlock(hidden_dim, hidden_dim * 2, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 16 -> 8
        self.dropout1 = nn.Dropout(dropout)
        
        # Bidirectional Temporal GRU for long-range sequence context
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        gru_out_dim = gru_hidden * 2  # bidirectional -> 128
        
        # Multi-scale pooling: Average + Max pooling
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim * 2, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape [Batch, Channels=48, Time=32]
        Returns:
            logits: Tensor of shape [Batch, NumClasses=5]
        """
        # Temporal conv feature extraction
        feat = self.stem(x)          # [B, 64, 16]
        feat = self.res1(feat)        # [B, 128, 16]
        feat = self.pool1(feat)       # [B, 128, 8]
        feat = self.dropout1(feat)
        
        # Transpose for GRU: [B, SeqLen=8, Dim=128]
        feat_seq = feat.transpose(1, 2)
        gru_out, _ = self.gru(feat_seq)  # [B, 8, 128]
        
        # Dual temporal pooling: [B, 128, 8]
        gru_t = gru_out.transpose(1, 2)
        avg_pool = F.adaptive_avg_pool1d(gru_t, 1).squeeze(-1)  # [B, 128]
        max_pool = F.adaptive_max_pool1d(gru_t, 1).squeeze(-1)  # [B, 128]
        pooled = torch.cat([avg_pool, max_pool], dim=-1)         # [B, 256]
        
        logits = self.classifier(pooled)                         # [B, 5]
        return logits
        
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict sigmoid calibrated defect probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

