"""
Phase 1: Small 1D CNN baseline.

- Conv/ReLU/Pool blocks
- Global average pooling
- Linear classifier
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SimpleECG1DCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Makes the head independent of exact input length after pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),          # (B, 64, 1) -> (B, 64)
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.gap(x)
        x = self.head(x)
        return x
