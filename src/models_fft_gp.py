"""
Variable-length ECG classifier with time-domain and FFT branches.

- Time branch: Conv/ReLU/Pool stack + masked global average pooling
- Frequency branch: per-sample FFT magnitude + compact spectral encoder
- Fusion head: concatenates both views before classification
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_global_avg_pool_1d(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Average only across valid timesteps for each sample."""
    steps = torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
    mask = steps < lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).to(dtype=x.dtype)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (x * mask).sum(dim=-1) / denom


class ECGFFTGlobalPoolNet(nn.Module):
    def __init__(self, num_classes: int = 4, fft_bins: int = 256):
        super().__init__()
        self.fft_bins = int(fft_bins)

        self.time_backbone = nn.Sequential(
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

        self.freq_backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(64 + (32 * 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def _encode_frequency(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        spectra = []
        for signal, length in zip(x, lengths.tolist()):
            valid = signal[:, :length]
            spectrum = torch.fft.rfft(valid, dim=-1).abs()
            spectrum = torch.log1p(spectrum)
            spectrum = F.adaptive_avg_pool1d(spectrum, self.fft_bins)
            spectra.append(spectrum)

        freq = torch.stack(spectra, dim=0)
        return self.freq_backbone(freq)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        time_features = self.time_backbone(x)
        pooled_lengths = torch.clamp(lengths // 8, min=1)
        time_vec = masked_global_avg_pool_1d(time_features, pooled_lengths)

        freq_vec = self._encode_frequency(x, lengths)

        fused = torch.cat([time_vec, freq_vec], dim=1)
        return self.head(fused)
