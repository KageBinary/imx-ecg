"""
Variable-length ECG classifier with time-domain and FFT branches.

- Time branch: configurable Conv/BN/ReLU/Pool stack + masked global average pooling
- Frequency branch: batched FFT magnitude pooled to fixed bins, linear projection only
- Fusion head: concatenates both views before classification

Architecture is fully parameterized so CLI flags and hyperparameter search can
control depth, width, and regularization without touching this file.
"""

from __future__ import annotations

from typing import Sequence

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
    """
    Args:
        num_classes:   Number of output classes.
        fft_bins:      Number of frequency bins after adaptive pooling.
        time_channels: Output channel count for each time-branch conv block.
                       Length determines the number of pooling layers.
                       Example: (16, 32, 64) → 3 blocks, 3× downsampling.
        time_kernels:  Kernel size for each block. Must match len(time_channels).
                       Single int broadcasts to all blocks.
        pool_stride:   Stride of each MaxPool1d layer (applied after every block).
        freq_hidden:   Output size of the FFT linear projection.
        head_hidden:   Hidden size of the classification head.
        dropout:       Dropout probability in the head.
    """

    def __init__(
        self,
        num_classes: int = 4,
        fft_bins: int = 256,
        time_channels: Sequence[int] = (16, 32, 64),
        time_kernels: int | Sequence[int] = (7, 5, 5),
        pool_stride: int = 2,
        freq_hidden: int = 128,
        head_hidden: int = 128,
        dropout: float = 0.3,
        backbone_dropout: float = 0.0,
    ):
        super().__init__()
        self.fft_bins = int(fft_bins)
        self.pool_stride = int(pool_stride)
        self.num_blocks = len(time_channels)

        # Broadcast single kernel size to all blocks
        if isinstance(time_kernels, int):
            time_kernels = [time_kernels] * self.num_blocks

        if len(time_kernels) != self.num_blocks:
            raise ValueError(
                f"time_kernels length ({len(time_kernels)}) must match "
                f"time_channels length ({self.num_blocks})"
            )

        # Build time backbone dynamically
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k in zip(time_channels, time_kernels):
            block: list[nn.Module] = [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            if backbone_dropout > 0.0:
                block.append(nn.Dropout1d(backbone_dropout))
            block.append(nn.MaxPool1d(pool_stride))
            layers += block
            in_ch = out_ch
        self.time_backbone = nn.Sequential(*layers)
        self._time_out_channels = int(time_channels[-1])

        self.freq_proj = nn.Linear(fft_bins, freq_hidden)

        self.head = nn.Sequential(
            nn.Linear(self._time_out_channels + freq_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def _encode_frequency(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Compute FFT per-sample on the unpadded signal to avoid spectral artifacts
        # from zero-padding. adaptive_avg_pool1d normalises variable-length spectra.
        spectra: list[torch.Tensor] = []
        for i in range(x.shape[0]):
            L = int(lengths[i].item())
            sig = x[i, :, :L]                                          # (1, L)
            spec = torch.fft.rfft(sig, dim=-1).abs()                   # (1, L//2+1)
            spec = torch.log1p(spec)
            spec = F.adaptive_avg_pool1d(spec, self.fft_bins)          # (1, fft_bins)
            spectra.append(spec.squeeze(0))                             # (fft_bins,)
        spectrum = torch.stack(spectra, dim=0)                          # (B, fft_bins)
        return F.relu(self.freq_proj(spectrum))                         # (B, freq_hidden)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        time_features = self.time_backbone(x)
        pool_factor = self.pool_stride ** self.num_blocks
        pooled_lengths = torch.clamp(lengths // pool_factor, min=1)
        time_vec = masked_global_avg_pool_1d(time_features, pooled_lengths)

        freq_vec = self._encode_frequency(x, lengths)

        fused = torch.cat([time_vec, freq_vec], dim=1)
        return self.head(fused)
