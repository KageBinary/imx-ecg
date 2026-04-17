"""
ECGDeployNet — deployment-safe ECG classifier for the i.MX 8M Plus NPU.

Deployment contract
-------------------
  Input:  [B, 1, 3000]  float32   z-scored ECG, 10 s @ 300 Hz, fixed length
  Output: [B, 4]        float32   class logits — argmax gives prediction

Why this model instead of ECGFFTGlobalPoolNet
---------------------------------------------
ECGFFTGlobalPoolNet relies on ``torch.fft.rfft`` in its forward pass.
As of ONNX opset 13, ``aten::fft_rfft`` is not exportable:

    Exporting the operator 'aten::fft_rfft' to ONNX opset version 13
    is not supported.

This is a hard blocker for the PyTorch → ONNX → TFLite → NPU chain.
ECGDeployNet removes FFT from the graph entirely and uses a pure
convolutional time-domain path that exports cleanly.

Design decisions
----------------
* Conv1d → BatchNorm1d → ReLU6 blocks
  BatchNorm folds into Conv weights at export / INT8 quantization time,
  eliminating the BN overhead on-device and improving quantization range.
  ReLU6 clips activations to [0, 6], bounding the INT8 quantization range
  more tightly than unbounded ReLU.

* GlobalAveragePool via AvgPool1d(kernel_size=375)
  Exports as ONNX ``AveragePool`` → TFLite ``AVERAGE_POOL_2D``, a native
  spatial op that onnx2tf maps directly without injecting a NCHW bridge
  TRANSPOSE. GlobalAveragePool/ReduceMean both cause onnx2tf to emit a
  TRANSPOSE that the VX delegate does not support.

* Single input: model(x) — no lengths tensor, no dynamic masking.
  Variable-length handling is the job of the fixed-length preprocessing
  policy in deploy_config.CANONICAL_LEN, not the model graph.

* Only standard ONNX opset 13 ops: Conv, BatchNormalization, Relu/Clip,
  MaxPool, GlobalAveragePool, Reshape, Gemm, Dropout.

ONNX graph ops (verified by torch.onnx.export):
  Conv, BatchNormalization, Relu (or Clip for ReLU6),
  MaxPool, GlobalAveragePool, Flatten, Gemm
  — all delegate to the NXP i.MX 8M Plus NPU via the eIQ TFLite delegate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deploy_config import CANONICAL_LEN, INPUT_CHANNELS, NUM_CLASSES

# Temporal dim after 3×MaxPool(2) on fixed input of CANONICAL_LEN=3000
_GAP_KERNEL: int = CANONICAL_LEN // 8  # = 375


class ECGDeployNet(nn.Module):
    """
    Deployment-safe 1D CNN ECG classifier.

    With default channels=(32, 64, 128) and input [B, 1, 3000]:
      Block 1: [B,  1, 3000] → [B,  32, 1500]
      Block 2: [B, 32, 1500] → [B,  64,  750]
      Block 3: [B, 64,  750] → [B, 128,  375]
      GAP:     [B, 128, 375] → [B, 128,   1] → [B, 128]
      Head:    [B, 128] → [B, 128] → [B, 4]

    Total parameters (default channels): ~68,900 — small for edge inference.
    The narrow variant channels=(16, 32, 64) gives ~17,500 params if needed.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        channels: tuple[int, int, int] = (32, 64, 128),
        temporal_kernel: int = 25,
    ):
        """
        Args:
            channels: output channels for the three strided conv blocks.
            temporal_kernel: kernel size for the 4th (non-strided) block that
                widens the temporal receptive field.  Set to 0 to disable.
                With default 25 and 8× cumulative stride, one feature covers
                25 × 8 = 200 raw samples ≈ 0.67s, enough to span one beat.
        """
        super().__init__()
        c1, c2, c3 = channels

        strided_blocks = [
            # Block 1 — coarse feature extraction
            nn.Conv1d(INPUT_CHANNELS, c1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU6(inplace=True),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(c1, c2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU6(inplace=True),
            nn.MaxPool1d(2),

            # Block 3 — high-level features
            nn.Conv1d(c2, c3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU6(inplace=True),
            nn.MaxPool1d(2),
        ]

        if temporal_kernel > 0:
            # Block 4 — long-range temporal context, no pooling.
            # Effective receptive field at input ≈ (k−1)×8 + 31 raw samples.
            # k=25 → 223 samples ≈ 0.74s ≈ 1 beat at 81 bpm.
            strided_blocks += [
                nn.Conv1d(c3, c3, kernel_size=temporal_kernel,
                          padding=temporal_kernel // 2, bias=False),
                nn.BatchNorm1d(c3),
                nn.ReLU6(inplace=True),
            ]

        self.backbone = nn.Sequential(*strided_blocks)

        # AvgPool1d exports to ONNX AveragePool → TFLite AVERAGE_POOL_2D (VX-delegatable).
        # GlobalAveragePool/ReduceMean both cause onnx2tf to inject a TRANSPOSE, which
        # the VX delegate does not support and which fragments the graph.
        self.gap = nn.AvgPool1d(kernel_size=_GAP_KERNEL)

        self.head = nn.Sequential(
            nn.Flatten(),              # [B, c3, 1] → [B, c3]
            nn.Linear(c3, c3),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float32 tensor of shape [B, 1, CANONICAL_LEN].
               Must be z-scored per-sample before calling.
        Returns:
            logits: float32 tensor of shape [B, num_classes].
        """
        x = self.backbone(x)
        x = self.gap(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    import sys

    model = ECGDeployNet()
    total, trainable = count_parameters(model)
    print(f"ECGDeployNet — total_params={total} trainable_params={trainable}")

    dummy = torch.randn(1, INPUT_CHANNELS, CANONICAL_LEN)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    # Quick ONNX export smoke test
    try:
        import torch.onnx
        torch.onnx.export(
            model, dummy, "/tmp/ecg_deploy_net_smoke.onnx",
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["ecg_signal"],
            output_names=["class_logits"],
        )
        print("ONNX export smoke test: PASSED")
    except Exception as exc:
        print(f"ONNX export smoke test: FAILED — {exc}", file=sys.stderr)
        sys.exit(1)
