"""
Canonical deployment constants for the ECG i.MX 8M Plus pipeline.

Every script in the deployment chain (training, export, quantization,
evaluation, benchmarking) must import these values rather than
hard-coding them. Changing a constant here propagates everywhere.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fixed input shape — the single deployed inference contract.
#
# All recordings are center-cropped or zero-padded to this length at
# preprocessing time. 3000 samples = 10 s at 300 Hz, which covers the
# majority of PhysioNet 2017 records with minimal distortion.
# ---------------------------------------------------------------------------
CANONICAL_LEN: int = 3000          # samples (10 s @ 300 Hz)
INPUT_CHANNELS: int = 1            # single-lead ECG
NUM_CLASSES: int = 4

# Export-time batch size. Always 1 for edge deployment.
EXPORT_BATCH: int = 1

# ONNX opset. 13 is widely supported by onnx-tf, onnx2tf, and TFLite toolchains.
ONNX_OPSET: int = 13

# Input tensor name used in every ONNX/TFLite artifact.
INPUT_NAME: str = "ecg_signal"
OUTPUT_NAME: str = "class_logits"

# ---------------------------------------------------------------------------
# Label mapping — frozen from the PhysioNet/CinC 2017 Challenge.
# Do NOT reorder. Class IDs must be stable across all artifacts.
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, int] = {"N": 0, "A": 1, "O": 2, "~": 3}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES: list[str] = ["Normal", "AF", "Other", "Noisy"]
