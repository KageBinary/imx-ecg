"""
Run inference on an ECG signal using an ONNX model (no PyTorch needed).

Usage:
    python export/infer_onnx.py \
        --onnx exports/ecg_model.onnx \
        --input data2017/training/A00001.mat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

LABEL_NAMES = {0: "Normal (N)", 1: "AFib (A)", 2: "Other (O)", 3: "Noisy (~)"}


def load_ecg_from_mat(mat_path: str, target_length: int) -> np.ndarray:
    from scipy.io import loadmat

    data = loadmat(mat_path)
    signal = data["val"][0].astype(np.float32)

    # Z-score normalize
    mean = np.mean(signal)
    std = np.std(signal)
    if std > 0:
        signal = (signal - mean) / std
    else:
        signal = signal - mean

    # Pad or truncate to target length
    if len(signal) > target_length:
        start = (len(signal) - target_length) // 2
        signal = signal[start:start + target_length]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)), mode="constant")

    # Shape: (1, 1, target_length) = (batch, channels, length)
    return signal.reshape(1, 1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX ECG inference.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to .onnx model")
    parser.add_argument("--input", type=str, required=True, help="Path to .mat ECG file")
    parser.add_argument("--input-length", type=int, default=10000, help="Expected signal length")
    args = parser.parse_args()

    signal = load_ecg_from_mat(args.input, args.input_length)

    session = ort.InferenceSession(args.onnx)
    input_name = session.get_inputs()[0].name

    logits = session.run(None, {input_name: signal})[0]
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax

    predicted_class = int(np.argmax(logits, axis=1)[0])

    print(f"[infer_onnx] input={args.input}")
    print(f"[infer_onnx] prediction: {LABEL_NAMES[predicted_class]} (class {predicted_class})")
    print(f"[infer_onnx] probabilities: {dict(zip(LABEL_NAMES.values(), probabilities[0].tolist()))}")


if __name__ == "__main__":
    main()
