"""
Run inference on an ECG signal using a TFLite model (no PyTorch needed).

Usage:
    python export/infer_tflite.py \
        --tflite exports/ecg_model.tflite \
        --input data2017/training/A00001.mat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LABEL_NAMES = {0: "Normal (N)", 1: "AFib (A)", 2: "Other (O)", 3: "Noisy (~)"}


def load_ecg_from_mat(mat_path: str, target_length: int) -> np.ndarray:
    from scipy.io import loadmat

    data = loadmat(mat_path)
    signal = data["val"][0].astype(np.float32)

    mean = np.mean(signal)
    std = np.std(signal)
    if std > 0:
        signal = (signal - mean) / std
    else:
        signal = signal - mean

    if len(signal) > target_length:
        start = (len(signal) - target_length) // 2
        signal = signal[start:start + target_length]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)), mode="constant")

    return signal.reshape(1, 1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="TFLite ECG inference.")
    parser.add_argument("--tflite", type=str, required=True, help="Path to .tflite model")
    parser.add_argument("--input", type=str, required=True, help="Path to .mat ECG file")
    parser.add_argument("--input-length", type=int, default=10000, help="Expected signal length")
    args = parser.parse_args()

    signal = load_ecg_from_mat(args.input, args.input_length)

    try:
        from ai_edge_litert import interpreter as tfl_interp
        interpreter = tfl_interp.Interpreter(model_path=args.tflite)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=args.tflite)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], signal)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]["index"])

    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    predicted_class = int(np.argmax(logits, axis=1)[0])

    print(f"[infer_tflite] input={args.input}")
    print(f"[infer_tflite] prediction: {LABEL_NAMES[predicted_class]} (class {predicted_class})")
    print(f"[infer_tflite] probabilities: {dict(zip(LABEL_NAMES.values(), probabilities[0].tolist()))}")


if __name__ == "__main__":
    main()
