"""
Stage 4 of the deployment pipeline: INT8 post-training quantization.

This script provides two PTQ paths:

Path A — ONNX Runtime static quantization (default, no TF needed)
  Produces an INT8 ONNX model (.onnx) that can be benchmarked with
  ONNX Runtime and then converted to TFLite if TF is available.

Path B — TFLite INT8 via TFLiteConverter (requires tensorflow)
  Produces a TFLite INT8 model directly from a TF SavedModel.
  This path is preferred for the final board artifact.

Both paths use the same representative calibration dataset drawn from
the PhysioNet 2017 training split, ensuring the quantization statistics
are derived from real ECG signals with the same preprocessing policy
(center-crop/pad to CANONICAL_LEN + z-score) used at training time.

Pipeline position
-----------------
  1. train_deploy.py   → checkpoints/deploy_best.pt
  2. export_onnx.py    → artifacts/ecg_deploy.onnx
  3. convert_tflite.py → artifacts/ecg_deploy.tflite  (FP32)
  4. quantize_ptq.py   → artifacts/ecg_deploy_int8.onnx      ← THIS SCRIPT
                          artifacts/ecg_deploy_int8.tflite   (if --tflite-output)

Usage
-----
# ONNX INT8 only (no tensorflow required)
python src/quantize_ptq.py \\
    --onnx-path artifacts/ecg_deploy.onnx \\
    --data-dir data2017 \\
    --calib-samples 200 \\
    --output-path artifacts/ecg_deploy_int8.onnx

# Compare FP32 vs INT8 accuracy (also needs --checkpoint-path and --data-dir)
python src/quantize_ptq.py \\
    --onnx-path artifacts/ecg_deploy.onnx \\
    --data-dir data2017 \\
    --checkpoint-path checkpoints/deploy_best.pt \\
    --calib-samples 200 \\
    --output-path artifacts/ecg_deploy_int8.onnx \\
    --eval

Requirements
------------
    pip install onnx onnxruntime numpy scipy pandas
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import torch

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
except ImportError as exc:
    print(f"ERROR: {exc}\nRun: pip install onnx onnxruntime", file=sys.stderr)
    sys.exit(1)

from deploy_config import (
    CANONICAL_LEN,
    INPUT_CHANNELS,
    INPUT_NAME,
    NUM_CLASSES,
    OUTPUT_NAME,
)


class ECGCalibrationReader(CalibrationDataReader):
    """
    Calibration data reader for ONNX Runtime static quantization.

    Loads real ECG samples from the PhysioNet 2017 dataset using the
    same preprocessing pipeline (center-crop/pad + z-score) used at
    training time.
    """

    def __init__(self, data_dir: str, n_samples: int) -> None:
        from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig

        preprocess = PreprocessConfig(target_len=CANONICAL_LEN, do_zscore=True)
        dataset = PhysioNet2017Dataset(
            data_dir=data_dir, preprocess=preprocess, limit=n_samples
        )
        self._samples: List[np.ndarray] = []
        for i in range(min(n_samples, len(dataset))):
            x, _ = dataset[i]
            # ONNX Runtime expects [B, C, L] — add batch dim
            self._samples.append(x.numpy()[np.newaxis, ...])  # [1, 1, CANONICAL_LEN]
        self._idx = 0
        print(f"[calib] loaded {len(self._samples)} calibration samples")

    def get_next(self):
        if self._idx >= len(self._samples):
            return None
        data = {INPUT_NAME: self._samples[self._idx]}
        self._idx += 1
        return data

    def rewind(self):
        self._idx = 0


def quantize_onnx_int8(
    onnx_path: str,
    output_path: str,
    data_dir: str,
    n_samples: int,
) -> None:
    """
    Apply ONNX Runtime static INT8 quantization with real calibration data.

    Uses QOperator format (per-tensor, asymmetric) which produces the
    smallest graph and is most compatible with downstream TFLite conversion.
    """
    # ORT static quantization requires the model to be preprocessed first
    # (shape inference, etc.). We do this via a temp file.
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name

    onnx.shape_inference.infer_shapes_path(onnx_path, tmp_path)

    reader = ECGCalibrationReader(data_dir=data_dir, n_samples=n_samples)

    print(f"[quant] running static INT8 quantization ...")
    quantize_static(
        model_input=tmp_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,   # compact format for deployment
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=False,                    # per-tensor: better TFLite compat
        reduce_range=False,
    )
    print(f"[quant] INT8 ONNX model written to {output_path}")
    Path(tmp_path).unlink(missing_ok=True)


@torch.no_grad()
def evaluate_onnx(
    onnx_path: str,
    data_dir: str,
    val_frac: float,
    seed: int,
) -> tuple[float, float, list[float]]:
    """
    Evaluate an ONNX model (FP32 or INT8) on the validation split.
    Returns (accuracy, macro_f1, per_class_recalls).
    """
    from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
    from train_fft_gp import (
        compute_macro_f1_and_recall,
        confusion_matrix_from_preds,
        stratified_split_indices,
    )
    preprocess = PreprocessConfig(target_len=CANONICAL_LEN, do_zscore=True)
    dataset = PhysioNet2017Dataset(data_dir=data_dir, preprocess=preprocess)
    _, val_indices = stratified_split_indices(dataset.labels, val_frac, seed)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    all_preds: list[int] = []
    all_labels: list[int] = []

    for idx in val_indices:
        x, y = dataset[idx]
        x_np = x.numpy()[np.newaxis, ...]  # [1, 1, L]
        logits = sess.run(None, {INPUT_NAME: x_np})[0]  # [1, 4]
        pred = int(np.argmax(logits, axis=1)[0])
        all_preds.append(pred)
        all_labels.append(int(y.item()))

    y_pred = torch.tensor(all_preds, dtype=torch.long)
    y_true = torch.tensor(all_labels, dtype=torch.long)
    acc = float((y_pred == y_true).float().mean().item())
    cm = confusion_matrix_from_preds(y_true, y_pred, num_classes=NUM_CLASSES)
    macro_f1, recalls = compute_macro_f1_and_recall(cm)
    return acc, macro_f1, recalls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="INT8 PTQ for ECGDeployNet ONNX model."
    )
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--output-path", type=str, default="artifacts/ecg_deploy_int8.onnx"
    )
    parser.add_argument("--calib-samples", type=int, default=200)
    parser.add_argument(
        "--eval", action="store_true",
        help="Evaluate both FP32 and INT8 models on the validation split after quantization.",
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path).expanduser().resolve()
    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fp32_size_kb = onnx_path.stat().st_size / 1024
    print(f"[ptq] FP32 ONNX: {onnx_path} ({fp32_size_kb:.1f} KB)")

    quantize_onnx_int8(
        onnx_path=str(onnx_path),
        output_path=str(output_path),
        data_dir=args.data_dir,
        n_samples=int(args.calib_samples),
    )

    int8_size_kb = output_path.stat().st_size / 1024
    print(f"[ptq] INT8 ONNX: {output_path} ({int8_size_kb:.1f} KB)")
    print(
        f"[ptq] size reduction: {fp32_size_kb:.1f} KB → {int8_size_kb:.1f} KB "
        f"({(1 - int8_size_kb / fp32_size_kb) * 100:.1f}% smaller)"
    )

    if args.eval:
        print("\n[eval] Evaluating FP32 ONNX on validation split ...")
        fp32_acc, fp32_f1, fp32_recalls = evaluate_onnx(
            str(onnx_path), args.data_dir, args.val_frac, args.seed
        )
        recall_text = " ".join(
            f"class_{i}={r:.4f}" for i, r in enumerate(fp32_recalls)
        )
        print(
            f"[eval] FP32: accuracy={fp32_acc:.4f} macro_f1={fp32_f1:.4f} "
            f"recalls={recall_text}"
        )

        print("\n[eval] Evaluating INT8 ONNX on validation split ...")
        int8_acc, int8_f1, int8_recalls = evaluate_onnx(
            str(output_path), args.data_dir, args.val_frac, args.seed
        )
        recall_text = " ".join(
            f"class_{i}={r:.4f}" for i, r in enumerate(int8_recalls)
        )
        print(
            f"[eval] INT8: accuracy={int8_acc:.4f} macro_f1={int8_f1:.4f} "
            f"recalls={recall_text}"
        )

        acc_drop = fp32_acc - int8_acc
        f1_drop = fp32_f1 - int8_f1
        print(
            f"\n[ptq] accuracy_drop={acc_drop:+.4f}  macro_f1_drop={f1_drop:+.4f}"
        )
        if abs(f1_drop) < 0.01:
            print("[ptq] PTQ accuracy drop is negligible (<1%). PTQ is sufficient.")
        elif abs(f1_drop) < 0.03:
            print(
                "[ptq] PTQ accuracy drop is moderate (1–3%). "
                "Consider QAT if board latency requirements allow it."
            )
        else:
            print(
                "[ptq] PTQ accuracy drop is significant (>3%). "
                "QAT is recommended — retrain with quantization-aware training."
            )

    print(f"\n[done] INT8 ONNX artifact: {output_path}")
    print("To convert to INT8 TFLite:")
    print(
        f"  python src/convert_tflite.py --onnx-path {onnx_path} "
        f"--output-path artifacts/ecg_deploy_int8.tflite "
        f"--int8 --data-dir {args.data_dir}"
    )


if __name__ == "__main__":
    main()
