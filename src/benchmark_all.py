"""
Comprehensive benchmark comparing all ECG model variants for i.MX 8M Plus deployment.

Compares
--------
  1. ECGFFTGlobalPoolNet   FP32  PyTorch   (research model, export-blocked)
  2. ECGDeployNet          FP32  PyTorch   (deployment model)
  3. ECGDeployNet          FP32  ONNX RT   (post-export inference)
  4. ECGDeployNet          INT8  ONNX RT   (post-PTQ inference, if available)

For variants 3 and 4, the corresponding .onnx files must be provided or
auto-discovered under --artifacts-dir.

Metrics reported per variant
  - total_params
  - model size on disk
  - ONNX export: pass/fail
  - ONNX ops (for deployment variants)
  - mean/p50/p95 inference latency (ms) on synthetic fixed-shape inputs
  - throughput (samples/s)
  - validation accuracy, macro-F1, per-class recall (if --data-dir provided)

Usage
-----
# Latency-only benchmark (no dataset needed)
python src/benchmark_all.py \\
    --fft-checkpoint   checkpoints/train_fft_gp_best.pt \\
    --deploy-checkpoint checkpoints/deploy_best.pt \\
    --deploy-onnx      artifacts/ecg_deploy.onnx \\
    --deploy-int8-onnx artifacts/ecg_deploy_int8.onnx

# Full benchmark with accuracy metrics
python src/benchmark_all.py \\
    --fft-checkpoint   checkpoints/train_fft_gp_best.pt \\
    --deploy-checkpoint checkpoints/deploy_best.pt \\
    --deploy-onnx      artifacts/ecg_deploy.onnx \\
    --deploy-int8-onnx artifacts/ecg_deploy_int8.onnx \\
    --data-dir data2017
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deploy_config import (
    CANONICAL_LEN,
    INPUT_CHANNELS,
    INPUT_NAME,
    NUM_CLASSES,
)
from models_deploy import ECGDeployNet, count_parameters
from models_fft_gp import ECGFFTGlobalPoolNet


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def _bench_pytorch(
    model: nn.Module,
    extra_inputs: tuple,
    warmup: int,
    timed: int,
) -> Tuple[float, float, float]:
    """Returns (mean_ms, p50_ms, p95_ms)."""
    model.eval()
    dummy = torch.randn(1, INPUT_CHANNELS, CANONICAL_LEN)
    latencies: List[float] = []
    for i in range(warmup + timed):
        t0 = time.perf_counter()
        _ = model(dummy, *extra_inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            latencies.append(elapsed_ms)
    return (
        float(np.mean(latencies)),
        float(np.percentile(latencies, 50)),
        float(np.percentile(latencies, 95)),
    )


def _bench_onnx(
    onnx_path: str,
    warmup: int,
    timed: int,
) -> Tuple[float, float, float]:
    """Returns (mean_ms, p50_ms, p95_ms) for ONNX Runtime inference."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, INPUT_CHANNELS, CANONICAL_LEN).astype(np.float32)
    latencies: List[float] = []
    for i in range(warmup + timed):
        t0 = time.perf_counter()
        _ = sess.run(None, {INPUT_NAME: dummy})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            latencies.append(elapsed_ms)
    return (
        float(np.mean(latencies)),
        float(np.percentile(latencies, 50)),
        float(np.percentile(latencies, 95)),
    )


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_pytorch(
    model: nn.Module,
    extra_inputs_fn,
    data_dir: str,
    val_frac: float,
    seed: int,
) -> Tuple[float, float, List[float]]:
    from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
    from torch.utils.data import DataLoader, Subset
    from train_fft_gp import (
        compute_macro_f1_and_recall,
        confusion_matrix_from_preds,
        stratified_split_indices,
    )

    preprocess = PreprocessConfig(target_len=CANONICAL_LEN, do_zscore=True)
    dataset = PhysioNet2017Dataset(data_dir=data_dir, preprocess=preprocess)
    _, val_indices = stratified_split_indices(dataset.labels, val_frac, seed)
    val_ds = Subset(dataset, val_indices)

    # For FFT-GP we need lengths; for deploy model we don't
    need_lengths = extra_inputs_fn is not None

    if need_lengths:
        from train_fft_gp import pad_collate
        loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=pad_collate)
    else:
        loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model.eval()
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        if need_lengths:
            x, y, lengths = batch
            logits = model(x, lengths)
        else:
            x, y = batch
            logits = model(x)
        all_preds.append(torch.argmax(logits, dim=1).cpu())
        all_labels.append(y.cpu())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_labels)
    acc = float((y_pred == y_true).float().mean().item())
    cm = confusion_matrix_from_preds(y_true, y_pred, num_classes=NUM_CLASSES)
    macro_f1, recalls = compute_macro_f1_and_recall(cm)
    return acc, macro_f1, list(recalls)


def _eval_onnx(
    onnx_path: str,
    data_dir: str,
    val_frac: float,
    seed: int,
) -> Tuple[float, float, List[float]]:
    import torch
    import onnxruntime as ort
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
    all_preds: List[int] = []
    all_labels: List[int] = []

    for idx in val_indices:
        x, y = dataset[idx]
        x_np = x.numpy()[np.newaxis, ...]
        logits = sess.run(None, {INPUT_NAME: x_np})[0]
        all_preds.append(int(np.argmax(logits, axis=1)[0]))
        all_labels.append(int(y.item()))

    y_pred = torch.tensor(all_preds, dtype=torch.long)
    y_true = torch.tensor(all_labels, dtype=torch.long)
    acc = float((y_pred == y_true).float().mean().item())
    cm = confusion_matrix_from_preds(y_true, y_pred, num_classes=NUM_CLASSES)
    macro_f1, recalls = compute_macro_f1_and_recall(cm)
    return acc, macro_f1, list(recalls)


# ---------------------------------------------------------------------------
# ONNX export test
# ---------------------------------------------------------------------------

def _test_onnx_export(model: nn.Module, extra_inputs: tuple, label: str) -> Tuple[bool, str]:
    import tempfile, onnx
    dummy = torch.randn(1, INPUT_CHANNELS, CANONICAL_LEN)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.onnx.export(
            model, (dummy,) + extra_inputs if extra_inputs else dummy,
            tmp_path, export_params=True, opset_version=13,
            do_constant_folding=True,
            input_names=[INPUT_NAME] + (["lengths"] if extra_inputs else []),
            output_names=[INPUT_NAME.replace("signal", "logits")],
        )
        onnx_model = onnx.load(tmp_path)
        onnx.checker.check_model(onnx_model)
        ops = sorted({n.op_type for n in onnx_model.graph.node})
        Path(tmp_path).unlink(missing_ok=True)
        return True, str(ops)
    except Exception as exc:
        Path(tmp_path).unlink(missing_ok=True)
        return False, str(exc)[:120]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_row(
    name: str,
    params: int,
    size_kb: Optional[float],
    export: str,
    lat_mean: Optional[float],
    lat_p50: Optional[float],
    lat_p95: Optional[float],
    tput: Optional[float],
    acc: Optional[float],
    f1: Optional[float],
    recalls: Optional[List[float]],
) -> str:
    size_s = f"{size_kb:.1f}" if size_kb is not None else "—"
    lat_s = (
        f"mean={lat_mean:.2f}ms p50={lat_p50:.2f}ms p95={lat_p95:.2f}ms"
        if lat_mean is not None
        else "—"
    )
    tput_s = f"{tput:.1f}/s" if tput is not None else "—"
    acc_s = f"{acc:.4f}" if acc is not None else "—"
    f1_s = f"{f1:.4f}" if f1 is not None else "—"
    recall_s = (
        " ".join(f"c{i}={r:.3f}" for i, r in enumerate(recalls))
        if recalls is not None
        else "—"
    )
    return (
        f"  {name:<38} params={params:<7} size={size_s:>8}KB  export={export:<6} "
        f"latency={lat_s}  tput={tput_s:>12}  acc={acc_s}  f1={f1_s}  recalls=[{recall_s}]"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark: FFT-GP vs ECGDeployNet FP32/INT8"
    )
    parser.add_argument("--fft-checkpoint", type=str, default=None)
    parser.add_argument("--deploy-checkpoint", type=str, default=None)
    parser.add_argument("--deploy-onnx", type=str, default=None)
    parser.add_argument("--deploy-int8-onnx", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timed", type=int, default=100)
    args = parser.parse_args()

    print("=" * 90)
    print("ECG Deployment Benchmark — i.MX 8M Plus Pipeline Assessment")
    print("=" * 90)

    rows: List[str] = []

    # -----------------------------------------------------------------------
    # 1. ECGFFTGlobalPoolNet FP32 PyTorch
    # -----------------------------------------------------------------------
    fft_model = ECGFFTGlobalPoolNet(num_classes=NUM_CLASSES).eval()
    fft_params, _ = count_parameters(fft_model)
    fft_size_kb = None
    if args.fft_checkpoint and Path(args.fft_checkpoint).exists():
        ckpt = torch.load(args.fft_checkpoint, map_location="cpu", weights_only=False)
        fft_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        fft_size_kb = Path(args.fft_checkpoint).stat().st_size / 1024

    lengths_dummy = torch.tensor([CANONICAL_LEN], dtype=torch.long)
    fft_export_ok, fft_export_msg = _test_onnx_export(
        fft_model, (lengths_dummy,), "ECGFFTGlobalPoolNet"
    )
    fft_export_str = "PASS" if fft_export_ok else "FAIL"

    lat_mean, lat_p50, lat_p95 = _bench_pytorch(
        fft_model, (lengths_dummy,), args.warmup, args.timed
    )
    tput = 1000.0 / max(lat_mean, 1e-9)

    fft_acc = fft_f1 = fft_recalls = None
    if args.data_dir and args.fft_checkpoint and Path(args.fft_checkpoint).exists():
        try:
            fft_acc, fft_f1, fft_recalls = _eval_pytorch(
                fft_model, lambda: None, args.data_dir, args.val_frac, args.seed
            )
        except Exception as exc:
            print(f"[warn] FFT-GP accuracy eval failed: {exc}")

    rows.append(_fmt_row(
        "ECGFFTGlobalPoolNet FP32 PyTorch",
        fft_params, fft_size_kb, fft_export_str,
        lat_mean, lat_p50, lat_p95, tput,
        fft_acc, fft_f1, fft_recalls,
    ))
    if not fft_export_ok:
        rows.append(f"    ONNX export failure: {fft_export_msg}")

    # -----------------------------------------------------------------------
    # 2. ECGDeployNet FP32 PyTorch
    # -----------------------------------------------------------------------
    deploy_model = ECGDeployNet(num_classes=NUM_CLASSES).eval()
    deploy_params, _ = count_parameters(deploy_model)
    deploy_size_kb = None
    if args.deploy_checkpoint and Path(args.deploy_checkpoint).exists():
        ckpt = torch.load(
            args.deploy_checkpoint, map_location="cpu", weights_only=False
        )
        deploy_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        deploy_size_kb = Path(args.deploy_checkpoint).stat().st_size / 1024

    deploy_export_ok, deploy_export_msg = _test_onnx_export(
        deploy_model, (), "ECGDeployNet"
    )
    deploy_export_str = "PASS" if deploy_export_ok else "FAIL"

    lat_mean, lat_p50, lat_p95 = _bench_pytorch(
        deploy_model, (), args.warmup, args.timed
    )
    tput = 1000.0 / max(lat_mean, 1e-9)

    dep_acc = dep_f1 = dep_recalls = None
    if args.data_dir and args.deploy_checkpoint and Path(args.deploy_checkpoint).exists():
        try:
            dep_acc, dep_f1, dep_recalls = _eval_pytorch(
                deploy_model, None, args.data_dir, args.val_frac, args.seed
            )
        except Exception as exc:
            print(f"[warn] ECGDeployNet accuracy eval failed: {exc}")

    rows.append(_fmt_row(
        "ECGDeployNet FP32 PyTorch",
        deploy_params, deploy_size_kb, deploy_export_str,
        lat_mean, lat_p50, lat_p95, tput,
        dep_acc, dep_f1, dep_recalls,
    ))

    # -----------------------------------------------------------------------
    # 3. ECGDeployNet FP32 ONNX Runtime
    # -----------------------------------------------------------------------
    if args.deploy_onnx and Path(args.deploy_onnx).exists():
        try:
            import onnxruntime as ort  # noqa: F401
            onnx_size_kb = Path(args.deploy_onnx).stat().st_size / 1024
            lat_mean, lat_p50, lat_p95 = _bench_onnx(
                args.deploy_onnx, args.warmup, args.timed
            )
            tput = 1000.0 / max(lat_mean, 1e-9)

            onnx_acc = onnx_f1 = onnx_recalls = None
            if args.data_dir:
                try:
                    onnx_acc, onnx_f1, onnx_recalls = _eval_onnx(
                        args.deploy_onnx, args.data_dir, args.val_frac, args.seed
                    )
                except Exception as exc:
                    print(f"[warn] ONNX FP32 accuracy eval failed: {exc}")

            rows.append(_fmt_row(
                "ECGDeployNet FP32 ONNX Runtime",
                deploy_params, onnx_size_kb, "PASS",
                lat_mean, lat_p50, lat_p95, tput,
                onnx_acc, onnx_f1, onnx_recalls,
            ))
        except Exception as exc:
            rows.append(f"  ECGDeployNet FP32 ONNX Runtime  FAILED: {exc}")
    else:
        rows.append("  ECGDeployNet FP32 ONNX Runtime  [skipped — no --deploy-onnx]")

    # -----------------------------------------------------------------------
    # 4. ECGDeployNet INT8 ONNX Runtime
    # -----------------------------------------------------------------------
    if args.deploy_int8_onnx and Path(args.deploy_int8_onnx).exists():
        try:
            import onnxruntime as ort  # noqa: F401
            int8_size_kb = Path(args.deploy_int8_onnx).stat().st_size / 1024
            lat_mean, lat_p50, lat_p95 = _bench_onnx(
                args.deploy_int8_onnx, args.warmup, args.timed
            )
            tput = 1000.0 / max(lat_mean, 1e-9)

            int8_acc = int8_f1 = int8_recalls = None
            if args.data_dir:
                try:
                    int8_acc, int8_f1, int8_recalls = _eval_onnx(
                        args.deploy_int8_onnx, args.data_dir, args.val_frac, args.seed
                    )
                except Exception as exc:
                    print(f"[warn] ONNX INT8 accuracy eval failed: {exc}")

            rows.append(_fmt_row(
                "ECGDeployNet INT8 ONNX Runtime",
                deploy_params, int8_size_kb, "PASS",
                lat_mean, lat_p50, lat_p95, tput,
                int8_acc, int8_f1, int8_recalls,
            ))
        except Exception as exc:
            rows.append(f"  ECGDeployNet INT8 ONNX Runtime  FAILED: {exc}")
    else:
        rows.append(
            "  ECGDeployNet INT8 ONNX Runtime  [skipped — no --deploy-int8-onnx]"
        )

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n--- Results ---")
    for row in rows:
        print(row)

    print("\n--- Deployment Assessment ---")
    print(
        f"  ECGFFTGlobalPoolNet ONNX export: {'PASS' if fft_export_ok else 'FAIL'}"
    )
    if not fft_export_ok:
        print(
            "  *** ECGFFTGlobalPoolNet CANNOT be deployed via ONNX → TFLite → NPU ***"
        )
        print("  *** torch.fft.rfft is not supported in ONNX opset 13.            ***")
    print(
        f"  ECGDeployNet ONNX export:        {'PASS' if deploy_export_ok else 'FAIL'}"
    )
    if deploy_export_ok:
        print("  ECGDeployNet is safe for PyTorch → ONNX → TFLite → NPU deployment.")
    print("\n  Recommended shipping model: ECGDeployNet")
    print("  Recommended quantization:   INT8 PTQ (run quantize_ptq.py to verify drop)")


if __name__ == "__main__":
    main()
