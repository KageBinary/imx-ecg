"""
Board test suite — sanity check, latency, and accuracy.

Usage:
    # Sanity + latency only (no data needed):
    python3 board_test.py

    # Full suite including accuracy:
    python3 board_test.py --test-data /root/imx-ecg/data/processed
"""
import argparse
import time

import numpy as np

MODEL      = "/root/imx-ecg/artifacts/ecg_deploy_int8.tflite"
VX_DELEGATE = "/usr/lib/libvx_delegate.so"
CLASSES    = ["Normal", "AF", "Other", "Noisy"]


def load_interpreter(use_npu=True):
    import tflite_runtime.interpreter as tflite
    import os
    delegates = []
    if use_npu and os.path.exists(VX_DELEGATE):
        delegates = [tflite.load_delegate(VX_DELEGATE)]
    interp = tflite.Interpreter(model_path=MODEL, experimental_delegates=delegates)
    interp.allocate_tensors()
    return interp


def quantize_input(x_f32, interp):
    inp = interp.get_input_details()[0]
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        return np.clip(np.round(x_f32 / scale + zp), -128, 127).astype(np.int8)
    return x_f32.astype(inp["dtype"])


def run_inference(interp, x):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    raw = interp.get_tensor(out["index"])
    if out["dtype"] == np.int8:
        scale, zp = out["quantization"]
        return (raw.astype(np.float32) - zp) * scale
    return raw.astype(np.float32)


# ── 1. Sanity check ──────────────────────────────────────────────────────────

def test_sanity():
    print("=" * 60)
    print("1. SANITY CHECK")
    print("=" * 60)
    interp = load_interpreter(use_npu=True)
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f"  Input  : shape={inp_det['shape']}  dtype={inp_det['dtype'].__name__}")
    print(f"  Output : shape={out_det['shape']}  dtype={out_det['dtype'].__name__}")

    dummy = quantize_input(np.zeros(inp_det["shape"], dtype=np.float32), interp)
    logits = run_inference(interp, dummy)
    pred = int(logits.argmax())
    print(f"  Logits : {np.round(logits.flatten(), 3)}")
    print(f"  Predicted class : {pred} ({CLASSES[pred]})")
    print("  PASSED\n")


# ── 2. Latency ───────────────────────────────────────────────────────────────

def test_latency(warmup=10, reps=100):
    print("=" * 60)
    print("2. LATENCY")
    print("=" * 60)
    for label, use_npu in [("CPU", False), ("NPU", True)]:
        interp = load_interpreter(use_npu=use_npu)
        inp_det = interp.get_input_details()[0]
        dummy = quantize_input(np.random.randn(*inp_det["shape"]).astype(np.float32), interp)

        for _ in range(warmup):
            run_inference(interp, dummy)

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            run_inference(interp, dummy)
            times.append((time.perf_counter() - t0) * 1000)

        times = np.array(times)
        print(f"  {label}  mean={times.mean():.2f}ms  "
              f"p50={np.percentile(times,50):.2f}ms  "
              f"p95={np.percentile(times,95):.2f}ms  "
              f"min={times.min():.2f}ms")

    print()


# ── 3. Accuracy ──────────────────────────────────────────────────────────────

def f1_scores(y_true, y_pred, n_classes):
    scores = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom > 0 else 0.0)
    return scores


def test_accuracy(data_dir):
    print("=" * 60)
    print("3. ACCURACY  (test set)")
    print("=" * 60)
    import os
    x_path = os.path.join(data_dir, "X_test.npy")
    y_path = os.path.join(data_dir, "y_test.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"  SKIP — test data not found at {data_dir}")
        return

    X = np.load(x_path)  # [N, 1, 3000]  float32, z-scored
    y = np.load(y_path)  # [N]            int labels
    N = len(X)
    print(f"  Samples : {N}  |  Classes : {np.bincount(y.astype(int)).tolist()}")

    interp = load_interpreter(use_npu=True)
    inp_det = interp.get_input_details()[0]

    preds = []
    for i in range(N):
        x = X[i : i + 1]  # [1, 1, 3000]
        # onnx2tf converts to NHWC [1, 3000, 1]
        x_nhwc = x.transpose(0, 2, 1).astype(np.float32)
        x_q = quantize_input(x_nhwc, interp)
        logits = run_inference(interp, x_q)
        preds.append(int(logits.argmax()))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N}...")

    preds = np.array(preds)
    y_int = y.astype(int)
    acc = (preds == y_int).mean() * 100
    f1s = f1_scores(y_int, preds, len(CLASSES))
    macro_f1 = np.mean(f1s)

    print(f"\n  Accuracy   : {acc:.1f}%")
    print(f"  Macro-F1   : {macro_f1:.3f}")
    print(f"  Per-class F1:")
    for i, (name, f1) in enumerate(zip(CLASSES, f1s)):
        n = (y_int == i).sum()
        print(f"    {name:<8}  F1={f1:.3f}  (n={n})")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", default=None,
                    help="Path to processed data dir containing X_test.npy / y_test.npy")
    ap.add_argument("--no-npu", action="store_true", help="Skip NPU latency test")
    args = ap.parse_args()

    test_sanity()
    test_latency()
    if args.test_data:
        test_accuracy(args.test_data)
    else:
        print("Tip: pass --test-data /root/imx-ecg/data/processed to run accuracy test")
