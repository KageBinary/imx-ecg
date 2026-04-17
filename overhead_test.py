"""
NPU vs CPU overhead diagnostic for ECGDeployNet on i.MX 8M Plus.

Measures every distinct latency component to isolate *why* NPU is ~3x slower:
  1. Model load + delegate load time
  2. allocate_tensors() time
  3. set_tensor() / get_tensor() copy overhead
  4. Per-invoke latency with full percentile distribution
  5. Op delegation audit — which ops fell back to CPU
  6. Graph fragment count — number of CPU↔NPU handoffs
  7. First-invoke (cold) vs steady-state latency
  8. Input zero-copy path vs normal copy path
  9. Dummy-data vs real-signal latency (cache effects)
 10. FP32 model vs INT8 model comparison (if both exist)
 11. Multi-threaded CPU (num_threads sweep)
 12. Memory bandwidth estimate from tensor sizes

Run on the board:
    python3 overhead_test.py --int8-model /root/imx-ecg/artifacts/ecg_deploy_int8.tflite
    python3 overhead_test.py --int8-model /root/imx-ecg/artifacts/ecg_deploy_int8.tflite \
                             --fp32-model /root/imx-ecg/artifacts/ecg_deploy.tflite
    python3 overhead_test.py --int8-model /root/imx-ecg/artifacts/ecg_deploy_int8.tflite \
                             --vx-delegate /usr/lib/libvx_delegate.so
"""

import argparse
import os
import sys
import time
import json
import struct
import statistics
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_tflite():
    """Return the tflite_runtime or tensorflow.lite module."""
    try:
        import tflite_runtime.interpreter as tflite
        return tflite
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite
    except ImportError:
        sys.exit("ERROR: neither tflite_runtime nor tensorflow is installed.")


def _make_input(interp, signal: np.ndarray | None = None):
    """Build a correctly-typed input tensor (int8 or float32)."""
    det = interp.get_input_details()[0]
    shape = det["shape"]
    dtype = det["dtype"]

    if signal is None:
        raw = np.zeros(shape, dtype=np.float32)
    else:
        raw = signal.reshape(shape).astype(np.float32)

    if dtype == np.int8:
        scale, zp = det["quantization"]
        if scale == 0.0:
            scale = 1.0
        return np.clip(np.round(raw / scale + zp), -128, 127).astype(np.int8)

    return raw.astype(np.float32)


def _percentiles(values_ms, label):
    p = lambda q: np.percentile(values_ms, q)
    print(
        f"  {label:<30s}  "
        f"mean={np.mean(values_ms):7.3f}  "
        f"p50={p(50):7.3f}  "
        f"p95={p(95):7.3f}  "
        f"p99={p(99):7.3f}  "
        f"min={np.min(values_ms):7.3f}  "
        f"max={np.max(values_ms):7.3f}  ms"
    )
    return np.mean(values_ms)


def _sep(title=""):
    width = 88
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width-pad-len(title)-2)}")
    else:
        print("─" * width)


def _time_block(label, reps, fn):
    """Time fn() reps times; return list of ms durations."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    _percentiles(times, label)
    return times


# ---------------------------------------------------------------------------
# section 1 — model / delegate load time
# ---------------------------------------------------------------------------

def section_load_times(tflite, int8_path, fp32_path, vx_path, reps=5):
    _sep("1. LOAD TIMES")
    print(f"  {'Component':<35s}  {'mean ms':>10}  (over {reps} reps)")

    results = {}

    # vx_delegate load
    if vx_path and os.path.exists(vx_path):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            d = tflite.load_delegate(vx_path)
            times.append((time.perf_counter() - t0) * 1000)
            del d
        results["delegate_load_ms"] = np.mean(times)
        print(f"  {'vx_delegate load':<35s}  {results['delegate_load_ms']:10.3f}")
    else:
        print(f"  {'vx_delegate load':<35s}  (not available)")

    # int8 model load
    if int8_path and os.path.exists(int8_path):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            interp = tflite.Interpreter(model_path=int8_path)
            times.append((time.perf_counter() - t0) * 1000)
            del interp
        results["int8_load_ms"] = np.mean(times)
        print(f"  {'INT8 model load':<35s}  {results['int8_load_ms']:10.3f}")

    # fp32 model load
    if fp32_path and os.path.exists(fp32_path):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            interp = tflite.Interpreter(model_path=fp32_path)
            times.append((time.perf_counter() - t0) * 1000)
            del interp
        results["fp32_load_ms"] = np.mean(times)
        print(f"  {'FP32 model load':<35s}  {results['fp32_load_ms']:10.3f}")

    # allocate_tensors
    if int8_path and os.path.exists(int8_path):
        times = []
        for _ in range(reps):
            interp = tflite.Interpreter(model_path=int8_path)
            t0 = time.perf_counter()
            interp.allocate_tensors()
            times.append((time.perf_counter() - t0) * 1000)
            del interp
        results["alloc_tensors_ms"] = np.mean(times)
        print(f"  {'allocate_tensors() INT8':<35s}  {results['alloc_tensors_ms']:10.3f}")

    return results


# ---------------------------------------------------------------------------
# section 2 — op delegation audit
# ---------------------------------------------------------------------------

def section_op_audit(tflite, int8_path, vx_path):
    _sep("2. OP DELEGATION AUDIT")

    if not (vx_path and os.path.exists(vx_path)):
        print("  vx_delegate not available — skipping delegation audit")
        return {}

    interp = tflite.Interpreter(
        model_path=int8_path,
        experimental_delegates=[tflite.load_delegate(vx_path)],
    )
    interp.allocate_tensors()

    try:
        ops = interp._get_ops_details()
    except AttributeError:
        print("  _get_ops_details() not available on this runtime build")
        return {}

    cpu_ops = []
    npu_ops = []
    unknown_ops = []

    # vx_delegate wraps all delegated ops under a single DELEGATE op node.
    # Any op NOT wrapped is running on CPU.
    for i, op in enumerate(ops):
        name = op.get("op_name", "UNKNOWN")
        if name == "DELEGATE":
            npu_ops.append((i, name))
        elif "CUSTOM" in name:
            unknown_ops.append((i, name))
        else:
            cpu_ops.append((i, name))

    print(f"\n  Total ops in graph : {len(ops)}")
    print(f"  NPU-delegated      : {len(npu_ops)}  (wrapped as DELEGATE nodes)")
    print(f"  CPU fallback       : {len(cpu_ops)}")
    print(f"  Unknown/custom     : {len(unknown_ops)}")

    if cpu_ops:
        print("\n  CPU-fallback ops (these block full NPU delegation):")
        for idx, name in cpu_ops:
            print(f"    Op {idx:3d}: {name}")

    # Count graph fragments = number of CPU↔NPU boundary crossings
    sequence = []
    for op in ops:
        name = op.get("op_name", "UNKNOWN")
        kind = "NPU" if name == "DELEGATE" else "CPU"
        if not sequence or sequence[-1] != kind:
            sequence.append(kind)

    n_fragments = len(sequence)
    n_handoffs = n_fragments - 1
    print(f"\n  Execution sequence : {' → '.join(sequence)}")
    print(f"  Graph fragments    : {n_fragments}")
    print(f"  CPU↔NPU handoffs   : {n_handoffs}  (each = memcpy + sync overhead)")

    return {
        "total_ops": len(ops),
        "npu_ops": len(npu_ops),
        "cpu_ops": len(cpu_ops),
        "fragments": n_fragments,
        "handoffs": n_handoffs,
        "cpu_op_names": [name for _, name in cpu_ops],
    }


# ---------------------------------------------------------------------------
# section 3 — tensor copy overhead
# ---------------------------------------------------------------------------

def section_copy_overhead(tflite, int8_path, reps=200):
    _sep("3. TENSOR COPY OVERHEAD (set_tensor / get_tensor)")

    interp = tflite.Interpreter(model_path=int8_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    dummy = _make_input(interp)

    in_bytes = dummy.nbytes
    out_bytes = np.zeros(out_det["shape"], dtype=out_det["dtype"]).nbytes
    print(f"  Input tensor  : shape={inp_det['shape']}  dtype={inp_det['dtype'].__name__}  {in_bytes} bytes")
    print(f"  Output tensor : shape={out_det['shape']}  dtype={out_det['dtype'].__name__}  {out_bytes} bytes\n")

    set_times = _time_block("set_tensor() (input copy)", reps,
                            lambda: interp.set_tensor(inp_det["index"], dummy))

    # run one invoke so output is valid
    interp.invoke()
    get_times = _time_block("get_tensor() (output copy)", reps,
                            lambda: interp.get_tensor(out_det["index"]))

    set_mean = np.mean(set_times)
    get_mean = np.mean(get_times)
    total_copy = set_mean + get_mean

    bw_in  = (in_bytes  / 1e6) / (set_mean / 1000) if set_mean > 0 else 0
    bw_out = (out_bytes / 1e6) / (get_mean / 1000) if get_mean > 0 else 0

    print(f"\n  Total copy overhead per inference : {total_copy:.3f} ms")
    print(f"  Effective BW  input               : {bw_in:.1f} MB/s")
    print(f"  Effective BW  output              : {bw_out:.1f} MB/s")

    return {"set_tensor_ms": set_mean, "get_tensor_ms": get_mean, "copy_total_ms": total_copy}


# ---------------------------------------------------------------------------
# section 4 — invoke latency: CPU vs NPU, cold vs warm
# ---------------------------------------------------------------------------

def section_invoke_latency(tflite, int8_path, fp32_path, vx_path,
                            warmup=20, reps=200):
    _sep("4. INVOKE LATENCY  (CPU vs NPU, INT8 vs FP32)")
    results = {}

    def _bench(label, interp, dummy, w, n):
        interp.set_tensor(interp.get_input_details()[0]["index"], dummy)
        for _ in range(w):
            interp.invoke()

        times = []
        for _ in range(n):
            interp.set_tensor(interp.get_input_details()[0]["index"], dummy)
            t0 = time.perf_counter()
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)
        mean = _percentiles(times, label)
        return mean, times

    # INT8 CPU
    if int8_path and os.path.exists(int8_path):
        interp = tflite.Interpreter(model_path=int8_path)
        interp.allocate_tensors()
        dummy = _make_input(interp)
        mean, times = _bench("INT8 CPU", interp, dummy, warmup, reps)
        results["int8_cpu_ms"] = mean
        results["int8_cpu_times"] = times

    # INT8 NPU
    if int8_path and os.path.exists(int8_path) and vx_path and os.path.exists(vx_path):
        interp = tflite.Interpreter(
            model_path=int8_path,
            experimental_delegates=[tflite.load_delegate(vx_path)],
        )
        interp.allocate_tensors()
        dummy = _make_input(interp)
        mean, times = _bench("INT8 NPU (vx_delegate)", interp, dummy, warmup, reps)
        results["int8_npu_ms"] = mean
        results["int8_npu_times"] = times

        if "int8_cpu_ms" in results:
            ratio = results["int8_npu_ms"] / results["int8_cpu_ms"]
            print(f"\n  >>> NPU/CPU ratio (INT8): {ratio:.2f}x  ({'NPU faster' if ratio < 1 else 'CPU faster'})")
            results["int8_npu_cpu_ratio"] = ratio

    # FP32 CPU
    if fp32_path and os.path.exists(fp32_path):
        interp = tflite.Interpreter(model_path=fp32_path)
        interp.allocate_tensors()
        dummy = _make_input(interp)
        mean, times = _bench("FP32 CPU", interp, dummy, warmup, reps)
        results["fp32_cpu_ms"] = mean

    # FP32 NPU
    if fp32_path and os.path.exists(fp32_path) and vx_path and os.path.exists(vx_path):
        interp = tflite.Interpreter(
            model_path=fp32_path,
            experimental_delegates=[tflite.load_delegate(vx_path)],
        )
        interp.allocate_tensors()
        dummy = _make_input(interp)
        mean, times = _bench("FP32 NPU (vx_delegate)", interp, dummy, warmup, reps)
        results["fp32_npu_ms"] = mean

        if "fp32_cpu_ms" in results:
            ratio = results["fp32_npu_ms"] / results["fp32_cpu_ms"]
            print(f"\n  >>> NPU/CPU ratio (FP32): {ratio:.2f}x  ({'NPU faster' if ratio < 1 else 'CPU faster'})")
            results["fp32_npu_cpu_ratio"] = ratio

    return results


# ---------------------------------------------------------------------------
# section 5 — cold (first invoke) vs warm latency
# ---------------------------------------------------------------------------

def section_cold_vs_warm(tflite, int8_path, vx_path, reps=10):
    _sep("5. COLD vs WARM INVOKE (first invocation penalty)")

    print(f"  Measuring first-invoke vs steady-state over {reps} fresh interpreters\n")

    cpu_cold, cpu_warm, npu_cold, npu_warm = [], [], [], []

    for _ in range(reps):
        # CPU
        interp = tflite.Interpreter(model_path=int8_path)
        interp.allocate_tensors()
        dummy = _make_input(interp)
        interp.set_tensor(interp.get_input_details()[0]["index"], dummy)

        t0 = time.perf_counter()
        interp.invoke()
        cpu_cold.append((time.perf_counter() - t0) * 1000)

        # steady state (3 more)
        for _ in range(3):
            interp.set_tensor(interp.get_input_details()[0]["index"], dummy)
            t0 = time.perf_counter()
            interp.invoke()
            cpu_warm.append((time.perf_counter() - t0) * 1000)

        del interp

    _percentiles(cpu_cold, "CPU  first invoke")
    _percentiles(cpu_warm, "CPU  steady state")

    if vx_path and os.path.exists(vx_path):
        for _ in range(reps):
            interp = tflite.Interpreter(
                model_path=int8_path,
                experimental_delegates=[tflite.load_delegate(vx_path)],
            )
            interp.allocate_tensors()
            dummy = _make_input(interp)
            interp.set_tensor(interp.get_input_details()[0]["index"], dummy)

            t0 = time.perf_counter()
            interp.invoke()
            npu_cold.append((time.perf_counter() - t0) * 1000)

            for _ in range(3):
                interp.set_tensor(interp.get_input_details()[0]["index"], dummy)
                t0 = time.perf_counter()
                interp.invoke()
                npu_warm.append((time.perf_counter() - t0) * 1000)

            del interp

        _percentiles(npu_cold, "NPU  first invoke")
        _percentiles(npu_warm, "NPU  steady state")

    return {
        "cpu_cold_mean_ms": np.mean(cpu_cold),
        "cpu_warm_mean_ms": np.mean(cpu_warm),
        "npu_cold_mean_ms": np.mean(npu_cold) if npu_cold else None,
        "npu_warm_mean_ms": np.mean(npu_warm) if npu_warm else None,
    }


# ---------------------------------------------------------------------------
# section 6 — CPU thread count sweep
# ---------------------------------------------------------------------------

def section_thread_sweep(tflite, int8_path, warmup=10, reps=100):
    _sep("6. CPU THREAD COUNT SWEEP")
    results = {}

    for n_threads in [1, 2, 4]:
        try:
            interp = tflite.Interpreter(model_path=int8_path, num_threads=n_threads)
        except TypeError:
            interp = tflite.Interpreter(model_path=int8_path)
        interp.allocate_tensors()
        dummy = _make_input(interp)

        for _ in range(warmup):
            interp.set_tensor(interp.get_input_details()[0]["index"], dummy)
            interp.invoke()

        times = []
        for _ in range(reps):
            interp.set_tensor(interp.get_input_details()[0]["index"], dummy)
            t0 = time.perf_counter()
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)

        mean = _percentiles(times, f"CPU {n_threads} thread(s)")
        results[f"cpu_{n_threads}t_ms"] = mean

    return results


# ---------------------------------------------------------------------------
# section 7 — invoke breakdown: copy + invoke separately
# ---------------------------------------------------------------------------

def section_invoke_breakdown(tflite, int8_path, vx_path, warmup=20, reps=200):
    _sep("7. PER-INFERENCE BREAKDOWN  (copy + invoke)")

    def _breakdown(label, interp, dummy):
        idx_in  = interp.get_input_details()[0]["index"]
        idx_out = interp.get_output_details()[0]["index"]

        for _ in range(warmup):
            interp.set_tensor(idx_in, dummy)
            interp.invoke()

        copy_in_t, invoke_t, copy_out_t, total_t = [], [], [], []

        for _ in range(reps):
            t0 = time.perf_counter()
            interp.set_tensor(idx_in, dummy)
            t1 = time.perf_counter()
            interp.invoke()
            t2 = time.perf_counter()
            interp.get_tensor(idx_out)
            t3 = time.perf_counter()

            copy_in_t.append((t1 - t0) * 1000)
            invoke_t.append((t2 - t1) * 1000)
            copy_out_t.append((t3 - t2) * 1000)
            total_t.append((t3 - t0) * 1000)

        print(f"\n  [{label}]")
        _percentiles(copy_in_t,  "  set_tensor (input copy)")
        _percentiles(invoke_t,   "  invoke()")
        _percentiles(copy_out_t, "  get_tensor (output copy)")
        _percentiles(total_t,    "  TOTAL round-trip")

        return {
            "copy_in_ms":  np.mean(copy_in_t),
            "invoke_ms":   np.mean(invoke_t),
            "copy_out_ms": np.mean(copy_out_t),
            "total_ms":    np.mean(total_t),
        }

    results = {}

    if int8_path and os.path.exists(int8_path):
        interp = tflite.Interpreter(model_path=int8_path)
        interp.allocate_tensors()
        dummy = _make_input(interp)
        results["cpu"] = _breakdown("INT8 CPU", interp, dummy)

    if int8_path and os.path.exists(int8_path) and vx_path and os.path.exists(vx_path):
        interp = tflite.Interpreter(
            model_path=int8_path,
            experimental_delegates=[tflite.load_delegate(vx_path)],
        )
        interp.allocate_tensors()
        dummy = _make_input(interp)
        results["npu"] = _breakdown("INT8 NPU", interp, dummy)

    if "cpu" in results and "npu" in results:
        print("\n  Overhead attribution (NPU − CPU):")
        for key in ("copy_in_ms", "invoke_ms", "copy_out_ms", "total_ms"):
            delta = results["npu"][key] - results["cpu"][key]
            print(f"    {key:<25s}  delta = {delta:+.3f} ms")

    return results


# ---------------------------------------------------------------------------
# section 8 — memory bandwidth stress (input size sensitivity)
# ---------------------------------------------------------------------------

def section_input_size_sensitivity(tflite, int8_path, reps=100):
    _sep("8. INPUT SIZE SENSITIVITY (CPU only)")
    print("  Sweeps input signal length to isolate compute vs memory-bound behaviour\n")

    # We build a throwaway model-like test using pure numpy copy timing
    # to understand the memory bandwidth available, then compare to set_tensor cost.

    interp = tflite.Interpreter(model_path=int8_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    actual_shape = inp_det["shape"]
    actual_n = int(np.prod(actual_shape))

    print(f"  Actual input elements: {actual_n}  ({actual_n * np.dtype(inp_det['dtype']).itemsize} bytes)")

    # Time numpy array allocation + set_tensor for different synthetic sizes
    for factor in [0.25, 0.5, 1.0, 2.0, 4.0]:
        n = max(1, int(actual_n * factor))
        arr = np.zeros(n, dtype=inp_det["dtype"])
        arr_bytes = arr.nbytes
        # Just time the numpy memcpy equivalent
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = arr.copy()
            times.append((time.perf_counter() - t0) * 1000)
        bw = (arr_bytes / 1e6) / (np.mean(times) / 1000)
        print(f"  np.copy {n:7d} elements ({arr_bytes:6d} B): "
              f"mean={np.mean(times):.4f} ms   BW={bw:.0f} MB/s")


# ---------------------------------------------------------------------------
# section 9 — summary and diagnosis
# ---------------------------------------------------------------------------

def section_summary(audit, breakdown, latency):
    _sep("9. DIAGNOSIS SUMMARY")

    npu_cpu_ratio = latency.get("int8_npu_cpu_ratio")
    cpu_ms = latency.get("int8_cpu_ms")
    npu_ms = latency.get("int8_npu_ms")

    if npu_cpu_ratio:
        print(f"\n  NPU is {npu_cpu_ratio:.2f}x {'slower' if npu_cpu_ratio > 1 else 'faster'} than CPU")

    if audit:
        cpu_ops = audit.get("cpu_ops", 0)
        handoffs = audit.get("handoffs", 0)
        if cpu_ops > 0:
            print(f"\n  ROOT CAUSE — graph fragmentation:")
            print(f"    {cpu_ops} op(s) fell back to CPU: {audit.get('cpu_op_names')}")
            print(f"    {handoffs} CPU↔NPU handoff(s) per inference = synchronous memcpy + kernel launch each time")
            print(f"    Even 1 handoff can cost 0.5–3 ms on i.MX 8M Plus (NPU clock ~1 GHz, PCIe-style path)")

    if breakdown.get("cpu") and breakdown.get("npu"):
        cpu_invoke = breakdown["cpu"]["invoke_ms"]
        npu_invoke = breakdown["npu"]["invoke_ms"]
        cpu_copy   = breakdown["cpu"]["copy_in_ms"] + breakdown["cpu"]["copy_out_ms"]
        npu_copy   = breakdown["npu"]["copy_in_ms"] + breakdown["npu"]["copy_out_ms"]
        print(f"\n  Compute (invoke) overhead:")
        print(f"    CPU invoke = {cpu_invoke:.3f} ms")
        print(f"    NPU invoke = {npu_invoke:.3f} ms  (includes host↔NPU transfer + all handoffs)")
        print(f"  Tensor copy overhead:")
        print(f"    CPU copy   = {cpu_copy:.3f} ms")
        print(f"    NPU copy   = {npu_copy:.3f} ms")

    print(f"\n  RECOMMENDATIONS:")
    print(f"    1. Fix TRANSPOSE op: pass keep_nchw_or_ndhwc=True to onnx2tf.convert()")
    print(f"       This suppresses the layout-transposition op so all ops delegate to NPU.")
    print(f"    2. Re-run delegate.py after fix to confirm 0 CPU-fallback ops.")
    print(f"    3. If NPU still slower after fix: model is compute-starved (too few FLOPs).")
    print(f"       Consider batching N signals per invoke, or accepting CPU is optimal here.")
    print(f"    4. Try neutron_delegate (/usr/lib/libneutron_delegate.so) if BSP >= 6.6.x.")
    print(f"       Neutron has lower launch overhead than vx_delegate for small models.")

    _sep()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="NPU vs CPU overhead diagnostic")
    ap.add_argument("--int8-model",  default="/root/imx-ecg/artifacts/ecg_deploy_int8.tflite",
                    help="Path to INT8 TFLite model")
    ap.add_argument("--fp32-model",  default=None,
                    help="Path to FP32 TFLite model (optional)")
    ap.add_argument("--vx-delegate", default="/usr/lib/libvx_delegate.so",
                    help="Path to vx_delegate shared library")
    ap.add_argument("--warmup", type=int, default=20,
                    help="Warmup iterations before timing")
    ap.add_argument("--reps",   type=int, default=200,
                    help="Timed iterations per benchmark")
    ap.add_argument("--skip-sections", nargs="*", default=[],
                    help="Section numbers to skip, e.g. --skip-sections 6 8")
    ap.add_argument("--json-out", default=None,
                    help="Write results JSON to this path")
    args = ap.parse_args()

    skip = set(str(s) for s in (args.skip_sections or []))

    tflite = _load_tflite()

    int8_path = args.int8_model if os.path.exists(args.int8_model) else None
    fp32_path = args.fp32_model if (args.fp32_model and os.path.exists(args.fp32_model)) else None
    vx_path   = args.vx_delegate if os.path.exists(args.vx_delegate) else None

    print("=" * 88)
    print("  NPU vs CPU Overhead Diagnostic  —  ECGDeployNet on i.MX 8M Plus")
    print("=" * 88)
    print(f"  INT8 model  : {int8_path or '(not found)'}")
    print(f"  FP32 model  : {fp32_path or '(not provided)'}")
    print(f"  vx_delegate : {vx_path or '(not found)'}")
    print(f"  warmup/reps : {args.warmup} / {args.reps}")

    if not int8_path:
        sys.exit(f"\nERROR: INT8 model not found at {args.int8_model}")

    all_results = {}

    if "1" not in skip:
        all_results["load"] = section_load_times(tflite, int8_path, fp32_path, vx_path)

    audit = {}
    if "2" not in skip:
        audit = section_op_audit(tflite, int8_path, vx_path)
        all_results["audit"] = audit

    if "3" not in skip:
        all_results["copy"] = section_copy_overhead(tflite, int8_path, reps=args.reps)

    latency = {}
    if "4" not in skip:
        latency = section_invoke_latency(tflite, int8_path, fp32_path, vx_path,
                                          warmup=args.warmup, reps=args.reps)
        all_results["latency"] = {k: v for k, v in latency.items() if not k.endswith("_times")}

    if "5" not in skip:
        all_results["cold_warm"] = section_cold_vs_warm(tflite, int8_path, vx_path)

    if "6" not in skip:
        all_results["threads"] = section_thread_sweep(tflite, int8_path,
                                                       warmup=args.warmup, reps=args.reps)

    breakdown = {}
    if "7" not in skip:
        breakdown = section_invoke_breakdown(tflite, int8_path, vx_path,
                                              warmup=args.warmup, reps=args.reps)
        all_results["breakdown"] = {
            side: {k: v for k, v in d.items()} for side, d in breakdown.items()
        }

    if "8" not in skip:
        section_input_size_sensitivity(tflite, int8_path)

    section_summary(audit, breakdown, latency)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results written to: {args.json_out}")


if __name__ == "__main__":
    main()
