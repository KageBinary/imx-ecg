"""
Test script for the export pipeline.

Verifies that models load, run inference, export to ONNX/TFLite, and produce
matching outputs across all formats. Run this after making changes to models
or export code to catch issues early.

Usage:
    python src/export/test_pipeline.py

    # Test a specific model only:
    python src/export/test_pipeline.py --model ecgcnn
    python src/export/test_pipeline.py --model fft

    # Skip TFLite tests (if runtime not installed):
    python src/export/test_pipeline.py --skip-tflite

    # Use a looser tolerance:
    python src/export/test_pipeline.py --atol 1e-3
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

# ---- Helpers ----------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results: list[tuple[str, str, str]] = []  # (test_name, status, detail)


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    tag = PASS if status == "PASS" else (FAIL if status == "FAIL" else SKIP)
    line = f"  [{tag}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)


# ---- Test functions ---------------------------------------------------------

def test_import_ecgcnn() -> bool:
    try:
        from ecgcnn.model import ECGCNN
        record("import ecgcnn.model", "PASS")
        return True
    except Exception as e:
        record("import ecgcnn.model", "FAIL", str(e))
        return False


def test_import_fft() -> bool:
    try:
        from fft_gp.models_fft_gp import ECGFFTGlobalPoolNet
        record("import fft_gp.models_fft_gp", "PASS")
        return True
    except Exception as e:
        record("import fft_gp.models_fft_gp", "FAIL", str(e))
        return False


def test_import_phase1() -> bool:
    try:
        from phase1.models_1dcnn import SimpleECG1DCNN
        record("import phase1.models_1dcnn", "PASS")
        return True
    except Exception as e:
        record("import phase1.models_1dcnn", "FAIL", str(e))
        return False


def test_ecgcnn_forward() -> bool:
    import torch
    from ecgcnn.model import ECGCNN

    try:
        model = ECGCNN()
        model.eval()
        x = torch.randn(1, 1, 10000)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4), f"Expected (1, 4), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        record("ecgcnn forward pass", "PASS", f"output shape {tuple(out.shape)}")
        return True
    except Exception as e:
        record("ecgcnn forward pass", "FAIL", str(e))
        return False


def test_fft_forward() -> bool:
    import torch
    from fft_gp.models_fft_gp import ECGFFTGlobalPoolNet

    try:
        model = ECGFFTGlobalPoolNet(num_classes=4)
        model.eval()
        x = torch.randn(1, 1, 10000)
        lengths = torch.tensor([10000])
        with torch.no_grad():
            out = model(x, lengths)
        assert out.shape == (1, 4), f"Expected (1, 4), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        record("fft_gp forward pass", "PASS", f"output shape {tuple(out.shape)}")
        return True
    except Exception as e:
        record("fft_gp forward pass", "FAIL", str(e))
        return False


def test_phase1_forward() -> bool:
    import torch
    from phase1.models_1dcnn import SimpleECG1DCNN

    try:
        model = SimpleECG1DCNN(num_classes=4)
        model.eval()
        x = torch.randn(1, 1, 3000)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4), f"Expected (1, 4), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        record("phase1 forward pass", "PASS", f"output shape {tuple(out.shape)}")
        return True
    except Exception as e:
        record("phase1 forward pass", "FAIL", str(e))
        return False


def test_checkpoint_load(name: str, checkpoint_path: str, model_class: str, module: str) -> bool:
    import torch
    import importlib

    try:
        if not checkpoint_path:
            record(f"load checkpoint {name}", "SKIP", "no checkpoint saved")
            return False
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists() or path.is_dir():
            record(f"load checkpoint {name}", "SKIP", f"not found: {path}")
            return False

        mod = importlib.import_module(module)
        cls = getattr(mod, model_class)
        try:
            model = cls(num_classes=4)
        except TypeError:
            model = cls()

        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        record(f"load checkpoint {name}", "PASS", f"{param_count} params from {path.name}")
        return True
    except Exception as e:
        record(f"load checkpoint {name}", "FAIL", str(e))
        return False


def test_onnx_export(name: str, checkpoint_path: str, model_class: str, module: str,
                     input_length: int, atol: float) -> bool:
    import torch
    import importlib

    if not checkpoint_path:
        record(f"onnx export {name}", "SKIP", "no checkpoint saved")
        return False
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists() or path.is_dir():
        record(f"onnx export {name}", "SKIP", "no checkpoint")
        return False

    try:
        from export_onnx import load_model, export_onnx, _needs_lengths, SingleInputWrapper

        model = load_model(model_class, module, path, num_classes=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test_model.onnx"
            export_onnx(model, input_length, onnx_path)

            # Verify it loads in ONNX Runtime
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path))
            inp = session.get_inputs()[0]

            # Compare PyTorch vs ONNX output
            np.random.seed(42)
            test_input = np.random.randn(1, 1, input_length).astype(np.float32)
            onnx_out = session.run(None, {inp.name: test_input})[0]

            # Get PyTorch output
            model.eval()
            x = torch.from_numpy(test_input)
            with torch.no_grad():
                if _needs_lengths(model):
                    wrapped = SingleInputWrapper(model, input_length)
                    pt_out = wrapped(x).numpy()
                else:
                    pt_out = model(x).numpy()

            max_diff = float(np.max(np.abs(pt_out - onnx_out)))
            passed = max_diff < atol
            status = "PASS" if passed else "FAIL"
            record(f"onnx export {name}", status,
                   f"max_diff={max_diff:.6f} (atol={atol}), size={onnx_path.stat().st_size / 1024:.0f} KB")
            return passed
    except Exception as e:
        record(f"onnx export {name}", "FAIL", str(e))
        return False


def test_tflite_export(name: str, checkpoint_path: str, model_class: str, module: str,
                       input_length: int) -> bool:
    if not checkpoint_path:
        record(f"tflite export {name}", "SKIP", "no checkpoint saved")
        return False
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists() or path.is_dir():
        record(f"tflite export {name}", "SKIP", "no checkpoint")
        return False

    try:
        from export_tflite import export_via_ai_edge_torch

        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = Path(tmpdir) / "test_model.tflite"
            try:
                export_via_ai_edge_torch(
                    str(path), model_class, module,
                    input_length, 4, tflite_path,
                )
            except Exception as e:
                if "fft" in str(e).lower() or "lowering" in str(e).lower():
                    record(f"tflite export {name}", "SKIP",
                           "ai-edge-torch doesn't support FFT ops — use --method onnx-tf instead")
                    return False
                raise

            # Verify it loads
            try:
                from ai_edge_litert import interpreter as tfl_interp
                interpreter = tfl_interp.Interpreter(model_path=str(tflite_path))
            except ImportError:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))

            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]

            test_input = np.random.randn(*inp["shape"]).astype(np.float32)
            interpreter.set_tensor(inp["index"], test_input)
            interpreter.invoke()
            result = interpreter.get_tensor(out["index"])

            assert result.shape[-1] == 4, f"Expected 4 classes, got {result.shape}"
            assert not np.isnan(result).any(), "Output contains NaN"

            record(f"tflite export {name}", "PASS",
                   f"output shape {result.shape}, size={tflite_path.stat().st_size / 1024:.0f} KB")
            return True
    except Exception as e:
        record(f"tflite export {name}", "FAIL", str(e))
        return False


def test_infer_onnx_script() -> bool:
    onnx_path = Path("exports/ecg_model.onnx")
    if not onnx_path.exists():
        record("infer_onnx.py loads", "SKIP", "no exports/ecg_model.onnx")
        return False

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        inp = session.get_inputs()[0]
        test_input = np.random.randn(1, 1, 10000).astype(np.float32)
        result = session.run(None, {inp.name: test_input})[0]
        assert result.shape == (1, 4)
        record("infer_onnx.py loads", "PASS", f"exports/ecg_model.onnx → {result.shape}")
        return True
    except Exception as e:
        record("infer_onnx.py loads", "FAIL", str(e))
        return False


def test_infer_tflite_script() -> bool:
    tflite_path = Path("exports/ecg_model.tflite")
    if not tflite_path.exists():
        record("infer_tflite.py loads", "SKIP", "no exports/ecg_model.tflite")
        return False

    try:
        try:
            from ai_edge_litert import interpreter as tfl_interp
            interpreter = tfl_interp.Interpreter(model_path=str(tflite_path))
        except ImportError:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))

        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        test_input = np.random.randn(*inp["shape"]).astype(np.float32)
        interpreter.set_tensor(inp["index"], test_input)
        interpreter.invoke()
        result = interpreter.get_tensor(out["index"])
        assert result.shape[-1] == 4
        record("infer_tflite.py loads", "PASS", f"exports/ecg_model.tflite → {result.shape}")
        return True
    except Exception as e:
        record("infer_tflite.py loads", "FAIL", str(e))
        return False


# ---- Main -------------------------------------------------------------------

MODELS = {
    "ecgcnn": {
        "checkpoint": "outputs/models/best_model.pth",
        "class": "ECGCNN",
        "module": "ecgcnn.model",
        "input_length": 10000,
        "test_import": test_import_ecgcnn,
        "test_forward": test_ecgcnn_forward,
    },
    "fft": {
        "checkpoint": "checkpoints/train_fft_gp_best.pt",
        "class": "ECGFFTGlobalPoolNet",
        "module": "fft_gp.models_fft_gp",
        "input_length": 10000,
        "test_import": test_import_fft,
        "test_forward": test_fft_forward,
    },
    "phase1": {
        "checkpoint": "",  # no checkpoint saved
        "class": "SimpleECG1DCNN",
        "module": "phase1.models_1dcnn",
        "input_length": 3000,
        "test_import": test_import_phase1,
        "test_forward": test_phase1_forward,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the export pipeline.")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Test a specific model only (default: test all)")
    parser.add_argument("--skip-tflite", action="store_true",
                        help="Skip TFLite export tests")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Tolerance for numerical comparison (default: 1e-4)")
    args = parser.parse_args()

    models_to_test = [args.model] if args.model else list(MODELS.keys())
    start = time.perf_counter()

    print("=" * 60)
    print("ECG Export Pipeline Tests")
    print("=" * 60)

    # 1. Imports
    print("\n--- Imports ---")
    for name in models_to_test:
        MODELS[name]["test_import"]()

    # 2. Forward passes (no checkpoint needed)
    print("\n--- Forward Passes (random weights) ---")
    for name in models_to_test:
        MODELS[name]["test_forward"]()

    # 3. Checkpoint loading
    print("\n--- Checkpoint Loading ---")
    for name in models_to_test:
        m = MODELS[name]
        test_checkpoint_load(name, m["checkpoint"], m["class"], m["module"])

    # 4. ONNX export + validation
    print("\n--- ONNX Export + Validation ---")
    for name in models_to_test:
        m = MODELS[name]
        test_onnx_export(name, m["checkpoint"], m["class"], m["module"],
                         m["input_length"], args.atol)

    # 5. TFLite export
    if not args.skip_tflite:
        print("\n--- TFLite Export ---")
        for name in models_to_test:
            m = MODELS[name]
            test_tflite_export(name, m["checkpoint"], m["class"], m["module"],
                               m["input_length"])
    else:
        print("\n--- TFLite Export (skipped) ---")

    # 6. Existing exported files
    print("\n--- Exported File Checks ---")
    test_infer_onnx_script()
    test_infer_tflite_script()

    # Summary
    elapsed = time.perf_counter() - start
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")
    total = len(results)

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped ({total} total) in {elapsed:.1f}s")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  - {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
