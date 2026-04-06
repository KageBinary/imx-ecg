"""
Validate that ONNX and TFLite exports produce the same output as the original PyTorch model.

Usage:
    python export/validate_export.py \
        --checkpoint outputs/models/best_model.pth \
        --onnx exports/ecg_model.onnx \
        --tflite exports/ecg_model.tflite \
        --input-length 10000
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))


def _needs_lengths(model: torch.nn.Module) -> bool:
    import inspect
    return "lengths" in list(inspect.signature(model.forward).parameters.keys())


def run_pytorch(model_class: str, module_name: str, checkpoint_path: Path,
                num_classes: int, test_input: np.ndarray) -> np.ndarray:
    mod = importlib.import_module(module_name)
    cls = getattr(mod, model_class)
    try:
        model = cls(num_classes=num_classes)
    except TypeError:
        model = cls()

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(test_input)
        if _needs_lengths(model):
            lengths = torch.full((x.shape[0],), x.shape[-1])
            output = model(x, lengths).numpy()
        else:
            output = model(x).numpy()
    return output


def run_onnx(onnx_path: Path, test_input: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: test_input})
    return result[0]


def run_tflite(tflite_path: Path, test_input: np.ndarray) -> np.ndarray:
    try:
        from ai_edge_litert import interpreter as tfl_interp
        interpreter = tfl_interp.Interpreter(model_path=str(tflite_path))
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


def compare(name_a: str, out_a: np.ndarray, name_b: str, out_b: np.ndarray, atol: float = 1e-4) -> bool:
    max_diff = float(np.max(np.abs(out_a - out_b)))
    mean_diff = float(np.mean(np.abs(out_a - out_b)))
    match = max_diff < atol

    status = "PASS" if match else "FAIL"
    print(f"[{status}] {name_a} vs {name_b}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} atol={atol}")

    if not match:
        print(f"  {name_a} output: {out_a.flatten()[:8]}")
        print(f"  {name_b} output: {out_b.flatten()[:8]}")

    return match


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ONNX/TFLite exports match PyTorch.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-class", type=str, default="ECGCNN")
    parser.add_argument("--module", type=str, default="model")
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--tflite", type=str, default=None)
    parser.add_argument("--input-length", type=int, default=10000)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for comparison")
    args = parser.parse_args()

    # Generate a deterministic test input
    np.random.seed(42)
    test_input = np.random.randn(1, 1, args.input_length).astype(np.float32)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    print(f"[validate] running PyTorch inference...")
    pt_output = run_pytorch(args.model_class, args.module, checkpoint_path, args.num_classes, test_input)
    print(f"[validate] PyTorch output shape={pt_output.shape} values={pt_output.flatten()[:4]}")

    all_pass = True

    if args.onnx:
        onnx_path = Path(args.onnx).expanduser().resolve()
        print(f"[validate] running ONNX Runtime inference...")
        onnx_output = run_onnx(onnx_path, test_input)
        print(f"[validate] ONNX output shape={onnx_output.shape} values={onnx_output.flatten()[:4]}")
        if not compare("PyTorch", pt_output, "ONNX", onnx_output, args.atol):
            all_pass = False

    if args.tflite:
        tflite_path = Path(args.tflite).expanduser().resolve()
        print(f"[validate] running TFLite inference...")
        tflite_output = run_tflite(tflite_path, test_input)
        print(f"[validate] TFLite output shape={tflite_output.shape} values={tflite_output.flatten()[:4]}")
        if not compare("PyTorch", pt_output, "TFLite", tflite_output, args.atol):
            all_pass = False

    if all_pass:
        print("\n[validate] ALL CHECKS PASSED")
    else:
        print("\n[validate] SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
