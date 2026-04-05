"""
Export an ONNX ECG model to TFLite format.

Usage:
    python export/export_tflite.py \
        --onnx exports/ecg_model.onnx \
        --output exports/ecg_model.tflite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def export_via_onnx_tf(onnx_path: Path, tflite_path: Path, quantize: bool) -> None:
    """Convert ONNX -> TF SavedModel -> TFLite using onnx-tf."""
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)

    saved_model_dir = tflite_path.parent / "tf_saved_model"
    tf_rep.export_graph(str(saved_model_dir))

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)


def export_via_ai_edge_torch(checkpoint_path: str, model_class: str, module_name: str,
                              input_length: int, num_classes: int, tflite_path: Path) -> None:
    """Convert PyTorch -> TFLite directly using litert-torch (formerly ai-edge-torch)."""
    import torch
    try:
        import litert_torch as ai_edge_torch
    except ImportError:
        import ai_edge_torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import importlib
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

    dummy_input = (torch.randn(1, 1, input_length),)
    edge_model = ai_edge_torch.convert(model, dummy_input)

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(tflite_path))


def export_via_onnx2tf(onnx_path: Path, tflite_path: Path, quantize: bool) -> None:
    """Convert ONNX -> TFLite using onnx2tf (pip install onnx2tf)."""
    import onnx2tf

    output_dir = tflite_path.parent / "tf_saved_model"
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        non_verbose=True,
    )

    # onnx2tf puts the tflite in the output dir
    generated = list(output_dir.glob("*.tflite"))
    if generated:
        generated[0].rename(tflite_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export to TFLite.")
    parser.add_argument("--onnx", type=str, default=None, help="Path to .onnx file (for onnx-tf or onnx2tf method)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth/.pt checkpoint (for ai-edge-torch method)")
    parser.add_argument("--model-class", type=str, default="ECGCNN", help="Model class name")
    parser.add_argument("--module", type=str, default="model", help="Python module with model class")
    parser.add_argument("--input-length", type=int, default=10000, help="ECG signal length")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of output classes")
    parser.add_argument("--output", type=str, default="exports/ecg_model.tflite", help="Output .tflite path")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic range quantization")
    parser.add_argument(
        "--method",
        choices=("ai-edge-torch", "onnx-tf", "onnx2tf"),
        default="ai-edge-torch",
        help="Conversion method (default: ai-edge-torch)",
    )
    args = parser.parse_args()

    tflite_path = Path(args.output).expanduser().resolve()

    if args.method == "ai-edge-torch":
        if not args.checkpoint:
            parser.error("--checkpoint is required for ai-edge-torch method")
        print(f"[export_tflite] method=ai-edge-torch checkpoint={args.checkpoint}")
        export_via_ai_edge_torch(
            args.checkpoint, args.model_class, args.module,
            args.input_length, args.num_classes, tflite_path,
        )
    elif args.method == "onnx-tf":
        if not args.onnx:
            parser.error("--onnx is required for onnx-tf method")
        print(f"[export_tflite] method=onnx-tf onnx={args.onnx}")
        export_via_onnx_tf(Path(args.onnx).resolve(), tflite_path, args.quantize)
    elif args.method == "onnx2tf":
        if not args.onnx:
            parser.error("--onnx is required for onnx2tf method")
        print(f"[export_tflite] method=onnx2tf onnx={args.onnx}")
        export_via_onnx2tf(Path(args.onnx).resolve(), tflite_path, args.quantize)

    print(f"[export_tflite] saved to {tflite_path}")
    print(f"[export_tflite] file_size_mb={tflite_path.stat().st_size / (1024 * 1024):.3f}")
    print("[export_tflite] done")


if __name__ == "__main__":
    main()
