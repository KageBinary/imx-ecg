"""
Export any PyTorch ECG model to ONNX format.

Usage:
    python export/export_onnx.py \
        --checkpoint outputs/models/best_model.pth \
        --model-class ECGCNN \
        --input-length 10000 \
        --output exports/ecg_model.onnx
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import onnx

# Allow imports from parent src/ directory
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))


def load_model(model_class_name: str, module_name: str, checkpoint_path: Path, num_classes: int) -> torch.nn.Module:
    """Dynamically load any model class from any module in src/."""
    mod = importlib.import_module(module_name)
    cls = getattr(mod, model_class_name)

    # Try instantiation with and without num_classes
    try:
        model = cls(num_classes=num_classes)
    except TypeError:
        model = cls()

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle checkpoints that wrap state_dict inside a dict
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(
    model: torch.nn.Module,
    input_length: int,
    output_path: Path,
    opset_version: int = 17,
) -> None:
    """Export a PyTorch model to ONNX."""
    dummy_input = torch.randn(1, 1, input_length)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        input_names=["ecg_signal"],
        output_names=["class_logits"],
        dynamic_axes={
            "ecg_signal": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Validate the exported model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print(f"[export_onnx] saved to {output_path}")
    print(f"[export_onnx] opset_version={opset_version}")
    print(f"[export_onnx] file_size_mb={output_path.stat().st_size / (1024 * 1024):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a PyTorch ECG model to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth or .pt checkpoint")
    parser.add_argument("--model-class", type=str, default="ECGCNN", help="Model class name (default: ECGCNN)")
    parser.add_argument("--module", type=str, default="model", help="Python module containing the model class (default: model)")
    parser.add_argument("--input-length", type=int, default=10000, help="ECG signal length (default: 10000)")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of output classes (default: 4)")
    parser.add_argument("--output", type=str, default="exports/ecg_model.onnx", help="Output .onnx file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output).expanduser().resolve()

    print(f"[export_onnx] loading {args.model_class} from {args.module}")
    model = load_model(args.model_class, args.module, checkpoint_path, args.num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[export_onnx] total_params={total_params}")

    export_onnx(model, args.input_length, output_path, opset_version=args.opset)
    print("[export_onnx] done")


if __name__ == "__main__":
    main()
