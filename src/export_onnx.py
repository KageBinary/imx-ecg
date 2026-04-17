"""
Stage 2 of the deployment pipeline: export a trained ECGDeployNet to ONNX
and validate numerical parity against the PyTorch checkpoint.

Pipeline position
-----------------
  1. train_deploy.py   → checkpoints/deploy_best.pt
  2. export_onnx.py    → artifacts/ecg_deploy.onnx          ← THIS SCRIPT
  3. convert_tflite.py → artifacts/ecg_deploy.tflite
  4. quantize_ptq.py   → artifacts/ecg_deploy_int8.onnx
                          artifacts/ecg_deploy_int8.tflite

Usage
-----
python src/export_onnx.py \\
    --checkpoint-path checkpoints/deploy_best.pt \\
    --output-path artifacts/ecg_deploy.onnx

Checks performed
----------------
* Checkpoint deploy contract matches deploy_config constants.
* ONNX checker passes (graph structure is valid).
* No dynamic axes in the exported graph (all shapes are static).
* Numerical parity: max absolute difference between PyTorch and ONNX
  Runtime outputs is below 1e-5 on 16 random inputs.
* ONNX graph ops are logged — any unfamiliar op is highlighted.

Requirements
------------
    pip install onnx onnxruntime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.onnx

# onnx and onnxruntime are required — give a clear error if missing.
try:
    import onnx
    from onnx import TensorProto
except ImportError:
    print("ERROR: 'onnx' is not installed. Run: pip install onnx", file=sys.stderr)
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print(
        "ERROR: 'onnxruntime' is not installed. Run: pip install onnxruntime",
        file=sys.stderr,
    )
    sys.exit(1)

# Add src/ to path so scripts can import sibling modules.
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deploy_config import (
    CANONICAL_LEN,
    EXPORT_BATCH,
    INPUT_CHANNELS,
    INPUT_NAME,
    NUM_CLASSES,
    ONNX_OPSET,
    OUTPUT_NAME,
)
from models_deploy import ECGDeployNet

# ONNX ops that are well-supported by onnx-tf, TFLite, and NXP eIQ delegate.
_SAFE_OPS = {
    "Conv",
    "BatchNormalization",
    "Relu",
    "Clip",          # ReLU6 exports as Clip(min=0, max=6)
    "MaxPool",
    "GlobalAveragePool",
    "Flatten",
    "Reshape",
    "Gemm",          # Linear/fully-connected
    "MatMul",
    "Add",
    "Mul",
    "Dropout",       # removed at export time (do_constant_folding handles inference)
    "Constant",
    "Cast",
    "Unsqueeze",
    "Squeeze",
    "Transpose",
    "Identity",
}


def _check_deploy_contract(checkpoint: dict) -> None:
    deploy = checkpoint.get("deploy", {})
    stored_len = deploy.get("canonical_len", None)
    if stored_len is not None and int(stored_len) != CANONICAL_LEN:
        raise ValueError(
            f"Checkpoint canonical_len={stored_len} does not match "
            f"deploy_config.CANONICAL_LEN={CANONICAL_LEN}. "
            "Retrain with the correct target length."
        )
    stored_model = deploy.get("model", None)
    if stored_model is not None and stored_model != "ECGDeployNet":
        raise ValueError(
            f"Checkpoint was trained with model='{stored_model}', not ECGDeployNet. "
            "Use export_onnx.py only with ECGDeployNet checkpoints."
        )


def _check_static_shapes(model_proto: "onnx.ModelProto") -> None:
    """Verify that all graph inputs have fully static (non-symbolic) shapes."""
    for inp in model_proto.graph.input:
        shape = inp.type.tensor_type.shape
        if shape is None:
            continue
        for dim in shape.dim:
            if dim.dim_param:  # symbolic dimension, e.g. "batch_size"
                raise RuntimeError(
                    f"Input '{inp.name}' has dynamic dimension '{dim.dim_param}'. "
                    "The deployed graph must use fully static shapes. "
                    "Ensure dynamic_axes=None (the default) in torch.onnx.export."
                )


def _warn_unknown_ops(model_proto: "onnx.ModelProto") -> None:
    ops = {n.op_type for n in model_proto.graph.node}
    unknown = ops - _SAFE_OPS
    print(f"[onnx] ops_in_graph={sorted(ops)}")
    if unknown:
        print(
            f"[onnx] WARNING: unknown/potentially unsafe ops: {sorted(unknown)}. "
            "Verify TFLite and NPU delegate compatibility before deploying."
        )
    else:
        print("[onnx] All ops are in the verified safe set.")


def run_parity_check(
    model: ECGDeployNet,
    onnx_path: str,
    n_samples: int = 16,
    tol: float = 1e-5,
) -> float:
    """
    Run n_samples random inputs through PyTorch and ONNX Runtime.
    Returns the maximum absolute difference observed.
    Raises if parity fails.
    """
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    model.eval()
    max_diff = 0.0
    for _ in range(n_samples):
        x = torch.randn(EXPORT_BATCH, INPUT_CHANNELS, CANONICAL_LEN)
        with torch.no_grad():
            pt_out = model(x).numpy()
        ort_out = sess.run(None, {INPUT_NAME: x.numpy()})[0]
        diff = float(np.abs(pt_out - ort_out).max())
        max_diff = max(max_diff, diff)
    if max_diff > tol:
        raise RuntimeError(
            f"Parity check FAILED: max_diff={max_diff:.2e} > tol={tol:.2e}. "
            "The ONNX graph does not reproduce PyTorch outputs within tolerance."
        )
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ECGDeployNet to ONNX and validate parity."
    )
    parser.add_argument(
        "--checkpoint-path", type=str, required=True,
        help="Path to a deploy_best.pt checkpoint from train_deploy.py.",
    )
    parser.add_argument(
        "--output-path", type=str, default="artifacts/ecg_deploy.onnx",
        help="Where to write the ONNX file.",
    )
    parser.add_argument(
        "--parity-samples", type=int, default=16,
        help="Number of random inputs used for the PyTorch vs ONNX parity check.",
    )
    parser.add_argument(
        "--parity-tol", type=float, default=1e-5,
        help="Max absolute difference allowed for parity to pass.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint ---
    print(f"[export] loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _check_deploy_contract(checkpoint)

    deploy_meta = checkpoint.get("deploy", {})
    channels = tuple(deploy_meta.get("channels", [32, 64, 128]))
    temporal_kernel = int(deploy_meta.get("temporal_kernel", 25))
    model = ECGDeployNet(num_classes=NUM_CLASSES, channels=channels, temporal_kernel=temporal_kernel)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[export] model=ECGDeployNet total_params={total_params} channels={list(channels)}")
    print(
        f"[export] deployment contract: "
        f"input=[{EXPORT_BATCH}, {INPUT_CHANNELS}, {CANONICAL_LEN}] "
        f"output=[{EXPORT_BATCH}, {NUM_CLASSES}] opset={ONNX_OPSET}"
    )

    # --- Export ---
    dummy = torch.zeros(EXPORT_BATCH, INPUT_CHANNELS, CANONICAL_LEN)
    print(f"[export] exporting to {output_path} ...")
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        # NO dynamic_axes — all shapes must be static for NPU deployment.
    )
    print("[export] torch.onnx.export: DONE")

    # --- Validate ONNX graph ---
    print("[validate] running onnx.checker ...")
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    print("[validate] onnx.checker: PASSED")

    _check_static_shapes(model_proto)
    print("[validate] static shapes: PASSED")

    _warn_unknown_ops(model_proto)

    # Report file size
    size_kb = output_path.stat().st_size / 1024
    print(f"[validate] onnx_size_kb={size_kb:.1f}")

    # --- Parity check ---
    print(
        f"[parity] comparing PyTorch vs ONNX Runtime "
        f"on {args.parity_samples} random inputs ..."
    )
    max_diff = run_parity_check(
        model, str(output_path),
        n_samples=int(args.parity_samples),
        tol=float(args.parity_tol),
    )
    print(f"[parity] max_abs_diff={max_diff:.2e} tol={args.parity_tol:.2e}: PASSED")
    print(f"\n[done] ONNX artifact ready: {output_path}")
    print("Next step: python src/convert_tflite.py --onnx-path", output_path)


if __name__ == "__main__":
    main()
