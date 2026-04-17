"""
Stage 3 of the deployment pipeline: convert an ONNX model to TFLite.

Pipeline position
-----------------
  1. train_deploy.py   → checkpoints/deploy_best.pt
  2. export_onnx.py    → artifacts/ecg_deploy.onnx
  3. convert_tflite.py → artifacts/ecg_deploy.tflite        ← THIS SCRIPT
                          artifacts/ecg_deploy_int8.tflite   (if --int8)
  4. quantize_ptq.py   → artifacts/ecg_deploy_int8.onnx
                          artifacts/ecg_deploy_int8.tflite

Conversion path
---------------
  ONNX → TensorFlow SavedModel (onnx2tf or onnx-tf) → TFLite

We prefer onnx2tf (https://github.com/PINTO0309/onnx2tf) because it
produces more NPU-friendly TFLite graphs than onnx-tf for 1D CNN models.
Fall back to onnx-tf if onnx2tf is unavailable.

Installation
------------
    # Preferred:
    pip install onnx2tf tensorflow

    # Alternative:
    pip install onnx-tf tensorflow

    # For INT8 TFLite with representative dataset you also need:
    pip install numpy scipy pandas

NXP i.MX 8M Plus deployment notes
-----------------------------------
The converted TFLite model should be tested with the NXP eIQ toolkit:
  https://www.nxp.com/design/software/development-software/eiq-ml-development-environment

For NPU delegation, use the vx_delegate or neutron_delegate depending on
your BSP version:
  interpreter.ModifyGraphWithDelegate(vx_delegate)

TFLite ops in ECGDeployNet after conversion
  CONV_2D (from Conv1d), MAX_POOL_2D, MEAN (from GlobalAveragePool),
  FULLY_CONNECTED, RELU6, RESHAPE — all delegate to the NPU.

Note on Conv1D → Conv2D mapping
  TFLite represents 1D convolutions as 2D (H=1). The onnx2tf and onnx-tf
  tools handle this layout transformation automatically. The resulting
  model uses NHWC layout internally but the interface accepts the
  [1, 1, 3000] shape as specified in deploy_config.

Usage
-----
# FP32 TFLite
python src/convert_tflite.py \\
    --onnx-path artifacts/ecg_deploy.onnx \\
    --output-path artifacts/ecg_deploy.tflite

# INT8 TFLite with representative dataset (requires data dir)
python src/convert_tflite.py \\
    --onnx-path artifacts/ecg_deploy.onnx \\
    --output-path artifacts/ecg_deploy.tflite \\
    --int8 \\
    --data-dir data2017 \\
    --calib-samples 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deploy_config import CANONICAL_LEN, INPUT_CHANNELS


def _build_representative_dataset_npy(npy_path: str):
    """Build a representative dataset generator from a pre-saved .npy file."""
    import numpy as np

    arr = np.load(npy_path).astype(np.float32)  # [n, 3000, 1] NHWC

    def gen():
        for i in range(len(arr)):
            yield [arr[i : i + 1]]  # [1, 3000, 1]

    return gen


def _build_representative_dataset(data_dir: str, n_samples: int):
    """
    Generator that yields calibration inputs for TFLite INT8 PTQ.
    Returns float32 tensors in [1, 1, 3000] shape.
    """
    import numpy as np
    from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
    from deploy_config import CANONICAL_LEN, INPUT_CHANNELS

    preprocess = PreprocessConfig(target_len=CANONICAL_LEN, do_zscore=True)
    dataset = PhysioNet2017Dataset(
        data_dir=data_dir, preprocess=preprocess, limit=n_samples
    )

    def gen():
        for i in range(min(n_samples, len(dataset))):
            x, _ = dataset[i]
            # TFLite representative dataset expects list of numpy arrays
            yield [x.numpy()[np.newaxis, ...]]  # [1, 1, L]

    return gen


def _save_calib_numpy(data_dir: str, n_samples: int, out_path: str) -> None:
    """
    Save calibration data as a numpy file in TF NHWC layout [n, 3000, 1].
    onnx2tf requires calibration data in the post-conversion TF input shape.
    Our data is already z-scored; mean=0 / std=1 passes it through unchanged.
    """
    import numpy as np
    import sys
    from pathlib import Path
    _SRC = Path(__file__).parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
    from deploy_config import CANONICAL_LEN

    preprocess = PreprocessConfig(target_len=CANONICAL_LEN, do_zscore=True)
    dataset = PhysioNet2017Dataset(data_dir=data_dir, preprocess=preprocess, limit=n_samples)
    n = min(n_samples, len(dataset))
    # TF layout after onnx2tf NCHW→NHWC: [n, 3000, 1]
    arr = np.zeros((n, CANONICAL_LEN, 1), dtype=np.float32)
    for i in range(n):
        x, _ = dataset[i]
        arr[i] = x.numpy().transpose(1, 0)  # [1, 3000] → [3000, 1]
    np.save(out_path, arr)
    print(f"[convert] saved {n} calibration samples to {out_path} shape={arr.shape}")


def _convert_with_onnx2tf(
    onnx_path: str,
    output_path: str,
    int8: bool,
    data_dir: str | None,
    calib_samples: int,
    calib_npy: str | None = None,
) -> str:
    """Convert ONNX → TFLite using onnx2tf.

    For INT8: uses onnx2tf's native full_integer_quant path with a numpy
    calibration file. Calibration data is saved in TF NHWC layout [n, 3000, 1].
    For FP32: copies the float32 tflite onnx2tf produces directly.
    """
    import glob
    import os
    import shutil

    import onnx2tf

    tmp_dir = str(Path(output_path).parent / "_tf_savedmodel")

    print(f"[convert] onnx2tf: ONNX → SavedModel at {tmp_dir}")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tmp_dir,
        verbosity="error",
        flatbuffer_direct_output_saved_model=True,
    )

    # Find the SavedModel onnx2tf wrote
    saved_model_dir = None
    if (Path(tmp_dir) / "saved_model.pb").exists():
        saved_model_dir = tmp_dir
    else:
        for sub in sorted(Path(tmp_dir).iterdir()):
            if sub.is_dir() and (sub / "saved_model.pb").exists():
                saved_model_dir = str(sub)
                break

    if saved_model_dir is None:
        raise RuntimeError(
            f"onnx2tf did not produce a SavedModel in {tmp_dir}. "
            f"Contents: {os.listdir(tmp_dir)}"
        )
    print(f"[convert] found SavedModel at {saved_model_dir}")

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8:
        if calib_npy is not None:
            print(f"[convert] INT8 PTQ from prebuilt calibration data: {calib_npy}")
            converter.representative_dataset = _build_representative_dataset_npy(calib_npy)
        elif data_dir is not None:
            print(f"[convert] INT8 PTQ with {calib_samples} calibration samples")
            converter.representative_dataset = _build_representative_dataset(data_dir, calib_samples)
        else:
            raise ValueError("--data-dir or --calib-npy is required for INT8 calibration")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        print("[convert] FP32 TFLite")

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path


def _convert_with_onnxtf(
    onnx_path: str,
    output_path: str,
    int8: bool,
    data_dir: str | None,
    calib_samples: int,
) -> str:
    """Fallback: convert ONNX → TFLite using onnx-tf."""
    import onnx
    import tensorflow as tf
    from onnx_tf.backend import prepare

    print("[convert] onnx-tf: loading ONNX model")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    savedmodel_dir = str(Path(output_path).parent / "_tf_savedmodel_onnxtf")
    print(f"[convert] onnx-tf: exporting SavedModel to {savedmodel_dir}")
    tf_rep.export_graph(savedmodel_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8:
        if data_dir is None:
            raise ValueError("--data-dir is required for INT8 calibration")
        print(f"[convert] applying INT8 PTQ with {calib_samples} calibration samples")
        converter.representative_dataset = _build_representative_dataset(
            data_dir, calib_samples
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ECGDeployNet ONNX model to TFLite."
    )
    parser.add_argument(
        "--onnx-path", type=str, required=True,
        help="Path to the ONNX file produced by export_onnx.py.",
    )
    parser.add_argument(
        "--output-path", type=str, default="artifacts/ecg_deploy.tflite",
        help="Where to write the TFLite model.",
    )
    parser.add_argument(
        "--int8", action="store_true",
        help="Apply INT8 post-training quantization. Requires --data-dir.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Dataset root for INT8 calibration representative dataset.",
    )
    parser.add_argument(
        "--calib-npy", type=str, default=None,
        help="Pre-saved calibration numpy file [n, 3000, 1] NHWC. Skips dataset loading.",
    )
    parser.add_argument(
        "--calib-samples", type=int, default=200,
        help="Number of calibration samples for INT8 PTQ.",
    )
    parser.add_argument(
        "--backend", choices=("auto", "onnx2tf", "onnx-tf"), default="auto",
        help="Conversion backend. 'auto' tries onnx2tf first, falls back to onnx-tf.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path).expanduser().resolve()
    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check that tensorflow is available
    try:
        import tensorflow as tf
        print(f"[convert] tensorflow version: {tf.__version__}")
    except ImportError:
        print(
            "ERROR: tensorflow is not installed.\n"
            "  pip install tensorflow\n"
            "  pip install onnx2tf   # preferred backend\n"
            "  OR\n"
            "  pip install onnx-tf   # fallback backend",
            file=sys.stderr,
        )
        sys.exit(1)

    # Try conversion backends
    used_backend = None
    if args.backend in ("auto", "onnx2tf"):
        try:
            import onnx2tf  # noqa: F401
            used_backend = "onnx2tf"
        except ImportError:
            if args.backend == "onnx2tf":
                print(
                    "ERROR: onnx2tf is not installed. pip install onnx2tf",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("[convert] onnx2tf not found, trying onnx-tf fallback")

    if used_backend is None and args.backend in ("auto", "onnx-tf"):
        try:
            import onnx_tf  # noqa: F401
            used_backend = "onnx-tf"
        except ImportError:
            if args.backend == "onnx-tf":
                print(
                    "ERROR: onnx-tf is not installed. pip install onnx-tf",
                    file=sys.stderr,
                )
                sys.exit(1)

    if used_backend is None:
        print(
            "ERROR: No conversion backend available.\n"
            "Install one of:\n"
            "  pip install onnx2tf   # preferred\n"
            "  pip install onnx-tf   # fallback",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[convert] using backend: {used_backend}")
    print(f"[convert] onnx_path: {onnx_path}")
    print(f"[convert] output_path: {output_path}")
    print(f"[convert] int8: {args.int8}")

    try:
        if used_backend == "onnx2tf":
            _convert_with_onnx2tf(
                str(onnx_path), str(output_path),
                int8=args.int8,
                data_dir=args.data_dir,
                calib_samples=int(args.calib_samples),
                calib_npy=args.calib_npy,
            )
        else:
            _convert_with_onnxtf(
                str(onnx_path), str(output_path),
                int8=args.int8,
                data_dir=args.data_dir,
                calib_samples=int(args.calib_samples),
            )
    except Exception as exc:
        print(f"ERROR: conversion failed: {exc}", file=sys.stderr)
        raise

    size_kb = output_path.stat().st_size / 1024
    quant_label = "INT8" if args.int8 else "FP32"
    print(f"[done] {quant_label} TFLite artifact: {output_path} ({size_kb:.1f} KB)")

    if args.int8:
        print("\nNext step for board deployment:")
        print("  Copy the .tflite file to the i.MX 8M Plus and run with:")
        print("    /usr/bin/tensorflow-lite-2.x.x/examples/label_image")
        print("  Or use the NXP eIQ inference demo with vx_delegate / neutron_delegate.")
    else:
        print("\nTo generate INT8 version:")
        print(f"  python src/convert_tflite.py --onnx-path {onnx_path} \\")
        print(f"    --output-path artifacts/ecg_deploy_int8.tflite \\")
        print(f"    --int8 --data-dir <data_dir>")


if __name__ == "__main__":
    main()
