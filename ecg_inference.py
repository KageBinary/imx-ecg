"""
Inference wrapper for ECGDeployNet.
Loads the model from a checkpoint and runs single-sample classification.
"""
import sys
import importlib.util
import numpy as np
import torch
from pathlib import Path

_PYC_DIR = Path(__file__).parent / "src" / "__pycache__"

CLASS_NAMES = ['Normal', 'AF', 'Other', 'Noisy']
CLASS_COLORS = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
CANONICAL_LEN = 3000
SAMPLE_RATE = 300


def _load_pyc(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class ECGInference:
    def __init__(self, checkpoint_path: str):
        # deploy_config must be registered before models_deploy imports it
        dc_pyc = next(_PYC_DIR.glob("deploy_config.cpython-*.pyc"))
        _load_pyc("deploy_config", dc_pyc)

        pyc = next(_PYC_DIR.glob("models_deploy.cpython-*.pyc"))
        models_mod = _load_pyc("models_deploy", pyc)

        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        deploy = ck["deploy"]

        self.model = models_mod.ECGDeployNet(
            num_classes=deploy["num_classes"],
            channels=deploy["channels"],
            temporal_kernel=deploy["temporal_kernel"],
        )
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        self.canonical_len = deploy["canonical_len"]

        epoch = ck.get("epoch", "?")
        f1 = ck.get("best_macro_f1", "?")
        print(f"Loaded {Path(checkpoint_path).name}  epoch={epoch}  macro-F1={f1:.4f}")

    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        """
        Replicates the deployment preprocessing contract exactly:
          1. Cast + sanitise
          2. Centre-crop or zero-pad to CANONICAL_LEN
          3. Z-score over the fixed-length window
        Returns float32 tensor [1, 1, CANONICAL_LEN].
        """
        x = np.nan_to_num(signal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        n = len(x)
        if n > self.canonical_len:
            start = (n - self.canonical_len) // 2
            x = x[start: start + self.canonical_len]
        elif n < self.canonical_len:
            pad = self.canonical_len - n
            pl = pad // 2
            x = np.pad(x, (pl, pad - pl), mode="constant", constant_values=0.0)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

    def classify(self, signal: np.ndarray):
        """
        Args:
            signal: 1-D float32 array of any length (will be resampled to CANONICAL_LEN).
        Returns:
            (pred_idx: int, class_name: str, probs: np.ndarray[4])
        """
        x = self.preprocess(signal)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        pred = int(probs.argmax())
        return pred, CLASS_NAMES[pred], probs


class ECGInferenceTFLite:
    """TFLite inference wrapper — uses NPU delegate when available."""

    _DELEGATE = "/usr/lib/libvx_delegate.so"

    def __init__(self, model_path: str, use_npu: bool = True):
        import os
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            from tensorflow import lite as tflite  # type: ignore

        delegates = []
        if use_npu and os.path.exists(self._DELEGATE):
            delegates = [tflite.load_delegate(self._DELEGATE)]
            print(f"NPU delegate loaded.")
        else:
            print("TFLite running on CPU (no NPU delegate found).")

        self.interp = tflite.Interpreter(
            model_path=str(model_path),
            experimental_delegates=delegates,
        )
        self.interp.allocate_tensors()
        self._inp = self.interp.get_input_details()[0]
        self._out = self.interp.get_output_details()[0]
        self.canonical_len = CANONICAL_LEN
        print(f"Loaded {Path(model_path).name}  (TFLite)")

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(signal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        n = len(x)
        if n > self.canonical_len:
            start = (n - self.canonical_len) // 2
            x = x[start: start + self.canonical_len]
        elif n < self.canonical_len:
            pad = self.canonical_len - n
            pl = pad // 2
            x = np.pad(x, (pl, pad - pl), mode="constant", constant_values=0.0)
        x = (x - x.mean()) / (x.std() + 1e-6)
        x = x.reshape(1, 1, self.canonical_len)
        if self._inp["dtype"] == np.int8:
            scale, zp = self._inp["quantization"]
            x = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
        else:
            x = x.astype(self._inp["dtype"])
        return x

    def classify(self, signal: np.ndarray):
        x = self.preprocess(signal)
        self.interp.set_tensor(self._inp["index"], x)
        self.interp.invoke()
        raw = self.interp.get_tensor(self._out["index"])
        if self._out["dtype"] == np.int8:
            scale, zp = self._out["quantization"]
            logits = (raw.astype(np.float32) - zp) * scale
        else:
            logits = raw.astype(np.float32)
        logits = logits.squeeze()
        # softmax — model may output logits or already-normalised scores
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        pred = int(probs.argmax())
        return pred, CLASS_NAMES[pred], probs
