"""
Hardware-agnostic inference benchmark for the FFT + global-pooling ECG model.

This measures model footprint and pure forward-pass performance using synthetic
variable-length batches, so you can compare CPU vs GPU or different machines
before moving to target hardware.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from fft_gp.models_fft_gp import ECGFFTGlobalPoolNet
from fft_gp.train_fft_gp import pick_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def make_synthetic_batch(
    batch_size: int,
    min_len: int,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,), dtype=torch.long)
    padded_len = int(lengths.max().item())
    x = torch.zeros((batch_size, 1, padded_len), dtype=torch.float32)
    for idx, length in enumerate(lengths.tolist()):
        x[idx, 0, :length] = torch.randn(length, dtype=torch.float32)
    return x.to(device), lengths.to(device)


@torch.no_grad()
def benchmark(
    model: ECGFFTGlobalPoolNet,
    device: torch.device,
    batch_size: int,
    min_len: int,
    max_len: int,
    warmup_batches: int,
    timed_batches: int,
    amp_enabled: bool,
) -> Tuple[List[float], int]:
    latencies_ms: List[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    for batch_idx in range(warmup_batches + timed_batches):
        x, lengths = make_synthetic_batch(batch_size, min_len, max_len, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            _ = model(x, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if batch_idx >= warmup_batches:
            latencies_ms.append(elapsed_ms)

    peak_cuda_bytes = 0
    if device.type == "cuda":
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    return latencies_ms, peak_cuda_bytes


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-len", type=int, default=2500)
    parser.add_argument("--max-len", type=int, default=9000)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--timed-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable autocast for CUDA benchmarking.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    set_seed(int(args.seed))
    device = pick_device(force_cpu=bool(args.cpu))
    amp_enabled = bool(args.amp) and device.type == "cuda"
    print(f"[device] {device}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fft_bins = int(checkpoint.get("args", {}).get("fft_bins", 256))
    model = ECGFFTGlobalPoolNet(num_classes=4, fft_bins=fft_bins).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    total_params, trainable_params = count_parameters(model)
    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    latencies_ms, peak_cuda_bytes = benchmark(
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        warmup_batches=int(args.warmup_batches),
        timed_batches=int(args.timed_batches),
        amp_enabled=amp_enabled,
    )

    mean_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p50_latency_ms = percentile(latencies_ms, 50)
    p95_latency_ms = percentile(latencies_ms, 95)
    throughput = float(args.batch_size) / max(mean_latency_ms / 1000.0, 1e-9)

    print(f"[checkpoint] path={checkpoint_path}")
    print(
        f"[model] total_params={total_params} trainable_params={trainable_params} "
        f"checkpoint_size_mb={checkpoint_size_mb:.3f} fft_bins={fft_bins}"
    )
    print(
        f"[benchmark] batch_size={args.batch_size} min_len={args.min_len} max_len={args.max_len} "
        f"warmup_batches={args.warmup_batches} timed_batches={args.timed_batches} amp={amp_enabled}"
    )
    print(
        f"[latency] mean_ms={mean_latency_ms:.3f} p50_ms={p50_latency_ms:.3f} "
        f"p95_ms={p95_latency_ms:.3f} throughput_samples_per_s={throughput:.3f}"
    )
    if device.type == "cuda":
        print(f"[cuda] peak_memory_mb={peak_cuda_bytes / (1024 * 1024):.3f}")


if __name__ == "__main__":
    main()
