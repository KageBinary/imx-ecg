"""
Evaluate and benchmark a saved FFT + global-pooling ECG checkpoint.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
from models_fft_gp import ECGFFTGlobalPoolNet
from train_fft_gp import (
    compute_macro_f1_and_recall,
    confusion_matrix_from_preds,
    format_confusion_matrix,
    pad_collate,
    pick_device,
    stratified_split_indices,
)


@torch.no_grad()
def evaluate(
    model: ECGFFTGlobalPoolNet,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
) -> Tuple[float, float, List[float], torch.Tensor, float, float, int]:
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    total_samples = 0
    measured_batches = 0
    total_time_s = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_idx, (x, y, lengths) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = model(x, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        total_samples += int(y.shape[0])

        if batch_idx >= warmup_batches:
            total_time_s += elapsed
            measured_batches += 1

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    cm = confusion_matrix_from_preds(y_true, y_pred, num_classes=4)
    macro_f1, recalls = compute_macro_f1_and_recall(cm)
    accuracy = float((y_pred == y_true).float().mean().item())

    measured_samples = max(total_samples - (warmup_batches * loader.batch_size), 1)
    avg_latency_ms = (total_time_s / max(measured_batches, 1)) * 1000.0
    throughput = measured_samples / max(total_time_s, 1e-9)

    peak_cuda_bytes = 0
    if device.type == "cuda":
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))

    return accuracy, macro_f1, recalls, cm, avg_latency_ms, throughput, peak_cuda_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="Use 1 for edge-style latency measurements.")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-len", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = pick_device(force_cpu=bool(args.cpu))
    print(f"[device] {device}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    fft_bins = int(checkpoint.get("args", {}).get("fft_bins", 256))

    preprocess = PreprocessConfig(
        target_len=int(args.target_len) if int(args.target_len) > 0 else None,
        do_zscore=True,
    )
    dataset = PhysioNet2017Dataset(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        preprocess=preprocess,
        limit=int(args.subset) if args.subset and args.subset > 0 else None,
    )
    _, val_indices = stratified_split_indices(dataset.labels, float(args.val_frac), int(args.seed))
    val_ds = Subset(dataset, val_indices)

    loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
        collate_fn=pad_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )

    model = ECGFFTGlobalPoolNet(num_classes=4, fft_bins=fft_bins).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    accuracy, macro_f1, recalls, cm, avg_latency_ms, throughput, peak_cuda_bytes = evaluate(
        model,
        loader,
        device,
        warmup_batches=int(args.warmup_batches),
    )

    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    recall_text = " ".join(f"class_{idx}_recall={score:.4f}" for idx, score in enumerate(recalls))

    print(f"[checkpoint] path={checkpoint_path}")
    print(f"[quality] accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} {recall_text}")
    print(format_confusion_matrix(cm))
    print(
        f"[performance] batch_size={args.batch_size} avg_latency_ms={avg_latency_ms:.3f} "
        f"throughput_samples_per_s={throughput:.3f}"
    )
    print(f"[footprint] model_size_mb={model_size_mb:.3f}")
    if device.type == "cuda":
        print(f"[cuda] peak_memory_mb={peak_cuda_bytes / (1024 * 1024):.3f}")


if __name__ == "__main__":
    main()
