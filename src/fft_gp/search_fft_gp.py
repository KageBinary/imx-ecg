"""
Random hyperparameter search for ECGFFTGlobalPoolNet.

Samples configs from a defined search space, runs a short training trial for
each, then ranks them by validation macro-F1 and saves all results to a JSON file.

Usage
-----
python src/search_fft_gp.py \
    --data-dir data2017 \
    --trials 30 \
    --trial-epochs 40 \
    --trial-patience 7 \
    --results-path search_results.json

After the search completes, the best config is printed and can be pasted
directly into train_fft_gp.py for a full training run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
from fft_gp.models_fft_gp import ECGFFTGlobalPoolNet
from fft_gp.train_fft_gp import (
    compute_class_weights,
    make_weighted_sampler,
    pad_collate,
    pick_device,
    run_one_epoch_train,
    run_one_epoch_val,
    set_seed,
    stratified_split_indices,
    summarize_split,
)

NUM_CLASSES = 4

# ---------------------------------------------------------------------------
# Search space definition
# Each entry is either a list of choices or a ("log_uniform", lo, hi) tuple.
# ---------------------------------------------------------------------------
SEARCH_SPACE: Dict[str, Any] = {
    # architecture
    "time_channels": [
        [16, 32, 64],
        [32, 64, 128],
        [16, 32, 64, 128],
        [32, 64, 128, 256],
        [64, 128, 256],
        [16, 32],
        [32, 64],
    ],
    "time_kernels_single": [3, 5, 7],   # broadcast to all blocks
    "pool_stride": [2],
    "fft_bins": [64, 128, 256, 512],
    "freq_hidden": [64, 128, 256],
    "head_hidden": [64, 128, 256],
    "dropout": [0.1, 0.2, 0.3, 0.5],
    # training
    "lr": ("log_uniform", 1e-4, 1e-2),
    "weight_decay": ("log_uniform", 1e-5, 1e-2),
    "batch_size": [16, 32, 64],
}


def sample_config(rng: random.Random) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key, space in SEARCH_SPACE.items():
        if isinstance(space, (list, tuple)) and isinstance(space[0], str) and space[0] == "log_uniform":
            _, lo, hi = space
            cfg[key] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            cfg[key] = rng.choice(space)
    return cfg


def build_model(cfg: Dict[str, Any]) -> ECGFFTGlobalPoolNet:
    return ECGFFTGlobalPoolNet(
        num_classes=NUM_CLASSES,
        fft_bins=int(cfg["fft_bins"]),
        time_channels=cfg["time_channels"],
        time_kernels=int(cfg["time_kernels_single"]),
        pool_stride=int(cfg["pool_stride"]),
        freq_hidden=int(cfg["freq_hidden"]),
        head_hidden=int(cfg["head_hidden"]),
        dropout=float(cfg["dropout"]),
    )


def run_trial(
    cfg: Dict[str, Any],
    train_ds: Subset,
    val_ds: Subset,
    train_labels: List[int],
    device: torch.device,
    epochs: int,
    patience: int,
    num_workers: int,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    amp_enabled = device.type == "cuda"

    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
    sampler = make_weighted_sampler(train_labels, NUM_CLASSES)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=pad_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=pad_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience // 2, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_f1 = float("-inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run += 1
        run_one_epoch_train(
            model, train_loader, optimizer, criterion, device,
            amp_enabled=amp_enabled, scaler=scaler, grad_clip=1.0,
        )
        _, _, macro_f1, _, _ = run_one_epoch_val(
            model, val_loader, criterion, device, num_classes=NUM_CLASSES,
        )
        scheduler.step(macro_f1)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Re-evaluate best state for final recall scores
    if best_state is not None:
        model.load_state_dict(best_state)
    _, _, macro_f1, recalls, _ = run_one_epoch_val(
        model, val_loader, criterion, device, num_classes=NUM_CLASSES,
    )

    return {
        "cfg": cfg,
        "best_macro_f1": float(macro_f1),
        "per_class_recall": [round(float(r), 4) for r in recalls],
        "total_params": total_params,
        "epochs_run": epochs_run,
    }


def _load_config(parser: argparse.ArgumentParser) -> None:
    pre, _ = parser.parse_known_args()
    if getattr(pre, "config", None):
        with open(pre.config) as f:
            overrides = json.load(f)
        parser.set_defaults(**overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random hyperparameter search for ECGFFTGlobalPoolNet")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--trials", type=int, default=20, help="Number of random configs to try.")
    parser.add_argument(
        "--trial-epochs", type=int, default=40,
        help="Max training epochs per trial. Early stopping will cut short configs "
             "that stop improving before this limit.",
    )
    parser.add_argument(
        "--trial-patience", type=int, default=7,
        help="Early-stop a trial after this many epochs with no macro-F1 improvement.",
    )
    parser.add_argument("--subset", type=int, default=0, help="Limit dataset size (0 = full).")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--results-path", type=str, default="search_results.json",
        help="Path to save all trial results as JSON.",
    )

    _load_config(parser)
    args = parser.parse_args()

    if args.data_dir is None:
        parser.error("--data-dir is required (either via CLI or in --config)")

    device = pick_device(force_cpu=bool(args.cpu))
    print(f"[device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_dir).expanduser().resolve()
    preprocess = PreprocessConfig(target_len=None, do_zscore=True)
    dataset = PhysioNet2017Dataset(
        data_dir=data_dir,
        preprocess=preprocess,
        limit=int(args.subset) if args.subset > 0 else None,
    )

    train_indices, val_indices = stratified_split_indices(
        dataset.labels, float(args.val_frac), int(args.seed)
    )
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    train_labels = [dataset.labels[i] for i in train_indices]
    val_labels = [dataset.labels[i] for i in val_indices]

    print(summarize_split("train", train_labels, NUM_CLASSES))
    print(summarize_split("val", val_labels, NUM_CLASSES))
    print(f"[search] trials={args.trials} trial_epochs={args.trial_epochs}")

    rng = random.Random(args.seed)
    results: List[Dict[str, Any]] = []

    for trial_idx in range(args.trials):
        cfg = sample_config(rng)
        trial_seed = args.seed + trial_idx + 1
        t0 = time.perf_counter()

        try:
            result = run_trial(
                cfg=cfg,
                train_ds=train_ds,
                val_ds=val_ds,
                train_labels=train_labels,
                device=device,
                epochs=int(args.trial_epochs),
                patience=int(args.trial_patience),
                num_workers=int(args.num_workers),
                seed=trial_seed,
            )
            elapsed = time.perf_counter() - t0
            result["trial"] = trial_idx
            result["elapsed_s"] = round(elapsed, 2)
            results.append(result)
            stopped_early = result["epochs_run"] < args.trial_epochs
            print(
                f"[trial {trial_idx + 1}/{args.trials}] "
                f"macro_f1={result['best_macro_f1']:.4f} "
                f"epochs={result['epochs_run']}/{args.trial_epochs}"
                f"{'  (early stop)' if stopped_early else ''} "
                f"params={result['total_params']} "
                f"time={elapsed:.1f}s "
                f"cfg={_fmt_cfg(cfg)}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"[trial {trial_idx + 1}/{args.trials}] FAILED ({exc}) cfg={_fmt_cfg(cfg)}")
            results.append({"trial": trial_idx, "error": str(exc), "cfg": cfg, "elapsed_s": round(elapsed, 2)})

    # Sort by macro-F1
    valid = [r for r in results if "best_macro_f1" in r]
    valid.sort(key=lambda r: r["best_macro_f1"], reverse=True)

    results_path = Path(args.results_path).expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[search] results saved to {results_path}")

    if not valid:
        print("[search] no successful trials")
        return

    print("\n--- Top 5 configs by macro-F1 ---")
    for rank, r in enumerate(valid[:5], 1):
        print(
            f"  #{rank}  macro_f1={r['best_macro_f1']:.4f}  "
            f"epochs={r['epochs_run']}  "
            f"recall={r['per_class_recall']}  params={r['total_params']}  "
            f"cfg={_fmt_cfg(r['cfg'])}"
        )

    best = valid[0]
    best_config_path = results_path.with_name(results_path.stem + "_best.json")
    _save_train_config(best["cfg"], args, best_config_path)
    print(f"\n[search] best config saved to {best_config_path}")
    print("  Use it with:")
    print(f"    python src/train_fft_gp.py --config {best_config_path} --epochs 50")
    print("\n--- Best config (CLI equivalent) ---")
    _print_train_command(best["cfg"], args)


def _fmt_cfg(cfg: Dict[str, Any]) -> str:
    parts = [
        f"ch={cfg['time_channels']}",
        f"k={cfg['time_kernels_single']}",
        f"fft={cfg['fft_bins']}",
        f"fh={cfg['freq_hidden']}",
        f"hh={cfg['head_hidden']}",
        f"do={cfg['dropout']:.2f}",
        f"lr={cfg['lr']:.2e}",
        f"wd={cfg['weight_decay']:.2e}",
        f"bs={cfg['batch_size']}",
    ]
    return " ".join(parts)


def _save_train_config(cfg: Dict[str, Any], search_args: argparse.Namespace, path: Path) -> None:
    """Save the best search config as a JSON readable by train_fft_gp.py --config."""
    train_cfg = {
        "data_dir": search_args.data_dir,
        "time_channels": cfg["time_channels"],
        "time_kernels": [cfg["time_kernels_single"]],
        "pool_stride": int(cfg["pool_stride"]),
        "fft_bins": int(cfg["fft_bins"]),
        "freq_hidden": int(cfg["freq_hidden"]),
        "head_hidden": int(cfg["head_hidden"]),
        "dropout": round(float(cfg["dropout"]), 4),
        "lr": round(float(cfg["lr"]), 8),
        "weight_decay": round(float(cfg["weight_decay"]), 8),
        "batch_size": int(cfg["batch_size"]),
        "balance": "weighted_loss",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(train_cfg, f, indent=2)


def _print_train_command(cfg: Dict[str, Any], search_args: argparse.Namespace) -> None:
    channels = " ".join(str(c) for c in cfg["time_channels"])
    print(
        f"python src/train_fft_gp.py \\\n"
        f"  --data-dir {search_args.data_dir} \\\n"
        f"  --time-channels {channels} \\\n"
        f"  --time-kernels {cfg['time_kernels_single']} \\\n"
        f"  --pool-stride {cfg['pool_stride']} \\\n"
        f"  --fft-bins {cfg['fft_bins']} \\\n"
        f"  --freq-hidden {cfg['freq_hidden']} \\\n"
        f"  --head-hidden {cfg['head_hidden']} \\\n"
        f"  --dropout {cfg['dropout']:.2f} \\\n"
        f"  --lr {cfg['lr']:.2e} \\\n"
        f"  --weight-decay {cfg['weight_decay']:.2e} \\\n"
        f"  --batch-size {cfg['batch_size']} \\\n"
        f"  --epochs 50 \\\n"
        f"  --balance weighted_loss"
    )


if __name__ == "__main__":
    main()
