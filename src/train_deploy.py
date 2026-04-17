"""
Training script for ECGDeployNet — the deployment-safe ECG classifier.

Key differences from train_fft_gp.py
--------------------------------------
* target_len is REQUIRED and fixed to deploy_config.CANONICAL_LEN (3000).
  Variable-length batching is not used; no pad_collate, no lengths tensor.
* The model takes a single input x: [B, 1, CANONICAL_LEN].
* Checkpoint saves the args so export_onnx.py can verify the shape contract.

Usage
-----
python src/train_deploy.py \\
    --data-dir data2017 \\
    --epochs 30 \\
    --batch-size 32 \\
    --balance weighted_loss \\
    --checkpoint-path checkpoints/deploy_best.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
from deploy_config import CANONICAL_LEN, CLASS_NAMES, NUM_CLASSES
from models_deploy import ECGDeployNet

# Re-use proven utilities from train_fft_gp rather than duplicating them.
from train_fft_gp import (
    accuracy,
    compute_class_weights,
    compute_macro_f1_and_recall,
    confusion_matrix_from_preds,
    format_confusion_matrix,
    make_weighted_sampler,
    pick_device,
    set_seed,
    stratified_split_indices,
    summarize_split,
)


class RandomWindowDataset(Dataset):
    """
    Training-only wrapper that applies random windowing to full-length ECG signals.

    Why this exists
    ---------------
    The deployment preprocessing uses a fixed center-crop to CANONICAL_LEN.
    During training, we can show the model random windows from longer recordings,
    which dramatically increases training variety and improves generalization —
    especially for the 'Other' class which has the most heterogeneous patterns.

    Preprocessing order matches deployment:
      1. Random window to CANONICAL_LEN (deployment uses center-crop instead)
      2. Z-score on the windowed signal

    The base dataset must be loaded with do_zscore=False and target_len=None
    so that raw cleaned signals of original length are returned.
    """

    def __init__(self, base: Dataset, window_len: int, labels: List[int]) -> None:
        self.base = base
        self.window_len = window_len
        self.labels = labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]  # x: [1, L], unnormalized, original length
        L = int(x.shape[-1])

        if L >= self.window_len:
            start = random.randint(0, L - self.window_len)
            x = x[:, start : start + self.window_len]
        else:
            # Pad short recordings — same as deployment center-pad
            pad = self.window_len - L
            x = F.pad(x, (pad // 2, pad - pad // 2))

        # Z-score after windowing — matches deployment order
        mu = x.mean()
        sigma = x.std()
        x = (x - mu) / (sigma + 1e-6)
        return x, y


def adjust_deploy_class_weights(class_weights: torch.Tensor) -> torch.Tensor:
    """Apply class-weighting tweak for ECGDeployNet.

    v1 (deploy_v2 run): Other×1.5 / Noisy cap 1.5
      → Other recall +18 pp but Normal dropped −9 pp (262 Normal→Other errors).
    v2 (deploy_v3 run): Other×1.2 / Noisy cap 2.0
      → Less aggressive Other push; Noisy cap raised slightly to avoid under-prediction.
    """
    adjusted = class_weights.clone()
    adjusted[2] = adjusted[2] * 1.2      # Other  (index 2): raw weight × 1.2
    adjusted[3] = torch.clamp(adjusted[3], max=2.0)  # Noisy (index 3): capped at 2.0
    return adjusted


def class_weight_summary(class_weights: torch.Tensor) -> Dict[str, float]:
    return {
        CLASS_NAMES[class_id]: float(class_weights[class_id].item())
        for class_id in range(NUM_CLASSES)
    }


def write_metrics_jsonl(metrics_path: Path, record: Dict[str, Any]) -> None:
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_one_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    loss_sum = acc_sum = 0.0
    batches = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.item())
        acc_sum += accuracy(logits.detach(), y)
        batches += 1
    return loss_sum / max(batches, 1), acc_sum / max(batches, 1)


@torch.no_grad()
def run_one_epoch_val(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, List[float], torch.Tensor]:
    model.eval()
    loss_sum = acc_sum = 0.0
    batches = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    amp_enabled = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += float(loss.item())
        acc_sum += accuracy(logits, y)
        batches += 1
        all_preds.append(torch.argmax(logits, dim=1).detach().cpu())
        all_targets.append(y.detach().cpu())
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    cm = confusion_matrix_from_preds(y_true, y_pred, num_classes=NUM_CLASSES)
    macro_f1, recalls = compute_macro_f1_and_recall(cm)
    return loss_sum / max(batches, 1), acc_sum / max(batches, 1), macro_f1, recalls, cm


def _load_config(parser: argparse.ArgumentParser) -> None:
    pre, _ = parser.parse_known_args()
    if getattr(pre, "config", None):
        with open(pre.config) as f:
            parser.set_defaults(**json.load(f))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ECGDeployNet with fixed-length input for deployment."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--target-len", type=int, default=CANONICAL_LEN,
        help=f"Fixed input length. Must match deploy_config.CANONICAL_LEN={CANONICAL_LEN}.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--balance",
        choices=("none", "weighted_loss", "weighted_sampler"),
        default="weighted_loss",
    )
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoints/deploy_best.pt",
    )
    parser.add_argument(
        "--channels", type=int, nargs=3, default=[32, 64, 128],
        metavar=("C1", "C2", "C3"),
        help="Output channels for the three conv blocks. Default: 32 64 128.",
    )
    parser.add_argument(
        "--temporal-kernel", type=int, default=25,
        help="Kernel size for the 4th long-range conv block (0 to disable). Default: 25.",
    )
    parser.add_argument(
        "--metrics-jsonl-path",
        type=str,
        default=None,
        help=(
            "Optional path for per-epoch JSONL metrics. "
            "Defaults to <checkpoint stem>.metrics.jsonl next to the checkpoint."
        ),
    )

    _load_config(parser)
    args = parser.parse_args()

    if args.data_dir is None:
        parser.error("--data-dir is required")

    if int(args.target_len) != CANONICAL_LEN:
        parser.error(
            f"--target-len must equal deploy_config.CANONICAL_LEN={CANONICAL_LEN} "
            f"to maintain export contract. Got {args.target_len}."
        )

    set_seed(args.seed)
    device = pick_device(force_cpu=bool(args.cpu))
    print(f"[device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_dir_path = Path(args.data_dir).expanduser().resolve()
    limit = int(args.subset) if args.subset > 0 else None

    # Use a fixed-length dataset purely to get labels for the split.
    label_ds = PhysioNet2017Dataset(
        data_dir=data_dir_path,
        preprocess=PreprocessConfig(target_len=CANONICAL_LEN, do_zscore=True),
        limit=limit,
    )
    train_indices, val_indices = stratified_split_indices(
        label_ds.labels, float(args.val_frac), int(args.seed)
    )
    train_labels = [label_ds.labels[i] for i in train_indices]
    val_labels = [label_ds.labels[i] for i in val_indices]

    # Training dataset: full-length signals, no z-score — RandomWindowDataset
    # applies random windowing + z-score per window (matches deployment order).
    train_base_ds = PhysioNet2017Dataset(
        data_dir=data_dir_path,
        preprocess=PreprocessConfig(target_len=None, do_zscore=False),
        limit=limit,
    )
    train_ds = RandomWindowDataset(
        base=Subset(train_base_ds, train_indices),
        window_len=CANONICAL_LEN,
        labels=train_labels,
    )

    # Validation dataset: fixed center-crop + z-score — identical to deployment.
    val_base_ds = PhysioNet2017Dataset(
        data_dir=data_dir_path,
        preprocess=PreprocessConfig(target_len=CANONICAL_LEN, do_zscore=True),
        limit=limit,
    )
    val_ds = Subset(val_base_ds, val_indices)
    print(f"[augment] train=random_window val=center_crop window_len={CANONICAL_LEN}")

    print(summarize_split("train", train_labels, NUM_CLASSES))
    print(summarize_split("val", val_labels, NUM_CLASSES))

    sampler = None
    shuffle = True
    class_weights = None
    if args.balance == "weighted_loss":
        raw_class_weights = compute_class_weights(train_labels, NUM_CLASSES)
        class_weights = adjust_deploy_class_weights(raw_class_weights).to(device)
        print(
            "[balance] weighted_loss "
            f"raw={class_weight_summary(raw_class_weights)} "
            f"adjusted={class_weight_summary(class_weights.detach().cpu())}"
        )
    elif args.balance == "weighted_sampler":
        sampler = make_weighted_sampler(train_labels, NUM_CLASSES)
        shuffle = False
        print("[balance] weighted_sampler")
    else:
        print("[balance] none")

    train_loader = DataLoader(
        train_ds,  # RandomWindowDataset — different window every epoch
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(args.num_workers),
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )

    model = ECGDeployNet(
        num_classes=NUM_CLASSES,
        channels=tuple(args.channels),
        temporal_kernel=int(args.temporal_kernel),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] ECGDeployNet total_params={total_params} "
        f"channels={args.channels} temporal_kernel={args.temporal_kernel}"
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=int(args.epochs) // 3, T_mult=2, eta_min=1e-5
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if args.metrics_jsonl_path:
        metrics_path = Path(args.metrics_jsonl_path).expanduser().resolve()
    else:
        metrics_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    print(f"[metrics] jsonl={metrics_path}")

    best_macro_f1 = float("-inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(int(args.epochs)):
        t0 = time.perf_counter()
        train_loss, train_acc = run_one_epoch_train(
            model, train_loader, optimizer, criterion, device,
            amp_enabled=amp_enabled, scaler=scaler, grad_clip=float(args.grad_clip),
        )
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_loss, val_acc, val_macro_f1, val_recalls, val_cm = run_one_epoch_val(
            model, val_loader, criterion, device,
        )
        val_time = time.perf_counter() - t1

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        recall_text = " ".join(
            f"class_{i}_recall={r:.4f}" for i, r in enumerate(val_recalls)
        )
        print(
            f"[epoch {epoch+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} lr={current_lr:.6f} "
            f"train_time_s={train_time:.2f} val_time_s={val_time:.2f}"
        )
        print(f"[val_metrics] {recall_text}")
        print(format_confusion_matrix(val_cm))

        checkpoint_saved = False
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "args": vars(args),
                    # Store deployment contract explicitly
                    "deploy": {
                        "model": "ECGDeployNet",
                        "canonical_len": CANONICAL_LEN,
                        "num_classes": NUM_CLASSES,
                        "channels": list(args.channels),
                        "temporal_kernel": int(args.temporal_kernel),
                        "input_shape": [1, 1, CANONICAL_LEN],
                        "preprocessing": "zscore_per_sample",
                    },
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
            checkpoint_saved = True
            print(
                f"[checkpoint] saved path={checkpoint_path} "
                f"epoch={best_epoch} macro_f1={best_macro_f1:.4f}"
            )
        else:
            epochs_without_improvement += 1

        write_metrics_jsonl(
            metrics_path,
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "lr": float(current_lr),
                "train_time_s": train_time,
                "val_time_s": val_time,
                "val_recalls": {
                    CLASS_NAMES[class_id]: float(val_recalls[class_id])
                    for class_id in range(NUM_CLASSES)
                },
                "confusion_matrix": val_cm.tolist(),
                "best_epoch_so_far": best_epoch,
                "best_macro_f1_so_far": best_macro_f1,
                "checkpoint_saved": checkpoint_saved,
            },
        )

        if not checkpoint_saved and epochs_without_improvement >= int(args.patience):
            print(
                f"[early_stop] no improvement for {epochs_without_improvement} epochs "
                f"(best_epoch={best_epoch} best_macro_f1={best_macro_f1:.4f})"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    print(
        f"[best] epoch={best_epoch} macro_f1={best_macro_f1:.4f} "
        f"checkpoint={checkpoint_path}"
    )


if __name__ == "__main__":
    main()
