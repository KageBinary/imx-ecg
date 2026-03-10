"""
Training script for the variable-length CNN + FFT model.

- Keeps original ECG length by default
- Pads only within each batch
- Uses masked global pooling in the model to ignore padding
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
from models_fft_gp import ECGFFTGlobalPoolNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def make_weighted_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    class_weights = compute_class_weights(labels, num_classes)
    sample_weights = class_weights[torch.tensor(labels, dtype=torch.long)]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(labels),
        replacement=True,
    )


def summarize_split(name: str, labels: List[int], num_classes: int) -> str:
    counts = Counter(labels)
    pieces = [f"class_{class_id}={counts.get(class_id, 0)}" for class_id in range(num_classes)]
    return f"[{name}] " + " ".join(pieces)


def stratified_split_indices(labels: List[int], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    """Split each class independently so validation keeps class coverage."""
    rng = random.Random(seed)
    buckets: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        buckets.setdefault(label, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for label in sorted(buckets):
        indices = buckets[label][:]
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_frac)))
        if val_count >= len(indices):
            val_count = len(indices) - 1
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def confusion_matrix_from_preds(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def format_confusion_matrix(cm: torch.Tensor) -> str:
    rows = ["[confusion_matrix] rows=true cols=pred"]
    for row in cm.tolist():
        rows.append(" ".join(f"{value:4d}" for value in row))
    return "\n".join(rows)


def compute_macro_f1_and_recall(cm: torch.Tensor) -> Tuple[float, List[float]]:
    recalls: List[float] = []
    f1s: List[float] = []
    for class_id in range(cm.shape[0]):
        tp = float(cm[class_id, class_id].item())
        fn = float(cm[class_id, :].sum().item() - tp)
        fp = float(cm[:, class_id].sum().item() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        recalls.append(recall)
        f1s.append(f1)
    macro_f1 = float(sum(f1s) / max(len(f1s), 1))
    return macro_f1, recalls


def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = [F.pad(x, (0, max_len - x.shape[-1])) for x in xs]
    x_batch = torch.stack(padded, dim=0)
    y_batch = torch.stack(ys, dim=0)
    return x_batch, y_batch, lengths


def run_one_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for x, y, lengths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x, lengths)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
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
    num_classes: int,
) -> Tuple[float, float, float, List[float], torch.Tensor]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    batches = 0
    amp_enabled = device.type == "cuda"
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for x, y, lengths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x, lengths)
            loss = criterion(logits, y)

        loss_sum += float(loss.item())
        acc_sum += accuracy(logits, y)
        batches += 1
        all_preds.append(torch.argmax(logits, dim=1).detach().cpu())
        all_targets.append(y.detach().cpu())

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    cm = confusion_matrix_from_preds(y_true, y_pred, num_classes=num_classes)
    macro_f1, recalls = compute_macro_f1_and_recall(cm)
    return loss_sum / max(batches, 1), acc_sum / max(batches, 1), macro_f1, recalls, cm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Folder with training/ and REFERENCE.csv")
    parser.add_argument(
        "--target-len",
        type=int,
        default=0,
        help="0 keeps original record lengths. Set a positive value to crop/pad before batching.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fft-bins", type=int, default=256)
    parser.add_argument(
        "--balance",
        choices=("none", "weighted_loss", "weighted_sampler"),
        default="weighted_loss",
        help="Mitigate class imbalance. weighted_loss is the safest default here.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience on macro-F1.")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/train_fft_gp_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(force_cpu=bool(args.cpu))
    print(f"[device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_dir).expanduser().resolve()
    preprocess = PreprocessConfig(
        target_len=int(args.target_len) if int(args.target_len) > 0 else None,
        do_zscore=True,
    )
    dataset = PhysioNet2017Dataset(
        data_dir=data_dir,
        preprocess=preprocess,
        limit=int(args.subset) if args.subset and args.subset > 0 else None,
    )

    train_indices, val_indices = stratified_split_indices(dataset.labels, float(args.val_frac), int(args.seed))
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    train_labels = [dataset.labels[idx] for idx in train_indices]
    val_labels = [dataset.labels[idx] for idx in val_indices]
    num_classes = 4

    print(summarize_split("train", train_labels, num_classes))
    print(summarize_split("val", val_labels, num_classes))

    sampler = None
    shuffle = True
    class_weights = None
    if args.balance == "weighted_loss":
        class_weights = compute_class_weights(train_labels, num_classes).to(device)
        print(f"[balance] weighted_loss class_weights={class_weights.detach().cpu().tolist()}")
    elif args.balance == "weighted_sampler":
        sampler = make_weighted_sampler(train_labels, num_classes)
        shuffle = False
        print("[balance] weighted_sampler")
    else:
        print("[balance] none")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(args.num_workers),
        drop_last=False,
        collate_fn=pad_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
        collate_fn=pad_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )

    model = ECGFFTGlobalPoolNet(num_classes=num_classes, fft_bins=int(args.fft_bins)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    amp_enabled = device.type == "cuda"
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = float("-inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(int(args.epochs)):
        train_start = time.perf_counter()
        train_loss, train_acc = run_one_epoch_train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
        train_time_s = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_loss, val_acc, val_macro_f1, val_recalls, val_cm = run_one_epoch_val(
            model,
            val_loader,
            criterion,
            device,
            num_classes=num_classes,
        )
        val_time_s = time.perf_counter() - val_start
        scheduler.step(val_macro_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        recall_text = " ".join(f"class_{i}_recall={score:.4f}" for i, score in enumerate(val_recalls))
        train_samples_per_s = len(train_ds) / max(train_time_s, 1e-9)
        val_samples_per_s = len(val_ds) / max(val_time_s, 1e-9)
        print(
            f"[epoch {epoch+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} lr={current_lr:.6f} "
            f"train_time_s={train_time_s:.2f} val_time_s={val_time_s:.2f} "
            f"train_samples_per_s={train_samples_per_s:.2f} val_samples_per_s={val_samples_per_s:.2f}"
        )
        print(f"[val_metrics] {recall_text}")
        print(format_confusion_matrix(val_cm))

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
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
            print(f"[checkpoint] saved_best path={checkpoint_path} epoch={best_epoch} macro_f1={best_macro_f1:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(args.patience):
                print(
                    f"[early_stop] no macro-F1 improvement for {epochs_without_improvement} epochs "
                    f"(best_epoch={best_epoch} best_macro_f1={best_macro_f1:.4f})"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    print(f"[best] epoch={best_epoch} macro_f1={best_macro_f1:.4f} checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
