"""
Phase 1 training script:
Download -> load -> preprocess/batch -> split -> train for one epoch

For Phase 1 you will run epochs=1
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
from phase1.models_1dcnn import SimpleECG1DCNN


def set_seed(seed: int) -> None:
    """Deterministic-ish behavior for splits and initial weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(force_cpu: bool) -> torch.device:
    """
    Phase 1 defaults to CPU unless MPS (Apple Silicon) or CUDA is available.
    You can force CPU to reduce weirdness early on.
    """
    if force_cpu:
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Simple accuracy for sanity checks in Phase 1."""
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weights so minority classes contribute useful gradient.
    Normalized to mean 1.0 to keep the loss scale stable.
    """
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def make_weighted_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    """Oversample minority classes during training."""
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


def run_one_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float]:
    """
    One training epoch (one full iteration over the training dataloader).
    Phase 1 requirement: this must run without crashing.
    """
    model.train()

    loss_sum = 0.0
    acc_sum = 0.0
    batches = 0

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
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
) -> Tuple[float, float]:
    """One validation pass for sanity check."""
    model.eval()

    loss_sum = 0.0
    acc_sum = 0.0
    batches = 0

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

    return loss_sum / max(batches, 1), acc_sum / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser()

    # Dataset location
    parser.add_argument("--data-dir", type=str, required=True, help="Folder with training/ and REFERENCE.csv")

    # Phase 1 knobs (keep training fast)
    parser.add_argument("--target-len", type=int, default=3000, help="Trim/pad length. Smaller = faster.")
    parser.add_argument("--batch-size", type=int, default=32, help="Reduce if you run out of memory.")
    parser.add_argument("--subset", type=int, default=0, help="Use only first N samples for quick smoke tests.")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction.")

    # Training
    parser.add_argument("--epochs", type=int, default=1, help="Default 1 for Phase 1; increase later.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--balance",
        choices=("none", "weighted_loss", "weighted_sampler"),
        default="weighted_loss",
        help="Mitigate class imbalance. weighted_loss is the safest default here.",
    )

    # Repro + performance
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0 for Phase 1 simplicity on macOS.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if MPS/CUDA exists.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(force_cpu=bool(args.cpu))
    print(f"[device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_dir).expanduser().resolve()

    preprocess = PreprocessConfig(
        target_len=int(args.target_len),
        do_zscore=True,
    )

    dataset = PhysioNet2017Dataset(
        data_dir=data_dir,
        preprocess=preprocess,
        limit=int(args.subset) if args.subset and args.subset > 0 else None,
    )

    # Train/val split (Phase 1 requirement)
    val_size = int(len(dataset) * float(args.val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_labels = [dataset.labels[idx] for idx in train_ds.indices]
    val_labels = [dataset.labels[idx] for idx in val_ds.indices]
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

    # DataLoaders (Phase 1 requirement: batching works)
    train_loader = DataLoader(
        train_ds,
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

    # 1D CNN model (Phase 1 requirement)
    model = SimpleECG1DCNN(num_classes=4).to(device)

    # Standard classification setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    amp_enabled = device.type == "cuda"

    # Supports multiple epochs later, but for Phase 1 run epochs=1
    for epoch in range(int(args.epochs)):
        train_loss, train_acc = run_one_epoch_train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
        val_loss, val_acc = run_one_epoch_val(model, val_loader, criterion, device)

        print(
            f"[epoch {epoch+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
