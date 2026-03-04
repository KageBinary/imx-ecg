"""
Phase 1 training script:
Download -> load -> preprocess/batch -> split -> train for one epoch

For Phase 1 you will run epochs=1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset_physionet2017 import PhysioNet2017Dataset, PreprocessConfig
from models_1dcnn import SimpleECG1DCNN


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


def run_one_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    One training epoch (one full iteration over the training dataloader).
    Phase 1 requirement: this must run without crashing.
    """
    model.train()

    loss_sum = 0.0
    acc_sum = 0.0
    batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

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

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

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

    # Repro + performance
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0 for Phase 1 simplicity on macOS.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if MPS/CUDA exists.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(force_cpu=bool(args.cpu))
    print(f"[device] {device}")

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

    # DataLoaders (Phase 1 requirement: batching works)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    # 1D CNN model (Phase 1 requirement)
    model = SimpleECG1DCNN(num_classes=4).to(device)

    # Standard classification setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # Supports multiple epochs later, but for Phase 1 run epochs=1
    for epoch in range(int(args.epochs)):
        train_loss, train_acc = run_one_epoch_train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = run_one_epoch_val(model, val_loader, criterion, device)

        print(
            f"[epoch {epoch+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
