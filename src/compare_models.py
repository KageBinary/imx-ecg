"""
Compare ECG models side-by-side on the test set.

To add a new model, just add an entry to the MODELS list below.

Usage:
    python src/compare_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# =============================================================================
# CONFIGURE YOUR MODELS HERE
# Each entry needs: name, module, class_name, checkpoint, input_length,
# and needs_lengths (True if forward() takes a lengths arg)
# =============================================================================
MODELS = [
    {
        "name": "ECGCNN",
        "module": "ecgcnn.model",
        "class_name": "ECGCNN",
        "checkpoint": "outputs/models/best_model.pth",
        "input_length": 10000,
        "needs_lengths": False,
    },
    {
        "name": "ECGFFTGlobalPoolNet",
        "module": "fft_gp.models_fft_gp",
        "class_name": "ECGFFTGlobalPoolNet",
        "checkpoint": "checkpoints/train_fft_gp_best.pt",
        "input_length": 10000,
        "needs_lengths": True,
    },
    # --- Add new models here ---
    # {
    #     "name": "MyNewModel",
    #     "module": "my_folder.my_model",
    #     "class_name": "MyNewModel",
    #     "checkpoint": "checkpoints/my_new_model.pt",
    #     "input_length": 10000,
    #     "needs_lengths": False,
    # },
]

LABEL_NAMES = ["Normal (N)", "AFib (A)", "Other (O)", "Noisy (~)"]
TEST_DATA = "data/processed/X_test.npy"
TEST_LABELS = "data/processed/y_test.npy"


def load_model(cfg: dict) -> torch.nn.Module:
    import importlib
    mod = importlib.import_module(cfg["module"])
    cls = getattr(mod, cfg["class_name"])

    try:
        model = cls(num_classes=4)
    except TypeError:
        model = cls()

    sd = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def run_inference(model: torch.nn.Module, X: np.ndarray, cfg: dict) -> np.ndarray:
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = torch.from_numpy(X[i:i + 64]).float()
            if batch.ndim == 2:
                batch = batch.unsqueeze(1)
            if cfg["needs_lengths"]:
                lengths = torch.full((batch.shape[0],), batch.shape[-1])
                logits = model(batch, lengths)
            else:
                logits = model(batch)
            all_preds.append(torch.argmax(logits, dim=1).numpy())
    return np.concatenate(all_preds)


def print_confusion_matrix(cm: np.ndarray) -> None:
    header = f"{'':>12}" + "".join(f"{'Pred '+l.split('(')[1].strip(')'):>10}" for l in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{LABEL_NAMES[i]:>12}" + "".join(f"{v:>10}" for v in row))


def main() -> None:
    X_test = np.load(TEST_DATA)
    y_test = np.load(TEST_LABELS)

    print(f"Test set: {X_test.shape[0]} samples")
    for i, label in enumerate(LABEL_NAMES):
        print(f"  {label}: {(y_test == i).sum()}")
    print()

    # Run each model
    model_results: list[dict] = []
    for cfg in MODELS:
        ckpt = Path(cfg["checkpoint"])
        if not ckpt.exists():
            print(f"[SKIP] {cfg['name']} — checkpoint not found: {ckpt}")
            print()
            continue

        print("=" * 60)
        print(f"  {cfg['name']}")
        print("=" * 60)

        model = load_model(cfg)
        params = sum(p.numel() for p in model.parameters())
        preds = run_inference(model, X_test, cfg)
        cm = confusion_matrix(y_test, preds)

        print(f"Parameters: {params:,}")
        print()
        print(classification_report(y_test, preds, target_names=LABEL_NAMES, digits=3, zero_division=0))
        print("Confusion Matrix:")
        print_confusion_matrix(cm)
        print()

        model_results.append({
            "name": cfg["name"],
            "preds": preds,
            "cm": cm,
            "accuracy": np.mean(preds == y_test),
            "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, preds, average="weighted", zero_division=0),
            "params": params,
        })

    if len(model_results) < 2:
        print("Need at least 2 models to compare.")
        return

    # Head-to-head comparison
    print("=" * 60)
    print("  HEAD-TO-HEAD COMPARISON")
    print("=" * 60)

    # Overall metrics table
    names = [r["name"] for r in model_results]
    name_width = max(len(n) for n in names)

    print()
    header = f"{'Metric':>15}" + "".join(f"{n:>{name_width + 2}}" for n in names) + f"{'Winner':>{name_width + 4}}"
    print(header)
    print("-" * len(header))

    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        values = [r[metric] for r in model_results]
        best_idx = int(np.argmax(values))
        row = f"{metric:>15}"
        for v in values:
            row += f"{v:>{name_width + 2}.3f}"
        row += f"{names[best_idx]:>{name_width + 4}}"
        print(row)

    # Per-class recall
    print()
    print("Per-class recall:")
    for i, label in enumerate(LABEL_NAMES):
        row = f"{label:>15}"
        recalls = []
        for r in model_results:
            total = r["cm"][i].sum()
            recall = r["cm"][i][i] / total if total > 0 else 0
            recalls.append(recall)
            row += f"{recall:>{name_width + 2}.1%}"
        best_idx = int(np.argmax(recalls))
        row += f"{names[best_idx]:>{name_width + 4}}"
        print(row)

    # Per-class precision
    print()
    print("Per-class precision:")
    for i, label in enumerate(LABEL_NAMES):
        row = f"{label:>15}"
        precisions = []
        for r in model_results:
            col_sum = r["cm"][:, i].sum()
            prec = r["cm"][i][i] / col_sum if col_sum > 0 else 0
            precisions.append(prec)
            row += f"{prec:>{name_width + 2}.1%}"
        best_idx = int(np.argmax(precisions))
        row += f"{names[best_idx]:>{name_width + 4}}"
        print(row)

    # Disagreement analysis (pairwise for first two models)
    print()
    print("Disagreement analysis:")
    for i in range(len(model_results)):
        for j in range(i + 1, len(model_results)):
            a, b = model_results[i], model_results[j]
            disagree = a["preds"] != b["preds"]
            both_right = (a["preds"] == y_test) & (b["preds"] == y_test)
            both_wrong = (a["preds"] != y_test) & (b["preds"] != y_test)
            only_a = (a["preds"] == y_test) & (b["preds"] != y_test)
            only_b = (a["preds"] != y_test) & (b["preds"] == y_test)

            print(f"\n  {a['name']} vs {b['name']}:")
            print(f"    Disagree:          {disagree.sum():>4} / {len(y_test)} ({disagree.mean():.1%})")
            print(f"    Both correct:      {both_right.sum():>4} ({both_right.mean():.1%})")
            print(f"    Both wrong:        {both_wrong.sum():>4} ({both_wrong.mean():.1%})")
            print(f"    Only {a['name']:} right: {only_a.sum():>4} ({only_a.mean():.1%})")
            print(f"    Only {b['name']:} right: {only_b.sum():>4} ({only_b.mean():.1%})")


if __name__ == "__main__":
    main()
