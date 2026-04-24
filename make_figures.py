"""Generate clean presentation figures from deploy_tk25 training metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

METRICS = Path("checkpoints/deploy_tk25.metrics.jsonl")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

CLASS_NAMES = ["Normal", "AF", "Other", "Noisy"]

records = [json.loads(l) for l in METRICS.read_text().splitlines() if l.strip()]
epochs      = [r["epoch"]        for r in records]
train_loss  = [r["train_loss"]   for r in records]
val_loss    = [r["val_loss"]     for r in records]
val_f1      = [r["val_macro_f1"] for r in records]
recalls     = {cls: [r["val_recalls"][cls] for r in records] for cls in CLASS_NAMES}

best_idx  = int(np.argmax(val_f1))
best_ep   = epochs[best_idx]
best_f1   = val_f1[best_idx]
best_rec  = records[best_idx]

STYLE = dict(
    figure_facecolor="white",
    axes_facecolor="#f7f7f7",
    axes_spines_color="#cccccc",
    grid_color="#e0e0e0",
)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f7f7f7",
    "axes.edgecolor":   "#cccccc",
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
    "legend.framealpha": 0.9,
    "lines.linewidth":  2.2,
})

BLUE   = "#1a73e8"
ORANGE = "#fa7b17"
GREEN  = "#34a853"
PURPLE = "#9334e6"
RED    = "#ea4335"
GREY   = "#5f6368"

# ── Figure 1: Macro-F1 over epochs ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(epochs, val_f1, color=BLUE, label="Val Macro-F1", zorder=3)
ax.axvline(best_ep, color=GREY, linestyle="--", linewidth=1.2, zorder=2)
ax.scatter([best_ep], [best_f1], color=BLUE, s=90, zorder=4)
ax.annotate(
    f" Best: {best_f1:.3f}\n Epoch {best_ep}",
    xy=(best_ep, best_f1), xytext=(best_ep + 4, best_f1 - 0.04),
    fontsize=12, color=GREY,
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Macro-F1")
ax.set_title("ECGDeployNet — Validation Macro-F1")
ax.set_xlim(1, epochs[-1])
ax.set_ylim(0.3, 0.75)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
fig.tight_layout()
fig.savefig(OUT / "f1_curve.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Train vs Val loss ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(epochs, train_loss, color=BLUE,   label="Train loss")
ax.plot(epochs, val_loss,   color=ORANGE, label="Val loss")
ax.axvline(best_ep, color=GREY, linestyle="--", linewidth=1.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("ECGDeployNet — Training & Validation Loss")
ax.set_xlim(1, epochs[-1])
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "loss_curves.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: Per-class recall ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

colors = [BLUE, RED, GREEN, ORANGE]
for cls, color in zip(CLASS_NAMES, colors):
    ax.plot(epochs, recalls[cls], color=color, label=cls)
ax.axvline(best_ep, color=GREY, linestyle="--", linewidth=1.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Recall")
ax.set_title("ECGDeployNet — Per-Class Recall")
ax.set_xlim(1, epochs[-1])
ax.set_ylim(0, 1.05)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "recall_curves.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Figure 4: Confusion matrix at best epoch ─────────────────────────────────
cm = np.array(best_rec["confusion_matrix"])
cm_pct = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_pct, annot=True, fmt=".1%",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    cmap="Blues", vmin=0, vmax=1,
    linewidths=0.5, linecolor="#cccccc",
    ax=ax, cbar_kws={"format": "%.0%%"},
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — Epoch {best_ep}  (Macro-F1 {best_f1:.3f})")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("Saved to figures/:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}")
