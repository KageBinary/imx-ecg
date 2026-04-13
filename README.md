# imx-ecg

ECG classification experiments for the PhysioNet/CinC Challenge 2017 dataset.

This repository contains two model paths:

1. A small fixed-length 1D CNN baseline for quick sanity checks.
2. A variable-length CNN + FFT model that keeps original record length, pads only inside each batch, and fuses time-domain and frequency-domain features.

The code is written as plain Python scripts under `src/`. There is no package wrapper, so you run everything from the repository root with commands like `python src/train_fft_gp.py ...`.

---

## What This Repo Does

Given single-lead ECG records stored as `.mat` files plus labels from `REFERENCE.csv`, the code:

- loads the PhysioNet 2017 training split from disk
- normalizes each record
- optionally crops or pads records to a fixed length
- trains either a baseline CNN or a variable-length FFT model
- evaluates a saved checkpoint on a validation split
- benchmarks inference latency and footprint

Class mapping (official 2017 challenge labels):

| Label | Class ID |
|---|---|
| `N` (Normal) | 0 |
| `A` (AF) | 1 |
| `O` (Other) | 2 |
| `~` (Noisy) | 3 |

---

## Requirements

### Software

- Python 3.10 or newer
- `pip`
- a virtual environment tool such as `venv`

### Python packages

- `torch`
- `numpy`
- `pandas`
- `scipy`

Tested with:

- Python `3.12.3`
- PyTorch `2.6.0+cu124`
- NumPy `2.3.5`
- pandas `3.0.1`
- SciPy `1.17.1`

---

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pandas scipy
```

---

## Dataset Layout

```text
data2017/
  REFERENCE.csv
  training/
    A00001.mat
    A00001.hea
    A00002.mat
    A00002.hea
    ...
```

- `REFERENCE.csv` — no header, columns are `record,label`
- `training/*.mat` — single-lead ECG signals under the `val` key
- `training/*.hea` — present but not read by the loader

---

## Quick Start

All commands assume you are in the repository root with the virtual environment active and the dataset at `data2017/`.

### 1. Baseline smoke test

Fastest way to verify the environment. Uses a fixed-length 1D CNN and defaults to one epoch.

```bash
python src/train_phase1.py \
  --data-dir data2017 \
  --epochs 1 \
  --batch-size 32 \
  --target-len 3000 \
  --balance weighted_loss
```

### 2. Train the FFT model

```bash
python src/train_fft_gp.py \
  --data-dir data2017 \
  --epochs 20 \
  --batch-size 32 \
  --fft-bins 256 \
  --balance weighted_loss \
  --checkpoint-path checkpoints/train_fft_gp_best.pt
```

Key training flags:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--grad-clip` | 1.0 | Gradient norm clipping (0 disables) |
| `--fft-bins` | 256 | FFT feature bins |
| `--balance` | `weighted_loss` | `none`, `weighted_loss`, or `weighted_sampler` |
| `--patience` | 7 | Early-stop epochs without macro-F1 improvement |
| `--val-frac` | 0.2 | Fraction of data held out for validation |
| `--target-len` | 0 | 0 = keep original lengths; positive value crops/pads |
| `--subset` | 0 | 0 = full dataset; N = use first N records only |
| `--num-workers` | 0 | DataLoader worker processes |
| `--cpu` | off | Force CPU |
| `--checkpoint-path` | `checkpoints/train_fft_gp_best.pt` | Where to save the best checkpoint |

### 3. Evaluate a saved checkpoint

```bash
python src/eval_fft_gp.py \
  --data-dir data2017 \
  --checkpoint-path checkpoints/train_fft_gp_best.pt \
  --batch-size 1
```

Prints accuracy, macro-F1, per-class recall, confusion matrix, latency, throughput, and model size on disk. `--batch-size 1` gives single-sample edge-style latency; larger values show throughput.

### 4. Synthetic inference benchmark

```bash
python src/benchmark_fft_gp.py \
  --checkpoint-path checkpoints/train_fft_gp_best.pt \
  --batch-size 1 \
  --min-len 2500 \
  --max-len 9000 \
  --timed-batches 200
```

No dataset required. Generates random variable-length signals and measures mean, p50, and p95 forward-pass latency plus throughput.

---

## Smoke Tests (end-to-end, CPU, fast)

```bash
python src/train_phase1.py --data-dir data2017 --subset 32 --epochs 1 --batch-size 8 --cpu

python src/train_fft_gp.py --data-dir data2017 --subset 64 --epochs 1 --batch-size 8 --cpu \
  --checkpoint-path /tmp/smoke_fft_gp.pt

python src/eval_fft_gp.py --data-dir data2017 \
  --checkpoint-path /tmp/smoke_fft_gp.pt \
  --subset 64 --batch-size 1 --cpu --warmup-batches 2

python src/benchmark_fft_gp.py --checkpoint-path /tmp/smoke_fft_gp.pt \
  --batch-size 1 --min-len 2500 --max-len 4000 \
  --warmup-batches 2 --timed-batches 5 --cpu
```

---

## Expected Console Output

Metrics from small runs are not meaningful — these show the expected format only.

### Training

```text
[device] cpu
[train] class_0=32 class_1=5 class_2=12 class_3=2
[val] class_0=8 class_1=1 class_2=3 class_3=1
[balance] weighted_loss class_weights=[0.153, 0.982, 0.409, 2.455]
[epoch 1/20] train_loss=1.4154 train_acc=0.0536 val_loss=1.3934 val_acc=0.1250 val_macro_f1=0.0667 lr=0.001000 train_time_s=1.23 val_time_s=0.03 train_samples_per_s=41.39 val_samples_per_s=400.13
[val_metrics] class_0_recall=0.0000 class_1_recall=0.0000 class_2_recall=0.6667 class_3_recall=0.0000
[confusion_matrix] rows=true cols=pred
   0    0    8    0
   0    0    1    0
   0    1    2    0
   0    0    1    0
[checkpoint] saved_best path=checkpoints/train_fft_gp_best.pt epoch=1 macro_f1=0.0667
[best] epoch=1 macro_f1=0.0667 checkpoint=checkpoints/train_fft_gp_best.pt
```

### Evaluation

```text
[device] cpu
[checkpoint] path=checkpoints/train_fft_gp_best.pt
[quality] accuracy=0.1538 macro_f1=0.0667 class_0_recall=0.0000 class_1_recall=0.0000 class_2_recall=0.6667 class_3_recall=0.0000
[confusion_matrix] rows=true cols=pred
   0    0    8    0
   0    0    1    0
   0    1    2    0
   0    0    1    0
[performance] batch_size=1 avg_latency_ms=1.837 throughput_samples_per_s=544.323
[footprint] model_size_mb=0.672
```

### Benchmark

```text
[device] cpu
[checkpoint] path=checkpoints/train_fft_gp_best.pt
[model] total_params=57316 trainable_params=57316 checkpoint_size_mb=0.672 fft_bins=256
[benchmark] batch_size=1 min_len=2500 max_len=9000 warmup_batches=10 timed_batches=200 amp=False
[latency] mean_ms=5.007 p50_ms=2.492 p95_ms=12.024 throughput_samples_per_s=199.735
```

---

## Code Layout

### Top-level

| Path | Description |
|---|---|
| `README.md` | This file |
| `data2017/` | Dataset root — `REFERENCE.csv` + `training/*.mat` |
| `checkpoints/` | Saved `.pt` model checkpoints |
| `src/` | All source scripts |

### `src/`

| File | Description |
|---|---|
| `dataset_physionet2017.py` | Loads `.mat` records, maps labels, cleans NaN/Inf, center-crops or pads to target length, z-score normalizes, returns `(1, L)` tensors |
| `models_1dcnn.py` | `SimpleECG1DCNN` — small fixed-length baseline used by `train_phase1.py` |
| `train_phase1.py` | Phase 1 training script for the baseline CNN; random split, loss + accuracy reporting |
| `models_fft_gp.py` | `ECGFFTGlobalPoolNet` — time branch (Conv/BN/ReLU/Pool + masked global avg pool) fused with FFT branch (batched rfft + compact spectral encoder) |
| `train_fft_gp.py` | Main training script: variable-length padded batching, stratified split, macro-F1, confusion matrix, LR scheduling, early stopping, checkpoint saving |
| `eval_fft_gp.py` | Loads a checkpoint, rebuilds the validation split, reports quality metrics and real-data inference speed |
| `benchmark_fft_gp.py` | Loads a checkpoint, measures forward-pass latency/throughput with synthetic inputs, no dataset required |

---

## Model Architecture (`ECGFFTGlobalPoolNet`)

```
Input: (B, 1, L)  — variable-length, zero-padded within each batch

Time branch
  Conv1d(1→16, k=7) → BN → ReLU → MaxPool1d(2)
  Conv1d(16→32, k=5) → BN → ReLU → MaxPool1d(2)
  Conv1d(32→64, k=5) → BN → ReLU → MaxPool1d(2)
  Masked global average pool  → (B, 64)

Frequency branch
  rfft(x) → |·| → log1p → AdaptiveAvgPool1d(fft_bins) → squeeze
  Linear(fft_bins→128) → ReLU  → (B, 128)

Fusion head
  Linear(192→128) → ReLU → Dropout(0.3) → Linear(128→4)
```

The masked global average pool ignores zero-padded timesteps so records of different lengths can be batched without the padding distorting the time-branch features.

---

## Checkpoints

The training script saves the best checkpoint (by validation macro-F1) as a dict:

```python
{
    "epoch":               int,
    "model_state_dict":    dict,
    "optimizer_state_dict": dict,
    "best_macro_f1":       float,
    "args":                dict,   # all CLI args used for that run
}
```

Default save path: `checkpoints/train_fft_gp_best.pt`

---

## Common Patterns

```bash
# CPU-only training
python src/train_fft_gp.py --data-dir data2017 --cpu

# Small subset before committing to a full run
python src/train_fft_gp.py --data-dir data2017 --subset 256 --epochs 2 --batch-size 8 --cpu

# Fixed-length mode (crop/pad to 3000 samples)
python src/train_fft_gp.py --data-dir data2017 --target-len 3000

# Disable gradient clipping
python src/train_fft_gp.py --data-dir data2017 --grad-clip 0

# Larger batch benchmark
python src/benchmark_fft_gp.py --checkpoint-path checkpoints/train_fft_gp_best.pt --batch-size 8
```

---

## Troubleshooting

**`python: command not found`**
Activate the venv: `source .venv/bin/activate`

**`Missing folder: .../training` or `Missing file: .../REFERENCE.csv`**
The `--data-dir` path is wrong or the dataset is not extracted in the expected layout.

**`Missing checkpoint: ...`**
Train first, or point `--checkpoint-path` at an existing `.pt` file.

**CUDA warnings on a CPU machine**
Pass `--cpu`. If warnings persist, install a CPU-only PyTorch build.

**Training is slow or runs out of memory**
Try: smaller `--batch-size`, `--subset N`, `--target-len 3000`, `--num-workers 0`, or `--cpu`.

---

## Recommended Workflow

1. `train_phase1.py` on a small `--subset` to verify the environment loads cleanly.
2. `train_fft_gp.py` for real training — monitor `val_macro_f1` and the confusion matrix.
3. `eval_fft_gp.py` to inspect final quality and per-class recall on the validation split.
4. `benchmark_fft_gp.py` to measure inference speed before deploying to target hardware.
