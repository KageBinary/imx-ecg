# imx-ecg

ECG classification experiments for the PhysioNet/CinC Challenge 2017 dataset.

This repository currently contains two model paths:

1. A small fixed-length 1D CNN baseline for quick sanity checks.
2. A variable-length CNN + FFT model that keeps original record length, pads only inside each batch, and fuses time-domain and frequency-domain features.

The code is written as plain Python scripts under `src/`. There is no package wrapper yet, so you run everything from the repository root with commands like `python src/train_fft_gp.py ...`.

## What This Repo Does

Given single-lead ECG records stored as `.mat` files plus labels from `REFERENCE.csv`, the code:

- loads the PhysioNet 2017 training split from disk
- normalizes each record
- optionally crops or pads records to a fixed length
- trains either a baseline CNN or a variable-length FFT model
- evaluates a saved checkpoint on a validation split
- benchmarks inference latency and footprint

The class mapping used by the code is:

- `N -> 0`
- `A -> 1`
- `O -> 2`
- `~ -> 3`

These are the official 2017 challenge labels.

## What Needs To Be Installed

### Required software

- Python 3.10 or newer
- `pip`
- a virtual environment tool such as `venv`

### Python packages

Minimum runtime dependencies:

- `torch`
- `numpy`
- `pandas`
- `scipy`

Tested locally with:

- Python `3.12.3`
- PyTorch `2.6.0+cu124`
- NumPy `2.3.5`
- pandas `3.0.1`
- SciPy `1.17.1`

If you want GPU acceleration, install the PyTorch build that matches your system. If you only want CPU training/evaluation, a CPU-only PyTorch install is fine.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pandas scipy
```

If your machine does not provide a `python` command until the virtual environment is activated, that is normal. After activation, `python` should resolve to `.venv/bin/python`.

## Dataset Layout

The scripts expect the PhysioNet/CinC Challenge 2017 training data in this shape:

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

Important details:

- `REFERENCE.csv` is required.
- `training/*.mat` is required.
- `training/*.hea` can be present, but the current loader does not read it.
- `REFERENCE.csv` is expected to have no header and to contain `record,label`.

## Quick Start

All commands below assume:

- you are in the repository root
- your virtual environment is active
- your dataset lives at `data2017/`

### 1. Run the baseline Phase 1 training script

This is the fastest sanity check. It uses a fixed-length 1D CNN and defaults to one epoch.

```bash
python src/train_phase1.py \
  --data-dir data2017 \
  --epochs 1 \
  --batch-size 32 \
  --target-len 3000 \
  --balance weighted_loss
```

Useful options:

- `--subset N` limits training to the first `N` records for a smoke test.
- `--cpu` forces CPU even if MPS or CUDA is available.
- `--num-workers 0` is the default and is the safest choice on macOS.

### 2. Train the main variable-length FFT model

This is the more complete training path.

```bash
python src/train_fft_gp.py \
  --data-dir data2017 \
  --epochs 10 \
  --batch-size 32 \
  --fft-bins 256 \
  --balance weighted_loss \
  --checkpoint-path checkpoints/train_fft_gp_best.pt
```

Notes:

- `--target-len 0` is the default and keeps original record lengths.
- validation is stratified per class
- best checkpoint is selected by validation macro-F1
- early stopping is controlled by `--patience` and defaults to `7`

### 3. Evaluate a saved checkpoint on the validation split

```bash
python src/eval_fft_gp.py \
  --data-dir data2017 \
  --checkpoint-path checkpoints/train_fft_gp_best.pt \
  --batch-size 1
```

This prints:

- checkpoint path
- validation accuracy
- validation macro-F1
- per-class recall
- confusion matrix
- average latency
- throughput
- model size on disk

### 4. Run the synthetic inference benchmark

This benchmark does not use the dataset. It creates random variable-length batches to measure forward-pass performance.

```bash
python src/benchmark_fft_gp.py \
  --checkpoint-path checkpoints/train_fft_gp_best.pt \
  --batch-size 1 \
  --min-len 2500 \
  --max-len 9000 \
  --timed-batches 100
```

This is useful for comparing CPU versus GPU, or for testing a machine before you deploy to target hardware.

## Example Commands For Smoke Tests

If you just want to verify that the code runs end to end, these are small CPU-only examples:

```bash
python src/train_phase1.py --data-dir data2017 --subset 32 --epochs 1 --batch-size 8 --cpu
python src/train_fft_gp.py --data-dir data2017 --subset 64 --epochs 1 --batch-size 8 --cpu --checkpoint-path /tmp/readme_fft_gp.pt
python src/eval_fft_gp.py --data-dir data2017 --checkpoint-path /tmp/readme_fft_gp.pt --subset 64 --batch-size 1 --cpu --warmup-batches 2
python src/benchmark_fft_gp.py --checkpoint-path /tmp/readme_fft_gp.pt --batch-size 1 --min-len 2500 --max-len 4000 --warmup-batches 2 --timed-batches 5 --cpu
```

## What It Looks Like When It Runs

These are real outputs from small smoke runs in this workspace. The metrics are not meaningful because the runs used tiny subsets and only one epoch; the point is to show expected console structure.

### Baseline training output

```text
[device] cpu
[train] class_0=16 class_1=2 class_2=7 class_3=1
[val] class_0=2 class_1=3 class_2=1 class_3=0
[balance] weighted_loss class_weights=[0.14659684896469116, 1.1727747917175293, 0.33507850766181946, 2.3455495834350586]
[epoch 1/1] train_loss=1.3917 train_acc=0.0938 val_loss=1.4555 val_acc=0.1667
```

### Main training output

```text
[device] cpu
[train] class_0=32 class_1=5 class_2=12 class_3=2
[val] class_0=8 class_1=1 class_2=3 class_3=1
[balance] weighted_loss class_weights=[0.15345267951488495, 0.9820971488952637, 0.40920716524124146, 2.455242872238159]
[epoch 1/1] train_loss=1.4154 train_acc=0.0536 val_loss=1.3934 val_acc=0.1250 val_macro_f1=0.0667 lr=0.001000 train_time_s=1.23 val_time_s=0.03 train_samples_per_s=41.39 val_samples_per_s=400.13
[val_metrics] class_0_recall=0.0000 class_1_recall=0.0000 class_2_recall=0.6667 class_3_recall=0.0000
[confusion_matrix] rows=true cols=pred
   0    0    8    0
   0    0    1    0
   0    1    2    0
   0    0    1    0
[checkpoint] saved_best path=/tmp/readme_fft_gp.pt epoch=1 macro_f1=0.0667
[best] epoch=1 macro_f1=0.0667 checkpoint=/tmp/readme_fft_gp.pt
```

### Evaluation output

```text
[device] cpu
[checkpoint] path=/tmp/readme_fft_gp.pt
[quality] accuracy=0.1538 macro_f1=0.0667 class_0_recall=0.0000 class_1_recall=0.0000 class_2_recall=0.6667 class_3_recall=0.0000
[confusion_matrix] rows=true cols=pred
   0    0    8    0
   0    0    1    0
   0    1    2    0
   0    0    1    0
[performance] batch_size=1 avg_latency_ms=1.837 throughput_samples_per_s=544.323
[footprint] model_size_mb=0.672
```

### Synthetic benchmark output

```text
[device] cpu
[checkpoint] path=/tmp/readme_fft_gp.pt
[model] total_params=57316 trainable_params=57316 checkpoint_size_mb=0.672 fft_bins=256
[benchmark] batch_size=1 min_len=2500 max_len=4000 warmup_batches=2 timed_batches=5 amp=False
[latency] mean_ms=5.007 p50_ms=2.492 p95_ms=12.024 throughput_samples_per_s=199.735
```

## How The Code Is Organized

### Top-level folders and files

| Path | What it does |
| --- | --- |
| `.gitignore` | Ignores Python cache files, virtual environments, model artifacts, logs, runs, and `data2017/`. |
| `README.md` | Project documentation and run guide. |
| `data2017/` | Expected dataset root containing `REFERENCE.csv` and `training/*.mat`. |
| `checkpoints/` | Default place to save trained `.pt` model checkpoints. |
| `src/` | All loader, model, training, evaluation, and benchmarking scripts. |

### Source files

| File | What it does |
| --- | --- |
| `src/dataset_physionet2017.py` | Loads `.mat` ECG records, maps labels to class IDs, cleans NaNs/Infs, optionally center-crops or center-pads to a target length, z-score normalizes each record, and returns tensors shaped for `Conv1d`. |
| `src/models_1dcnn.py` | Defines `SimpleECG1DCNN`, the small fixed-length baseline model used in `train_phase1.py`. |
| `src/train_phase1.py` | Phase 1 smoke-test training script. Uses the baseline CNN, random train/validation split, optional class balancing, and a standard epoch loop with loss and accuracy reporting. |
| `src/models_fft_gp.py` | Defines `ECGFFTGlobalPoolNet`, the variable-length model with a time branch, an FFT branch, masked global average pooling, and a fused classification head. |
| `src/train_fft_gp.py` | Main training script for the FFT/global-pooling model. Adds padded variable-length batching, stratified train/validation split, macro-F1 tracking, confusion matrix printing, learning-rate scheduling, early stopping, and checkpoint saving. |
| `src/eval_fft_gp.py` | Loads a saved FFT/global-pooling checkpoint, rebuilds the validation split, computes accuracy/macro-F1/per-class recall/confusion matrix, and reports real-data latency and throughput. |
| `src/benchmark_fft_gp.py` | Loads a saved checkpoint and measures synthetic forward-pass latency, percentiles, throughput, parameter count, and checkpoint size using random variable-length inputs. |

## How Each Script Behaves

### `src/train_phase1.py`

Use this when you want the simplest possible training path.

- fixed-length input via `--target-len` with default `3000`
- model: `SimpleECG1DCNN`
- random train/validation split
- metrics: loss and accuracy
- default epochs: `1`

This is mostly for "does the data loader work and can I train without crashing?"

### `src/train_fft_gp.py`

Use this for the main model.

- keeps original lengths by default
- pads within each batch using `pad_collate`
- passes true lengths into the model
- applies masked pooling so padded timesteps do not affect time-branch averaging
- computes macro-F1 and per-class recall
- saves the best checkpoint by validation macro-F1

### `src/eval_fft_gp.py`

Use this after training.

- loads a checkpoint
- recreates the same validation split logic
- reports both quality and inference speed

### `src/benchmark_fft_gp.py`

Use this when you care about deployment-ish numbers.

- no dataset required
- generates synthetic variable-length inputs
- reports mean, p50, and p95 latency
- reports throughput and model footprint

## Default Files Written By The Code

The main training script writes a checkpoint dictionary to the path you pass with `--checkpoint-path`.

By default:

```text
checkpoints/train_fft_gp_best.pt
```

The checkpoint contains:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `best_macro_f1`
- `args`

## Common Command Patterns

### Train on CPU

```bash
python src/train_fft_gp.py --data-dir data2017 --cpu
```

### Train on a small subset first

```bash
python src/train_fft_gp.py --data-dir data2017 --subset 256 --epochs 2 --batch-size 8 --cpu
```

### Use fixed-length preprocessing even with the FFT model

```bash
python src/train_fft_gp.py --data-dir data2017 --target-len 3000
```

### Evaluate a specific checkpoint

```bash
python src/eval_fft_gp.py --data-dir data2017 --checkpoint-path checkpoints/train_fft_gp_best.pt --cpu
```

### Benchmark with larger synthetic batches

```bash
python src/benchmark_fft_gp.py --checkpoint-path checkpoints/train_fft_gp_best.pt --batch-size 8 --min-len 2500 --max-len 9000
```

## Troubleshooting

### `python: command not found`

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

Or call the interpreter directly:

```bash
./.venv/bin/python src/train_fft_gp.py --data-dir data2017 --cpu
```

### `Missing folder: .../training` or `Missing file: .../REFERENCE.csv`

Your dataset path is wrong or the dataset is not extracted in the expected layout. Recheck the `data2017/` tree shown above.

### `Missing checkpoint: ...`

You need to train first or point evaluation/benchmark scripts at an existing `.pt` file.

### You see a CUDA warning on a CPU-only machine

If your PyTorch build includes CUDA support, PyTorch can emit CUDA initialization warnings even when you run on CPU. Passing `--cpu` is still the right choice on machines without a usable GPU. If you want to avoid those warnings entirely, install a CPU-only PyTorch build.

### Training is slow or runs out of memory

Try one or more of:

- `--cpu` if the accelerator backend is unstable
- smaller `--batch-size`
- `--subset` for smoke tests
- `--target-len 3000` or smaller
- `--num-workers 0`

## Practical Recommendation

If you are new to this repo, use this order:

1. `train_phase1.py` on a small subset to verify the environment.
2. `train_fft_gp.py` for actual model training.
3. `eval_fft_gp.py` to inspect validation metrics and real-data latency.
4. `benchmark_fft_gp.py` to compare inference speed across machines or settings.
