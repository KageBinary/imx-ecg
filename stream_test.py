"""
Streaming test harness for the ECG dashboard.

Loads PhysioNet 2017 .mat records and feeds them into the dashboard at the
correct 300 Hz sample rate (or faster, for development).

Usage examples
--------------
# Stream all records (AF class only) at real time:
    python stream_test.py --label A

# Stream a specific record at 5× speed:
    python stream_test.py --record A00004 --speed 5

# Stream random records, re-classify every 0.5 s:
    python stream_test.py --shuffle --classify-every 150

# Use a different checkpoint:
    python stream_test.py --checkpoint checkpoints/deploy_v2.pt --speed 3

Arguments
---------
--checkpoint    Path to .pt checkpoint  (default: checkpoints/deploy_v2.pt)
--data-dir      PhysioNet 2017 root     (default: data2017)
--record        Single record ID        (e.g. A00004; overrides --label / --shuffle)
--label         Class filter            N | A | O | ~
--shuffle       Randomise record order  (default: alphabetical)
--speed         Playback multiplier     (default: 3.0; use 1.0 for real time)
--classify-every  Re-classify every N samples (default: 150 = 0.5 s)
--max-records   Stop after this many records (0 = unlimited)
"""
from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io

from dashboard import ECGDashboard
from ecg_inference import ECGInference, SAMPLE_RATE

_LABEL_MAP = {'N': 'Normal', 'A': 'AF', 'O': 'Other', '~': 'Noisy'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_reference(data_dir: Path) -> dict[str, str]:
    ref: dict[str, str] = {}
    with open(data_dir / "REFERENCE.csv") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                ref[parts[0]] = parts[1]
    return ref


def load_mat_signal(mat_path: Path) -> np.ndarray:
    mat = scipy.io.loadmat(str(mat_path))
    return mat['val'].squeeze().astype(np.float32)


def stream_record(
    dashboard: ECGDashboard,
    signal: np.ndarray,
    true_label: str,
    speed: float,
) -> None:
    """
    Push samples from *signal* into the dashboard queue at the correct rate.
    Sends in small batches to keep the queue responsive without busy-spinning.
    """
    # How many real seconds per sample, scaled by speed
    seconds_per_sample = 1.0 / (SAMPLE_RATE * speed)
    # Aim for ~20 ms chunks so the OS sleep is manageable
    chunk_size = max(1, int(SAMPLE_RATE * speed * 0.020))
    chunk_sleep = seconds_per_sample * chunk_size

    for i in range(0, len(signal), chunk_size):
        chunk = signal[i: i + chunk_size]
        dashboard.push_samples(chunk, true_label)
        time.sleep(chunk_sleep)


# ---------------------------------------------------------------------------
# Streaming thread
# ---------------------------------------------------------------------------

def _stream_thread(
    dashboard: ECGDashboard,
    records: list[tuple[str, str]],  # [(record_id, label), ...]
    data_dir: Path,
    speed: float,
    max_records: int,
) -> None:
    done = 0
    for rec_id, label in records:
        if max_records and done >= max_records:
            break
        mat_path = data_dir / "training" / f"{rec_id}.mat"
        if not mat_path.exists():
            continue

        signal = load_mat_signal(mat_path)
        duration = len(signal) / SAMPLE_RATE
        true_name = _LABEL_MAP.get(label, label)
        title = (
            f"Record: {rec_id}  |  True label: {true_name}  "
            f"|  Duration: {duration:.1f} s  |  Speed: {speed}×"
        )
        print(f"  → {title}")
        dashboard.set_title(title)

        stream_record(dashboard, signal, label, speed)
        done += 1

        # Pause between records so the last window is visible
        time.sleep(max(0.5, 2.0 / speed))

    print("Stream finished.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feed PhysioNet 2017 ECG records into the real-time dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",      default="checkpoints/deploy_v2.pt")
    p.add_argument("--data-dir",        default="data2017")
    p.add_argument("--record",          default=None,
                   help="Single record ID (e.g. A00004)")
    p.add_argument("--label",           default=None, choices=["N", "A", "O", "~"],
                   help="Stream only records of this class")
    p.add_argument("--shuffle",         action="store_true",
                   help="Randomise record order")
    p.add_argument("--speed",           type=float, default=3.0,
                   help="Playback speed (1.0 = real time, 3.0 = 3× faster)")
    p.add_argument("--classify-every",  type=int, default=150,
                   help="Re-classify every N new samples (150 = 0.5 s)")
    p.add_argument("--max-records",     type=int, default=0,
                   help="Stop after this many records (0 = unlimited)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        sys.exit(f"Data directory not found: {data_dir}")
    if not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading model from {args.checkpoint} …")
    inference = ECGInference(args.checkpoint)
    dashboard = ECGDashboard(
        inference_fn=inference.classify,
        classify_every_n=args.classify_every,
    )

    reference = load_reference(data_dir)

    if args.record:
        records = [(args.record, reference.get(args.record, '?'))]
    else:
        records = sorted(reference.items())
        if args.label:
            records = [(r, l) for r, l in records if l == args.label]
        if args.shuffle:
            random.shuffle(records)

    if not records:
        sys.exit("No records matched the given filters.")

    n_show = min(5, len(records))
    print(f"Streaming {len(records)} record(s) at {args.speed}× speed "
          f"(first {n_show}: {[r for r, _ in records[:n_show]]})")
    print(f"Re-classifying every {args.classify_every} samples "
          f"({args.classify_every / SAMPLE_RATE:.2g} s)")

    t = threading.Thread(
        target=_stream_thread,
        args=(dashboard, records, data_dir, args.speed, args.max_records),
        daemon=True,
    )
    t.start()

    dashboard.run()


if __name__ == "__main__":
    main()
