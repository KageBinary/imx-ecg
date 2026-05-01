"""
terminal_dashboard.py — Text-mode ECG monitor, no GUI required.
Renders a Unicode block-character waveform + classification to the terminal.
Works over plain SSH with no X11 or browser needed.
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
from typing import Callable

import numpy as np

_CLASS_NAMES  = ['Normal', 'AF', 'Other', 'Noisy']
_CLASS_COLORS = ['\033[92m', '\033[91m', '\033[93m', '\033[95m']  # green red yellow magenta
_RESET = '\033[0m'
_BOLD  = '\033[1m'
_DIM   = '\033[2m'
_BLOCKS = ' ▁▂▃▄▅▆▇█'


class TerminalDashboard:
    """Drop-in replacement for ECGDashboard / ECGDashboardLite over plain SSH."""

    def __init__(
        self,
        inference_fn: Callable,
        classify_every_n: int = 125,
        title: str = 'ECG Monitor',
        sample_rate: int = 300,
    ):
        self._inference_fn    = inference_fn
        self.classify_every_n = classify_every_n
        self._title           = title

        self._queue     = queue.Queue()
        self._buf       = np.zeros(3000, dtype=np.float32)
        self._since_cls = 0
        self._total_rx  = 0

        self._name  = '—'
        self._conf  = 0.0
        self._pred  = -1
        self._probs = np.zeros(4, dtype=np.float32)
        self._lock  = threading.Lock()

    # ── public interface (matches ECGDashboard / ECGDashboardLite) ──────────────

    def push_sample(self, sample: float, label=None) -> None:
        self._queue.put(float(sample))

    def push_samples(self, samples, label=None) -> None:
        for s in samples:
            self._queue.put(float(s))

    def run(self) -> None:
        print('\033[?25l\033[2J', end='', flush=True)  # hide cursor, clear screen
        try:
            while True:
                self._drain()
                self._render()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            print(f'\033[?25h{_RESET}\033[2J\033[H', end='', flush=True)
            print('Stopped.')

    # ── internal ────────────────────────────────────────────────────────────────

    def _drain(self) -> None:
        new: list[float] = []
        try:
            while True:
                new.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        if not new:
            return
        n = len(new)
        self._total_rx += n
        self._buf = np.roll(self._buf, -n)
        self._buf[-n:] = np.array(new, dtype=np.float32)
        self._since_cls += n
        if self._since_cls >= self.classify_every_n:
            self._since_cls = 0
            self._classify()

    def _classify(self) -> None:
        pred, name, probs = self._inference_fn(self._buf.copy())
        with self._lock:
            self._pred  = pred
            self._name  = name
            self._conf  = float(probs[pred])
            self._probs = probs.copy()

    def _render(self) -> None:
        cols, _ = shutil.get_terminal_size(fallback=(80, 24))
        w = max(cols - 4, 10)
        sep = '━' * cols

        # ── waveform ──────────────────────────────────────────────────────────
        sig = self._buf[-w:].copy()
        std = float(sig.std())
        if std > 1e-6:
            sig = np.clip((sig - sig.mean()) / std, -3.0, 3.0)
            indices = np.clip(((sig + 3.0) / 6.0 * 8.0).astype(int), 0, 8)
        else:
            indices = np.zeros(len(sig), dtype=int)
        wave = ''.join(_BLOCKS[i] for i in indices)

        # ── classification ────────────────────────────────────────────────────
        with self._lock:
            name  = self._name
            conf  = self._conf
            pred  = self._pred
            probs = self._probs.copy()

        color = _CLASS_COLORS[pred] if pred >= 0 else ''
        filled = int(conf * 20)
        bar = f'{"█" * filled}{_DIM}{"░" * (20 - filled)}{_RESET}'

        # ── per-class probability rows ────────────────────────────────────────
        prob_lines = []
        for i, (cname, ccolor) in enumerate(zip(_CLASS_NAMES, _CLASS_COLORS)):
            p = float(probs[i])
            b = f'{"█" * int(p * 10)}{"░" * (10 - int(p * 10))}'
            marker = '▶ ' if i == pred else '  '
            prob_lines.append(f'  {marker}{ccolor}{cname:<6}{_RESET} {ccolor}{b}{_RESET}  {p:.0%}')

        lines = [
            '\033[H',
            f'  {_BOLD}ECG Monitor{_RESET}  {_DIM}{self._title}{_RESET}',
            f'  {sep}',
            '',
            f'  {color}{wave}{_RESET}',
            '',
            f'  {sep}',
            f'  {_BOLD}{color}{name:<8}{_RESET}  {color}{bar}  {conf:.0%}{_RESET}',
            f'  {sep}',
            '',
            *prob_lines,
            '',
            f'  {_DIM}rx: {self._total_rx}   buf: {self._since_cls}/{self.classify_every_n}   signal std: {std:.2f}{_RESET}',
        ]
        print('\n'.join(lines), end='', flush=True)
