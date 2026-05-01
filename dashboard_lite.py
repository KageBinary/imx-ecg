"""
Lightweight ECG dashboard for resource-constrained boards.

Two internal buffers:
  _disp_buf  — 750 samples (2.5 s display window, keeps the plot fast)
  _cls_buf   — 3000 samples (full model window, correct inference)

Compared to dashboard.py:
  - No audio
  - No glow stack (single line)
  - No history log, no probability bars
  - 10 fps default (vs 30 fps)
  - ~10× less matplotlib work per frame

Public API (same as ECGDashboard):
    from dashboard_lite import ECGDashboardLite
    dash = ECGDashboardLite(inference_fn)
    dash.push_sample(float_value)
    dash.push_samples(np.ndarray, true_label='N')
    dash.run()
"""
from __future__ import annotations

import queue
import time
from typing import Callable, Optional

import matplotlib
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

DISPLAY_SAMPLES  = 750    # 2.5 s shown on screen
CLASSIFY_SAMPLES = 3000   # full model window kept internally

CLASS_NAMES  = ['Normal', 'AF', 'Other', 'Noisy']
CLASS_COLORS = ['#00e676', '#ff4569', '#ffab40', '#ce93d8']

_BG    = '#0a0f1a'
_PANEL = '#0f1825'
_LINE  = '#00e5cc'
_DIM   = '#3a5068'
_MID   = '#6a8aaa'
_HI    = '#c0d8f0'


def _estimate_bpm(z: np.ndarray, sr: int = 300) -> int:
    peaks, _ = find_peaks(z, height=1.2, distance=int(sr * 0.35))
    if len(peaks) < 2:
        return 0
    return int(np.clip(60.0 / np.median(np.diff(peaks) / sr), 25, 240))


class ECGDashboardLite:
    def __init__(
        self,
        inference_fn: Callable,
        classify_every_n: int = 150,
        title: str = "",
        sample_rate: int = 300,
    ):
        self.inference_fn     = inference_fn
        self.classify_every_n = classify_every_n
        self.title            = title
        self._sr              = sample_rate

        self._queue: queue.Queue = queue.Queue(maxsize=30_000)
        self._disp_buf  = np.zeros(DISPLAY_SAMPLES,  dtype=np.float32)
        self._cls_buf   = np.zeros(CLASSIFY_SAMPLES, dtype=np.float32)
        self._since_cls  = 0
        self._frame      = 0
        self._last_frame = time.monotonic()
        self._fps        = 0.0

        self._pred    = 0
        self._name    = CLASS_NAMES[0]
        self._conf    = 0.25
        self._bpm     = 0
        self._bpm_ema = 0.0

        self._build()

    # ── Public API ────────────────────────────────────────────────────────────

    def push_sample(self, sample: float, true_label: Optional[str] = None) -> None:
        self._queue.put(float(sample))

    def push_samples(self, samples: np.ndarray, true_label: Optional[str] = None) -> None:
        for s in samples:
            self._queue.put(float(s))

    def set_title(self, title: str) -> None:
        self.title = title

    def run(self, interval_ms: int = 100) -> None:
        self._ani = animation.FuncAnimation(
            self.fig, self._update,
            interval=interval_ms, blit=True, cache_frame_data=False,
        )
        plt.show()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        plt.rcParams.update({
            'figure.facecolor':  _BG,
            'axes.facecolor':    _BG,
            'axes.edgecolor':    _DIM,
            'text.color':        _MID,
            'font.family':       'monospace',
            'font.size':         9,
        })

        self.fig = plt.figure(figsize=(10, 5))
        self.fig.patch.set_facecolor(_BG)

        gs = gridspec.GridSpec(
            2, 1,
            figure=self.fig,
            height_ratios=[5, 1],
            left=0.05, right=0.97,
            top=0.93, bottom=0.08,
            hspace=0.18,
        )

        # ── ECG axes ─────────────────────────────────────────────────────────
        self._ax_ecg = self.fig.add_subplot(gs[0])
        self._ax_ecg.set_facecolor(_BG)
        for sp in self._ax_ecg.spines.values():
            sp.set_visible(False)
        self._ax_ecg.spines['bottom'].set_visible(True)
        self._ax_ecg.spines['bottom'].set_color(_DIM)
        self._ax_ecg.set_xlim(0, DISPLAY_SAMPLES)
        self._ax_ecg.set_ylim(-4.5, 4.5)
        self._ax_ecg.set_yticks([])
        self._ax_ecg.set_xticks([])

        t = np.arange(DISPLAY_SAMPLES)
        self._ecg_line, = self._ax_ecg.plot(
            t, np.zeros(DISPLAY_SAMPLES),
            color=_LINE, lw=1.0, alpha=0.9,
        )

        # ── Status bar ───────────────────────────────────────────────────────
        self._ax_st = self.fig.add_subplot(gs[1])
        self._ax_st.set_facecolor(_PANEL)
        for sp in self._ax_st.spines.values():
            sp.set_color(_DIM)
        self._ax_st.set_xlim(0, 1)
        self._ax_st.set_ylim(0, 1)
        self._ax_st.set_xticks([])
        self._ax_st.set_yticks([])

        self._txt_cls = self._ax_st.text(
            0.02, 0.5, 'Normal',
            transform=self._ax_st.transAxes,
            fontsize=14, fontweight='bold',
            color=CLASS_COLORS[0], va='center',
        )
        self._txt_conf = self._ax_st.text(
            0.18, 0.5, '25%',
            transform=self._ax_st.transAxes,
            fontsize=11, color=_MID, va='center',
        )
        self._txt_bpm = self._ax_st.text(
            0.50, 0.5, 'BPM: —',
            transform=self._ax_st.transAxes,
            fontsize=12, color=_HI, va='center',
            fontfamily='monospace',
        )
        self._ax_st.text(
            0.80, 0.5, '2.5 s window',
            transform=self._ax_st.transAxes,
            fontsize=7, color=_DIM, va='center',
        )

        # Title
        self.fig.text(
            0.05, 0.97, self.title or 'ECG LIVE',
            fontsize=9, color=_DIM, va='top',
        )

    # ── Animation ─────────────────────────────────────────────────────────────

    def _update(self, _f: int):
        now = time.monotonic()
        elapsed = now - self._last_frame
        self._last_frame = now
        if elapsed > 0:
            self._fps = 0.1 * (1.0 / elapsed) + 0.9 * self._fps

        self._frame += 1

        # Drain queue
        new: list[float] = []
        try:
            while True:
                new.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        if not new:
            return [self._ecg_line, self._txt_cls, self._txt_conf, self._txt_bpm]

        new = np.asarray(new, dtype=np.float32)
        if len(new) >= len(self._disp_buf):
            self._disp_buf[:] = new[-len(self._disp_buf):]
        else:
            n = len(new)
            self._disp_buf = np.roll(self._disp_buf, -n)
            self._disp_buf[-n:] = new
        if len(new) >= len(self._cls_buf):
            self._cls_buf[:] = new[-len(self._cls_buf):]
        else:
            n = len(new)
            self._cls_buf = np.roll(self._cls_buf, -n)
            self._cls_buf[-n:] = new
        self._since_cls += len(new)
        if self._since_cls >= self.classify_every_n:
            self._since_cls = 0
            self._classify()

        # Z-score display buffer
        std = self._disp_buf.std()
        if std > 1e-6:
            disp = (self._disp_buf - self._disp_buf.mean()) / std
        else:
            disp = np.zeros(DISPLAY_SAMPLES)

        self._ecg_line.set_ydata(disp)

        # BPM every ~1 s (10 frames at 100ms)
        if self._frame % 10 == 0 and std > 1e-6:
            raw = _estimate_bpm(disp, sr=self._sr)
            if raw:
                alpha = 0.3 if self._bpm_ema == 0.0 else 0.15
                self._bpm_ema = alpha * raw + (1 - alpha) * self._bpm_ema
                self._bpm = int(round(self._bpm_ema))
            self._txt_bpm.set_text(
                f'BPM: {self._bpm}' if self._bpm else 'BPM: —'
            )

        if self._frame % 10 == 0:
            print(f"\rFPS: {self._fps:.1f}  queue: {self._queue.qsize()}", end="", flush=True)

        return [self._ecg_line, self._txt_cls, self._txt_conf, self._txt_bpm]

    def _classify(self) -> None:
        pred, name, probs = self.inference_fn(self._cls_buf.copy())
        self._pred = pred
        self._name = name
        self._conf = float(probs[pred])
        c = CLASS_COLORS[pred]
        self._txt_cls.set_text(name)
        self._txt_cls.set_color(c)
        self._txt_conf.set_text(f'{self._conf:.0%}')
        self._txt_conf.set_color(c)
