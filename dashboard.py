"""
Real-time ECG classification dashboard.

Public API:
    from ecg_inference import ECGInference
    from dashboard import ECGDashboard

    inf  = ECGInference("checkpoints/deploy_v2.pt")
    dash = ECGDashboard(inf.classify)
    dash.push_samples(np.array([...]), true_label='N')
    dash.run()
"""
from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Callable, Optional

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.signal import find_peaks

# ── Audio engine ─────────────────────────────────────────────────────────────
# Priority order:
#   1. sounddevice  — native Linux / macOS (needs: pip install sounddevice + libportaudio2)
#   2. PowerShell SoundPlayer — WSL2 (zero extra setup, uses Windows audio stack)
#   3. Silent fallback

import os
import subprocess
import tempfile
import wave

_AUDIO_SR = 44_100


def _sine(freq: float, dur: float, amp: float = 0.30,
          attack: float = 0.004, release: float = 0.020) -> np.ndarray:
    n   = int(_AUDIO_SR * dur)
    t   = np.linspace(0, dur, n, False)
    wav = np.sin(2 * np.pi * freq * t) * amp
    a   = min(int(_AUDIO_SR * attack),  n // 4)
    r   = min(int(_AUDIO_SR * release), n // 4)
    wav[:a]  *= np.linspace(0, 1, a)
    wav[-r:] *= np.linspace(1, 0, r)
    return wav.astype(np.float32)

def _silence(dur: float) -> np.ndarray:
    return np.zeros(int(_AUDIO_SR * dur), dtype=np.float32)

def _make_sounds() -> dict[str, np.ndarray]:
    """All sounds as float32 PCM arrays at _AUDIO_SR Hz."""
    return {
        # Short pip on every detected R-peak — like a real monitor
        'beep': _sine(880, 0.055, amp=0.28, attack=0.003, release=0.020),

        # Classification chimes — each class has a distinct character
        'Normal': np.concatenate([                      # rising two-note: reassuring
            _sine(660, 0.065, amp=0.20), _silence(0.018), _sine(880, 0.085, amp=0.20),
        ]),
        'AF': np.concatenate([                          # descending double: urgent
            _sine(700, 0.075, amp=0.28), _silence(0.018), _sine(550, 0.110, amp=0.28),
        ]),
        'Other': np.concatenate([                       # neutral single mid tone
            _sine(600, 0.110, amp=0.20),
        ]),
        'Noisy': np.concatenate([                       # low double blip: warning
            _sine(440, 0.065, amp=0.22), _silence(0.030), _sine(440, 0.065, amp=0.22),
        ]),
    }

def _to_pcm16(data: np.ndarray) -> bytes:
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

def _write_wav(path: str, data: np.ndarray) -> None:
    with wave.open(path, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(_AUDIO_SR)
        f.writeframes(_to_pcm16(data))


class _AudioPlayer:
    """
    Non-blocking audio. Sounds are submitted to an internal queue and played
    serially in a dedicated daemon thread so the animation loop is never stalled.

    Backends (tried in order):
      • sounddevice  (native Linux / macOS)
      • PowerShell SoundPlayer  (WSL2 — writes WAVs to /tmp, plays via Windows)
      • silent fallback
    """

    # Max queued items — older ones are dropped to keep latency low
    _BEEP_MAXQ  = 1   # heartbeat: only ever want the latest
    _CHIME_MAXQ = 2   # classification chimes

    def __init__(self) -> None:
        self._ok      = False
        self._backend = 'none'
        self._q: queue.Queue = queue.Queue()
        self._sounds: dict[str, np.ndarray] = {}
        self._win_paths: dict[str, str]     = {}

        if self._try_sounddevice():
            return
        if self._try_powershell():
            return
        print("Audio: no backend available — running silent")

    # ── Backend probes ────────────────────────────────────────────────────────

    def _try_sounddevice(self) -> bool:
        try:
            import sounddevice as sd
            sd.query_devices(kind='output')   # raises if no devices
            self._sd      = sd
            self._sounds  = _make_sounds()
            self._backend = 'sounddevice'
            self._ok      = True
            threading.Thread(target=self._worker_sd, daemon=True,
                             name='audio-sd').start()
            print("Audio: sounddevice ready")
            return True
        except Exception:
            return False

    def _try_powershell(self) -> bool:
        try:
            # Check wslpath and powershell.exe are both reachable
            subprocess.run(['wslpath', '--version'], capture_output=True, timeout=3)
            subprocess.run(
                ['powershell.exe', '-NoProfile', '-Command', 'echo ok'],
                capture_output=True, timeout=5,
            )
            # Write WAV files and resolve Windows UNC paths
            self._tmpdir = tempfile.mkdtemp(prefix='ecg_audio_')
            for name, data in _make_sounds().items():
                wsl_path = os.path.join(self._tmpdir, f'{name}.wav')
                _write_wav(wsl_path, data)
                win = subprocess.check_output(
                    ['wslpath', '-w', wsl_path], text=True, timeout=3,
                ).strip()
                self._win_paths[name] = win

            # Smoke test — play beep once at startup
            self._ps_play(self._win_paths['beep'])

            self._backend = 'powershell'
            self._ok      = True
            threading.Thread(target=self._worker_ps, daemon=True,
                             name='audio-ps').start()
            print("Audio: Windows SoundPlayer ready (WSL2)")
            return True
        except Exception as e:
            print(f"Audio: PowerShell backend failed ({e})")
            return False

    # ── Workers ───────────────────────────────────────────────────────────────

    def _worker_sd(self) -> None:
        while True:
            key = self._q.get()
            wav = self._sounds.get(key)
            if wav is not None:
                try:
                    self._sd.play(wav, _AUDIO_SR, blocking=True)
                except Exception:
                    pass

    def _worker_ps(self) -> None:
        while True:
            key = self._q.get()
            win = self._win_paths.get(key)
            if win:
                self._ps_play(win)

    @staticmethod
    def _ps_play(win_path: str) -> None:
        subprocess.run(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
             f"[System.Media.SoundPlayer]::new('{win_path}').PlaySync()"],
            capture_output=True, timeout=15,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def play(self, key: str) -> None:
        """Queue a sound. Non-blocking; drops if the queue is full."""
        if not self._ok:
            return
        maxq = self._BEEP_MAXQ if key == 'beep' else self._CHIME_MAXQ
        if self._q.qsize() < maxq:
            try:
                self._q.put_nowait(key)
            except queue.Full:
                pass

SAMPLE_RATE    = 300
WINDOW_SAMPLES = 3000

CLASS_NAMES  = ['Normal', 'AF', 'Other', 'Noisy']
CLASS_COLORS = ['#00e676', '#ff4569', '#ffab40', '#ce93d8']

# ── Palette ──────────────────────────────────────────────────────────────────
_BG       = '#050810'
_PANEL    = '#090f1c'
_PANEL2   = '#0c1525'
_BORDER   = '#13263f'
_ACCENT   = '#38bdf8'
_ECG      = '#00ffcc'
_ECG_CORE = '#d0fff5'
_TEXT_DIM = '#4a6380'
_TEXT_MID = '#7a9ab8'
_TEXT_HI  = '#c8dff0'

_HISTORY_LEN = 6
_LABEL_MAP   = {'N': 'Normal', 'A': 'AF', 'O': 'Other', '~': 'Noisy'}


def _estimate_bpm(z: np.ndarray, sr: int = SAMPLE_RATE) -> int:
    peaks, _ = find_peaks(z, height=1.2, distance=int(sr * 0.35))
    if len(peaks) < 2:
        return 0
    return int(np.clip(60.0 / np.median(np.diff(peaks) / sr), 25, 240))


class ECGDashboard:
    def __init__(
        self,
        inference_fn: Callable,
        classify_every_n: int = 150,
        title: str = "",
    ):
        self.inference_fn     = inference_fn
        self.classify_every_n = classify_every_n
        self.title            = title

        self._queue: queue.Queue = queue.Queue(maxsize=30_000)
        self._buffer    = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._since_cls = 0
        self._true_lbl: Optional[str] = None
        self._frame     = 0

        self._pred  = 0
        self._name  = CLASS_NAMES[0]
        self._probs = np.full(4, 0.25, dtype=np.float32)
        self._bpm   = 0
        self._history: deque = deque(maxlen=_HISTORY_LEN)
        self._total   = 0
        self._correct = 0
        self._flash   = 0

        self._audio        = _AudioPlayer()
        self._last_peaks   = np.array([], dtype=int)  # R-peak sample indices from last frame

        # suppress unused-variable linter noise — these attrs are set in build
        self._hdr_sub = self._bpm_val = self._live_dot = None
        self._eg3 = self._eg2 = self._eg1 = self._eln = self._peak_dots = None
        self._cls_tint = self._cls_stripe = None
        self._cls_lbl = self._cls_conf = self._cls_true = None
        self._pb_bars = self._pb_pct = None
        self._log_squares = self._log_names = self._log_pcts = None
        self._log_sep = self._log_acc = None

        self._build()

    # ── Public API ────────────────────────────────────────────────────────────

    def push_sample(self, sample: float, true_label: Optional[str] = None) -> None:
        self._queue.put((float(sample), true_label))

    def push_samples(self, samples: np.ndarray, true_label: Optional[str] = None) -> None:
        for s in samples:
            self._queue.put((float(s), true_label))

    def set_title(self, title: str) -> None:
        self.title = title
        if self._hdr_sub is not None:
            self._hdr_sub.set_text(title)

    def run(self, interval_ms: int = 33) -> None:
        self._ani = animation.FuncAnimation(
            self.fig, self._update,
            interval=interval_ms, blit=False, cache_frame_data=False,
        )
        plt.show()

    # ── Global style ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        # Override just the parts we care about instead of a full style preset
        plt.rcParams.update({
            'axes.facecolor':      _PANEL,
            'axes.edgecolor':      _BORDER,
            'axes.labelcolor':     _TEXT_MID,
            'axes.spines.top':     False,
            'axes.spines.right':   False,
            'xtick.color':         _TEXT_DIM,
            'ytick.color':         _TEXT_DIM,
            'figure.facecolor':    _BG,
            'text.color':          _TEXT_MID,
            'font.family':         'sans-serif',
            'font.size':           9,
        })

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor(_BG)

        gs = gridspec.GridSpec(
            3, 4,
            figure=self.fig,
            height_ratios=[0.09, 0.59, 0.32],
            width_ratios=[1.3, 1.1, 1.05, 0.85],
            left=0.04, right=0.975,
            top=0.975, bottom=0.055,
            hspace=0.30, wspace=0.28,
        )
        self._ax_hdr  = self.fig.add_subplot(gs[0, :])
        self._ax_ecg  = self.fig.add_subplot(gs[1, :])
        self._ax_cls  = self.fig.add_subplot(gs[2, :2])
        self._ax_prob = self.fig.add_subplot(gs[2, 2])
        self._ax_log  = self.fig.add_subplot(gs[2, 3])

        self._build_header()
        self._build_ecg()
        self._build_cls()
        self._build_probs()
        self._build_log()

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        ax = self._ax_hdr
        ax.set_facecolor(_PANEL2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ('top', 'bottom', 'left', 'right'):
            ax.spines[s].set_visible(s == 'bottom')
        ax.spines['bottom'].set_edgecolor(_BORDER)
        ax.spines['bottom'].set_linewidth(1.5)

        # Left: app title
        ax.text(0.01, 0.65, 'ECG CLASSIFICATION MONITOR',
                transform=ax.transAxes,
                color=_TEXT_HI, fontsize=11, fontweight='bold', va='center')
        ax.text(0.01, 0.18, 'NXP i.MX 8M Plus NPU  ·  PhysioNet 2017  ·  300 Hz',
                transform=ax.transAxes,
                color=_TEXT_DIM, fontsize=7.5, va='center')

        # Centre: record info
        self._hdr_sub = ax.text(0.38, 0.50, self.title,
                                transform=ax.transAxes,
                                color=_ACCENT, fontsize=9, va='center',
                                ha='left')

        # Right: BPM
        ax.text(0.72, 0.70, 'HEART RATE',
                transform=ax.transAxes,
                color=_TEXT_DIM, fontsize=6.5, va='center', fontweight='bold')
        self._bpm_val = ax.text(0.72, 0.18, '———',
                                transform=ax.transAxes,
                                color=_ACCENT, fontsize=19, fontweight='bold',
                                va='center', family='monospace')
        ax.text(0.835, 0.20, 'BPM',
                transform=ax.transAxes,
                color=_TEXT_MID, fontsize=9, va='center')

        # Right: LIVE
        self._live_dot = ax.text(0.925, 0.50, '●  LIVE',
                                 transform=ax.transAxes,
                                 color='#f87171', fontsize=10,
                                 fontweight='bold', va='center',
                                 path_effects=[
                                     pe.withStroke(linewidth=5,
                                                   foreground='#f87171',
                                                   alpha=0.25),
                                 ])

    # ── ECG ───────────────────────────────────────────────────────────────────

    def _build_ecg(self) -> None:
        ax = self._ax_ecg
        ax.set_facecolor(_BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_edgecolor(_BORDER)
        ax.spines['bottom'].set_edgecolor(_BORDER)

        t = np.linspace(0, 10, WINDOW_SAMPLES)
        z = np.zeros(WINDOW_SAMPLES)

        # Graph-paper grid — ultra-subtle
        for x in np.arange(0.2, 10, 0.2):
            ax.axvline(x, color='#0b1e33', lw=0.5, zorder=1)
        for x in range(1, 10):
            ax.axvline(x, color='#0f2640', lw=1.0, zorder=1)
        for y in np.arange(-3.5, 3.6, 0.5):
            ax.axhline(y, color='#0b1e33', lw=0.5, zorder=1)
        for y in np.arange(-3, 3.1, 1):
            ax.axhline(y, color='#0f2640', lw=1.0, zorder=1)

        # ECG glow stack
        self._eg3, = ax.plot(t, z, color=_ECG, lw=16, alpha=0.018, zorder=2)
        self._eg2, = ax.plot(t, z, color=_ECG, lw=8,  alpha=0.055, zorder=3)
        self._eg1, = ax.plot(t, z, color=_ECG, lw=3,  alpha=0.20,  zorder=4)
        self._eln, = ax.plot(t, z, color=_ECG_CORE, lw=0.9, alpha=0.95, zorder=5)

        # R-peak markers
        self._peak_dots, = ax.plot([], [], 'o',
                                   color='#fde047', ms=5, alpha=0.85,
                                   zorder=6, markeredgewidth=0,
                                   path_effects=[
                                       pe.withStroke(linewidth=8,
                                                     foreground='#fde047',
                                                     alpha=0.25),
                                   ])

        # Playhead
        ax.axvline(x=10, color='#f87171', lw=1.0, alpha=0.5, zorder=7)

        ax.set_xlim(0, 10)
        ax.set_ylim(-4.2, 4.2)
        ax.set_xlabel('Time (seconds)', color=_TEXT_DIM, fontsize=8)
        ax.set_yticks([])
        ax.tick_params(colors=_TEXT_DIM, labelsize=7.5, length=3)
        ax.set_xticks(range(11))

        self._t = t

    # ── Classification ────────────────────────────────────────────────────────

    def _build_cls(self) -> None:
        ax = self._ax_cls
        ax.set_facecolor(_PANEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)
            sp.set_linewidth(1.2)

        c = CLASS_COLORS[0]

        # Subtle panel tint (colour changes with class)
        self._cls_tint = Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            facecolor=c, alpha=0.05, zorder=0,
        )
        ax.add_patch(self._cls_tint)

        # Left accent stripe
        self._cls_stripe = Rectangle(
            (0, 0), 0.018, 1, transform=ax.transAxes,
            facecolor=c, alpha=1.0, zorder=2,
        )
        ax.add_patch(self._cls_stripe)

        # "CLASSIFICATION" label
        ax.text(0.06, 0.91, 'CLASSIFICATION',
                transform=ax.transAxes,
                color=_TEXT_DIM, fontsize=7, fontweight='bold',
                va='top')

        # Big class name
        self._cls_lbl = ax.text(
            0.52, 0.54, 'Normal',
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=42, fontweight='bold', color=c,
            path_effects=[
                pe.withStroke(linewidth=18, foreground=c, alpha=0.12),
                pe.withStroke(linewidth=6,  foreground=c, alpha=0.40),
            ],
        )

        # Confidence
        self._cls_conf = ax.text(
            0.52, 0.22, '25%  confidence',
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=13, color=_TEXT_MID,
        )

        # True-label feedback (no unicode — uses plain ASCII)
        self._cls_true = ax.text(
            0.52, 0.07, '',
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=8.5, color=_TEXT_DIM, style='italic',
        )

    # ── Probability bars ──────────────────────────────────────────────────────

    def _build_probs(self) -> None:
        ax = self._ax_prob
        ax.set_facecolor(_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)
            sp.set_linewidth(1.2)
        ax.text(0.04, 1.04, 'CLASS PROBABILITIES',
                transform=ax.transAxes,
                color=_TEXT_DIM, fontsize=7, fontweight='bold',
                va='bottom')

        y = np.arange(4)

        # Background track (full-width dim bar)
        ax.barh(y, [1.0]*4, color='#0f2036', height=0.52, zorder=1)
        # Foreground bars
        self._pb_bars = ax.barh(y, [0.25]*4, color=CLASS_COLORS,
                                height=0.52, zorder=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.65, 3.65)
        ax.set_yticks(y)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9.5, color=_TEXT_MID)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(['', '50%', '100%'], fontsize=7.5, color=_TEXT_DIM)
        ax.tick_params(colors=_TEXT_DIM, length=0)
        ax.axvline(x=0.5, color=_BORDER, lw=0.8, linestyle='--', zorder=3)

        # Text starts outside bars; _redraw_probs adjusts per value
        self._pb_pct = [
            ax.text(0.28, i, '25%', va='center', fontsize=8.5,
                    color='white', fontweight='bold', zorder=4)
            for i in y
        ]

    # ── Event log ─────────────────────────────────────────────────────────────

    def _build_log(self) -> None:
        ax = self._ax_log
        ax.set_facecolor(_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)
            sp.set_linewidth(1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, _HISTORY_LEN - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0.05, _HISTORY_LEN - 0.1, 'EVENT LOG',
                va='bottom', color=_TEXT_DIM, fontsize=7, fontweight='bold')

        self._log_squares = []
        self._log_names   = []
        self._log_pcts    = []
        for i in range(_HISTORY_LEN):
            y = _HISTORY_LEN - 1 - i   # 0=bottom, top=newest
            sq = FancyBboxPatch(
                (-0.01, y - 0.35), 0.10, 0.70,
                boxstyle='round,pad=0.01',
                facecolor=_BORDER, edgecolor='none',
                zorder=2,
            )
            ax.add_patch(sq)
            nm = ax.text(0.16, y, '---', va='center', ha='left',
                         fontsize=9, color=_TEXT_DIM)
            pt = ax.text(0.97, y, '', va='center', ha='right',
                         fontsize=8, color=_TEXT_DIM)
            self._log_squares.append(sq)
            self._log_names.append(nm)
            self._log_pcts.append(pt)

        # Separator below newest slot
        self._log_sep = ax.axhline(
            y=_HISTORY_LEN - 1.5, color=_BORDER, lw=0.8, linestyle='--',
        )

        self._log_acc = ax.text(0.5, -0.38, '', va='center', ha='center',
                                fontsize=7, color=_TEXT_DIM)

    # ── Animation ─────────────────────────────────────────────────────────────

    def _update(self, _f: int) -> None:
        self._frame += 1

        # Drain queue
        new: list[float] = []
        new_lbl: Optional[str] = None
        try:
            while True:
                s, lbl = self._queue.get_nowait()
                new.append(s)
                if lbl is not None:
                    new_lbl = lbl
        except queue.Empty:
            pass

        if new_lbl is not None:
            self._true_lbl = new_lbl

        if new:
            n = len(new)
            self._buffer = np.roll(self._buffer, -n)
            self._buffer[-n:] = new
            self._since_cls  += n
            if self._since_cls >= self.classify_every_n:
                self._since_cls = 0
                self._classify()

        # Z-score for display
        buf = self._buffer.copy()
        std = buf.std()
        disp = (buf - buf.mean()) / std if std > 1e-6 else np.zeros_like(buf)

        for ln in (self._eg3, self._eg2, self._eg1, self._eln):
            ln.set_ydata(disp)

        if std > 1e-6:
            pk, _ = find_peaks(disp, height=1.2, distance=int(SAMPLE_RATE * 0.35))
            if len(pk):
                self._peak_dots.set_data(self._t[pk], disp[pk])
                # Beep once per peak. A peak is "new" when it appears near
                # the right edge (i.e. arrived in this frame's batch of samples).
                # We track the rightmost peak index and beep only when a new
                # one appears to the right of the previous rightmost.
                rightmost = int(pk[-1])
                prev_right = int(self._last_peaks[-1]) if len(self._last_peaks) else -1
                if rightmost > prev_right:
                    self._audio.play('beep')
                self._last_peaks = pk
            else:
                self._peak_dots.set_data([], [])
                self._last_peaks = np.array([], dtype=int)
        else:
            self._peak_dots.set_data([], [])
            self._last_peaks = np.array([], dtype=int)

        # BPM (every ~0.5 s)
        if self._frame % 15 == 0 and std > 1e-6:
            self._bpm = _estimate_bpm(disp)
            self._bpm_val.set_text(
                str(self._bpm) if self._bpm else '———'
            )

        # LIVE blink
        self._live_dot.set_alpha(1.0 if self._frame % 36 < 18 else 0.20)

        # Accent stripe flash
        if self._flash > 0:
            self._flash -= 1
            t = self._flash / 10.0
            self._cls_stripe.set_alpha(0.6 + t * 0.4)
        else:
            self._cls_stripe.set_alpha(1.0)

    def _classify(self) -> None:
        pred, name, probs = self.inference_fn(self._buffer.copy())
        self._pred  = pred
        self._name  = name
        self._probs = probs
        self._total += 1
        self._flash  = 10
        self._audio.play(name)   # class chime on every new classification

        true_name = _LABEL_MAP.get(self._true_lbl or '', None)
        ok = (true_name == name) if true_name else None
        if ok:
            self._correct += 1
        self._history.append((pred, name, float(probs[pred]), ok))

        self._redraw_cls(pred, name, probs, true_name, ok)
        self._redraw_probs(probs)
        self._redraw_log()

    def _redraw_cls(self, pred, name, probs, true_name, ok) -> None:
        c    = CLASS_COLORS[pred]
        conf = probs[pred]

        self._cls_tint.set_facecolor(c)
        self._cls_stripe.set_facecolor(c)

        self._cls_lbl.set_text(name)
        self._cls_lbl.set_color(c)
        self._cls_lbl.set_path_effects([
            pe.withStroke(linewidth=18, foreground=c, alpha=0.12),
            pe.withStroke(linewidth=6,  foreground=c, alpha=0.40),
        ])

        self._cls_conf.set_text(f'{conf:.1%}  confidence')

        if true_name:
            if ok:
                self._cls_true.set_text(f'correct  (true: {true_name})')
                self._cls_true.set_color('#4ade80')
            else:
                self._cls_true.set_text(f'wrong  --  true: {true_name}')
                self._cls_true.set_color('#f87171')
        else:
            self._cls_true.set_text('')

    def _redraw_probs(self, probs) -> None:
        for bar, pct, v in zip(self._pb_bars, self._pb_pct, probs):
            w = float(v)
            bar.set_width(w)
            pct.set_text(f'{v:.0%}')
            if w >= 0.18:
                # Inside the bar
                pct.set_x(min(w - 0.04, 0.82))
                pct.set_ha('right')
                pct.set_color('white')
            else:
                # Outside (to the right of) the bar
                pct.set_x(w + 0.04)
                pct.set_ha('left')
                pct.set_color(_TEXT_MID)

    def _redraw_log(self) -> None:
        h = list(self._history)
        for slot in range(_HISTORY_LEN):
            hi = len(h) - 1 - slot
            sq   = self._log_squares[slot]
            name = self._log_names[slot]
            pct  = self._log_pcts[slot]

            if hi >= 0:
                pred, lname, conf, ok = h[hi]
                c = CLASS_COLORS[pred]
                alpha = max(0.30, 1.0 - slot * 0.13)
                sq.set_facecolor(c)
                name.set_text(lname)
                name.set_color(c)
                mark = '' if ok is None else ('  ok' if ok else '  !!')
                pct.set_text(f'{conf:.0%}{mark}')
                pct.set_color(c)
                for obj in (name, pct):
                    obj.set_alpha(alpha)
                sq.set_alpha(alpha * 0.85)
            else:
                sq.set_facecolor(_BORDER)
                sq.set_alpha(0.4)
                name.set_text('---')
                name.set_color(_TEXT_DIM)
                name.set_alpha(0.4)
                pct.set_text('')

        if self._total and self._true_lbl is not None:
            acc = self._correct / self._total
            self._log_acc.set_text(
                f'session  {self._correct}/{self._total}  ({acc:.0%})'
            )
        else:
            self._log_acc.set_text('')
