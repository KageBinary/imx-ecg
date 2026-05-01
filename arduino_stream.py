"""
arduino_stream.py — Live Arduino ECG → dashboard + raw data save.

Auto-detects the Arduino serial port, reads raw ADC samples (0–1023),
feeds them into the ECGDashboard, and optionally saves raw samples so
you can inspect the signal and decide what preprocessing is needed.

Usage
-----
    # Auto-detect port, display only:
    python arduino_stream.py

    # Save raw data (auto-named raw_<timestamp>.npy):
    python arduino_stream.py --save

    # Save to a specific file:
    python arduino_stream.py --save-to raw_ecg.npy
    python arduino_stream.py --save-to raw_ecg.csv

    # Apply 0.5–40 Hz bandpass before display / classification:
    python arduino_stream.py --filter

    # Skip classifier (display only, no model needed):
    python arduino_stream.py --no-classify

    # Explicit port / baud:
    python arduino_stream.py --port /dev/ttyACM1 --baud 115200

    # WSL2: pass the Windows COM port directly (COM3 → /dev/ttyS3 auto-resolved):
    python arduino_stream.py --port COM3

Arguments
---------
    --port           Serial port; auto-detected if omitted
    --baud           Baud rate (default: 115200)
    --fs             Arduino sample rate in Hz (default: 250)
    --checkpoint     Model checkpoint (default: checkpoints/deploy_best.pt)
    --save           Auto-name a .npy save file → raw_<timestamp>.npy
    --save-to        Explicit save path (.npy or .csv)
    --filter         Apply Butterworth bandpass before display (default 0.5–40 Hz)
    --hp             High-pass cutoff in Hz (default: 0.5; try 1.0 for less baseline wander)
    --no-classify    Disable inference, display waveform only
    --classify-every Reclassify every N samples (default: 125 ≈ 0.5 s at 250 Hz)
"""
from __future__ import annotations

import argparse
import atexit
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import math
import subprocess

import numpy as np
import serial
import serial.tools.list_ports
from scipy.signal import butter, lfilter, lfilter_zi, resample_poly

from dashboard import ECGDashboard
from dashboard_lite import ECGDashboardLite

# ── Constants ─────────────────────────────────────────────────────────────────

# VIDs for common microcontroller boards
_ARDUINO_VIDS = {
    0x2341,  # Arduino SA (Uno, Mega, Leonardo …)
    0x2A03,  # Arduino.org
    0x1A86,  # CH340 / CH341 (cheap clones)
    0x10C4,  # CP2102 / CP2104 (Silicon Labs)
    0x0403,  # FTDI FT232
    0x2E8A,  # Raspberry Pi (Pico / Pico W, MicroPython)
}

# Raw samples saved across the session; flushed on exit
_raw_samples: list[float] = []
_save_path: Optional[str] = None


# ── WSL2 helpers ──────────────────────────────────────────────────────────────

def _is_wsl2() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _com_to_ttyS(com: str) -> str:
    """Convert 'COM3' (case-insensitive) to '/dev/ttyS3'."""
    num = com.upper().replace("COM", "").strip()
    return f"/dev/ttyS{num}"


def _wsl2_find_arduino_port() -> Optional[str]:
    """
    Ask Windows via PowerShell for COM ports that look like Arduinos,
    then return the corresponding /dev/ttySN path.
    Falls back to the lowest-numbered COM port if no Arduino keyword found.
    """
    try:
        result = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "Get-WMIObject Win32_SerialPort | "
                "Select-Object DeviceID, Description | "
                "ConvertTo-Csv -NoTypeInformation",
            ],
            capture_output=True, text=True, timeout=8,
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            return None

        arduino_ports: list[str] = []
        all_ports: list[str] = []
        for line in lines[1:]:  # skip CSV header
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) < 2:
                continue
            dev_id, desc = parts[0], parts[1]
            if not dev_id.upper().startswith("COM"):
                continue
            all_ports.append(dev_id)
            if any(k in desc.lower() for k in ("arduino", "ch340", "ch341",
                                                "cp210", "ftdi", "usb serial",
                                                "pico", "micropython", "raspberry")):
                arduino_ports.append(dev_id)

        chosen = arduino_ports[0] if arduino_ports else (all_ports[0] if all_ports else None)
        return _com_to_ttyS(chosen) if chosen else None

    except Exception:
        return None


def _wsl2_list_ports_human() -> str:
    """Return a human-readable list of Windows COM ports (for error messages)."""
    try:
        result = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "Get-WMIObject Win32_SerialPort | "
                "Select-Object DeviceID, Description | "
                "ConvertTo-Csv -NoTypeInformation",
            ],
            capture_output=True, text=True, timeout=8,
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            return "  (no COM ports found on Windows side)"
        out = []
        for line in lines[1:]:
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 2:
                com, desc = parts[0], parts[1]
                tty = _com_to_ttyS(com) if com.upper().startswith("COM") else ""
                out.append(f"  {com} → {tty}  ({desc})")
        return "\n".join(out) if out else "  (no COM ports found on Windows side)"
    except Exception as e:
        return f"  (PowerShell query failed: {e})"


# ── Port detection ─────────────────────────────────────────────────────────────

def find_arduino_port() -> Optional[str]:
    """
    Return the first likely Arduino serial port.

    Always tries native Linux pyserial scan first (covers usbipd-attached
    devices on WSL2 as well as bare Linux). Falls back to querying Windows
    COM ports via PowerShell when running in WSL2 and nothing was found
    natively (i.e. the device is not yet attached via usbipd).
    """
    # Native Linux scan — works on bare Linux and WSL2 after usbipd attach
    acm_usb: list[str] = []
    for p in serial.tools.list_ports.comports():
        vid = p.vid or 0
        name = f"{p.description or ''} {p.manufacturer or ''}".lower()
        if vid in _ARDUINO_VIDS or any(k in name for k in ("arduino", "ch340", "pico", "micropython", "raspberry")):
            return p.device
        if p.device.startswith("/dev/ttyACM") or p.device.startswith("/dev/ttyUSB"):
            acm_usb.append(p.device)
    if acm_usb:
        return acm_usb[0]

    # WSL2 fallback: device not yet attached via usbipd, try Windows COM ports
    if _is_wsl2():
        return _wsl2_find_arduino_port()

    return None


def list_ports_human() -> str:
    if _is_wsl2():
        return _wsl2_list_ports_human()
    ports = serial.tools.list_ports.comports()
    if not ports:
        return "  (no serial ports found)"
    return "\n".join(
        f"  {p.device}  {p.description or ''}  VID={hex(p.vid) if p.vid else 'n/a'}"
        for p in ports
    )


# ── Filters ───────────────────────────────────────────────────────────────────

def _make_bandpass(fs: float, hp: float = 1.0, lp: float = 40.0, order: int = 2):
    """Returns (b_hp, a_hp, zi_hp, b_lp, a_lp, zi_lp) for a two-stage filter."""
    nyq = fs / 2.0
    b_hp, a_hp = butter(order, hp / nyq, btype="high")
    b_lp, a_lp = butter(order, lp / nyq, btype="low")
    zi_hp = lfilter_zi(b_hp, a_hp) * 0.0
    zi_lp = lfilter_zi(b_lp, a_lp) * 0.0
    return b_hp, a_hp, zi_hp, b_lp, a_lp, zi_lp


# ── Save helpers ──────────────────────────────────────────────────────────────

def _save_on_exit() -> None:
    if not _raw_samples or _save_path is None:
        return
    arr = np.array(_raw_samples, dtype=np.float32)
    path = Path(_save_path)
    if path.suffix.lower() == ".csv":
        np.savetxt(str(path), arr, delimiter=",", fmt="%.2f")
    else:
        np.save(str(path), arr)
    print(f"\nSaved {len(arr)} raw samples → {path}  (shape {arr.shape})")


def _sigint_handler(sig, frame) -> None:
    _save_on_exit()
    sys.exit(0)


# ── Serial reader thread ──────────────────────────────────────────────────────

def _resample_ratio(fs_in: float, fs_out: float) -> tuple[int, int]:
    """Return (up, down) integer ratio for resample_poly(up=up, down=down)."""
    scale = 1_000_000  # work in micro-Hz to stay integer
    g = math.gcd(round(fs_in * scale), round(fs_out * scale))
    return round(fs_out * scale) // g, round(fs_in * scale) // g


def _read_thread(
    dashboard: ECGDashboard,
    port: str,
    baud: int,
    use_filter: bool,
    fs: float,
    hp_cutoff: float = 1.0,
) -> None:
    """
    Open the serial port and push samples to the dashboard indefinitely.
    Each line from the Arduino must be a single integer (0–1023).

    Pipeline per sample:
      1. Save raw ADC integer to _raw_samples (unprocessed, for inspection)
      2. Optionally apply 0.5–40 Hz bandpass filter (sample-by-sample, stateful)
      3. Accumulate into a resample buffer; every CHUNK_IN samples, resample
         to 300 Hz and push the resulting CHUNK_OUT samples to the dashboard

    Resampling corrects the 250 Hz → 300 Hz mismatch so BPM and
    classification both use the correct time base.
    """
    global _raw_samples

    _DASHBOARD_FS = 300
    _CHUNK_IN     = 50   # accumulate this many input samples before resampling
    up, down = _resample_ratio(fs, _DASHBOARD_FS)
    _CHUNK_OUT = round(_CHUNK_IN * up / down)
    needs_resample = (up != down)

    if needs_resample:
        print(f"Resampling {fs:.0f} Hz → {_DASHBOARD_FS} Hz  (ratio {up}/{down}, "
              f"chunk {_CHUNK_IN} → {_CHUNK_OUT} samples)")

    filter_state = _make_bandpass(fs, hp=hp_cutoff) if use_filter else None

    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {port}: {e}")
        sys.exit(1)

    print(f"Reading from {port} at {baud} baud …  (Ctrl-C to stop)")

    b_hp = a_hp = zi_hp = b_lp = a_lp = zi_lp = None
    if filter_state:
        b_hp, a_hp, zi_hp, b_lp, a_lp, zi_lp = filter_state

    resample_buf: list[float] = []

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
        except serial.SerialException:
            print("Serial read error — port disconnected?")
            break

        try:
            raw = int(line)
        except ValueError:
            continue  # skip non-numeric lines (e.g. boot messages)

        if not (0 <= raw <= 1023):
            continue

        # Always save raw ADC value before any processing
        _raw_samples.append(float(raw))
        if len(_raw_samples) % 25 == 0:
            print(f"[DATA] samples_rx={len(_raw_samples)}  raw={raw}", flush=True)

        sample = float(raw)

        if use_filter and b_hp is not None:
            x = np.array([sample])
            y1, zi_hp = lfilter(b_hp, a_hp, x, zi=zi_hp)
            y2, zi_lp = lfilter(b_lp, a_lp, y1, zi=zi_lp)
            sample = float(y2[0])

        if needs_resample:
            resample_buf.append(sample)
            if len(resample_buf) >= _CHUNK_IN:
                chunk = np.array(resample_buf[:_CHUNK_IN], dtype=np.float32)
                resample_buf = resample_buf[_CHUNK_IN:]
                resampled = resample_poly(chunk, up=up, down=down).astype(np.float32)
                dashboard.push_samples(resampled)
        else:
            dashboard.push_sample(sample)


# ── Dummy inference (--no-classify) ──────────────────────────────────────────

def _dummy_classify(signal: np.ndarray):
    probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    return 0, "Normal", probs


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream live Arduino ECG into the real-time dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port",           default=None,
                   help="Serial port (auto-detected if omitted)")
    p.add_argument("--baud",           type=int, default=115200)
    p.add_argument("--fs",             type=float, default=250.0,
                   help="Arduino sample rate in Hz")
    p.add_argument("--checkpoint",     default="checkpoints/deploy_best.pt",
                   help="PyTorch checkpoint for classification")
    p.add_argument("--tflite",         default=None, metavar="PATH",
                   help="TFLite model path (uses NPU delegate on i.MX board); "
                        "e.g. artifacts/ecg_deploy_int8.tflite")
    p.add_argument("--save",           action="store_true",
                   help="Save raw samples on exit (auto-named raw_<timestamp>.npy)")
    p.add_argument("--save-to",        default=None, metavar="PATH",
                   help="Explicit save path (.npy or .csv); implies --save")
    p.add_argument("--filter",         action="store_true",
                   help="Apply Butterworth bandpass before display")
    p.add_argument("--hp",             type=float, default=0.5,
                   help="High-pass cutoff Hz (default: 0.5; 1.0 reduces baseline wander)")
    p.add_argument("--no-classify",    action="store_true",
                   help="Skip model inference — display waveform only")
    p.add_argument("--lite",           action="store_true",
                   help="Lightweight dashboard: 2.5s window, 10fps, no audio (good for the board)")
    p.add_argument("--classify-every", type=int, default=125,
                   help="Reclassify every N new samples")
    return p.parse_args(argv)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    global _save_path

    args = parse_args(argv)

    # ── Resolve save path ─────────────────────────────────────────────────────
    if args.save_to:
        _save_path = args.save_to
    elif args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_path = f"raw_{ts}.npy"

    if _save_path:
        atexit.register(_save_on_exit)
        signal.signal(signal.SIGINT, _sigint_handler)
        print(f"Raw samples will be saved to: {_save_path}")

    # ── Resolve serial port ───────────────────────────────────────────────────
    port = args.port

    # Allow passing Windows COM port names directly (e.g. --port COM3).
    # After usbipd attach the device lives at /dev/ttyACM0, not /dev/ttySN,
    # so fall back to auto-detect if the mapped path doesn't exist.
    if port is not None and port.upper().startswith("COM"):
        tty_s = _com_to_ttyS(port)
        if Path(tty_s).exists():
            port = tty_s
            print(f"Resolved {args.port} → {port}")
        else:
            print(f"{args.port} → {tty_s} not found; trying auto-detect (device may be attached via usbipd)")
            port = None  # let auto-detect run below

    if port is None:
        port = find_arduino_port()
        if port is None:
            print("Could not auto-detect an Arduino port.\n")
            if _is_wsl2():
                print("WSL2 detected. Available Windows COM ports:")
                print(list_ports_human())
                print("\nSpecify one with --port COM3  (or whichever number the Arduino is on)")
            else:
                print("Available ports:")
                print(list_ports_human())
                print("\nSpecify one with --port /dev/ttyACMx")
            sys.exit(1)
        print(f"Auto-detected Arduino on: {port}")
    else:
        print(f"Using port: {port}")

    # ── Load inference ────────────────────────────────────────────────────────
    if args.no_classify:
        print("Classification disabled — waveform display only.")
        inference_fn = _dummy_classify
        title = f"Arduino live  |  {port}  |  {args.fs:.0f} Hz  |  raw display"
    elif args.tflite:
        tflite_path = Path(args.tflite)
        if not tflite_path.exists():
            print(f"TFLite model not found: {tflite_path}")
            sys.exit(1)
        print(f"Loading TFLite model from {tflite_path} …")
        from ecg_inference import ECGInferenceTFLite
        inference = ECGInferenceTFLite(str(tflite_path))
        inference_fn = inference.classify
        title = f"ECG live  |  {port}  |  {args.fs:.0f} Hz  |  TFLite"
    else:
        ck_path = Path(args.checkpoint)
        if not ck_path.exists():
            print(f"Checkpoint not found: {ck_path}")
            print("Run with --no-classify to display without inference, or --tflite <path> for TFLite.")
            sys.exit(1)

        print(f"Loading model from {ck_path} …")
        from ecg_inference import ECGInference
        inference = ECGInference(str(ck_path))
        inference_fn = inference.classify
        title = f"Arduino live  |  {port}  |  {args.fs:.0f} Hz"

    if args.filter:
        title += "  |  0.5–40 Hz filtered"
    if _save_path:
        title += f"  |  saving → {Path(_save_path).name}"

    # ── Build dashboard ───────────────────────────────────────────────────────
    DashClass = ECGDashboardLite if args.lite else ECGDashboard
    dashboard = DashClass(
        inference_fn=inference_fn,
        classify_every_n=args.classify_every,
        title=title,
        sample_rate=int(args.fs),
    )

    if args.fs != 300.0:
        print(f"Arduino FS={args.fs:.0f} Hz → will resample to 300 Hz before display/classification.")

    # ── Start serial reader ───────────────────────────────────────────────────
    t = threading.Thread(
        target=_read_thread,
        args=(dashboard, port, args.baud, args.filter, args.fs, args.hp),
        daemon=True,
    )
    t.start()

    dashboard.run()

    # atexit handles the save when the matplotlib window closes
    _save_on_exit()


if __name__ == "__main__":
    main()
