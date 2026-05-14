"""
lowpass_presets.py — Batch-apply 18 low-pass filter presets to a WAV file.

Sekali input, langsung output 18 file WAV dengan preset low-pass berbeda-beda.
Menggunakan algoritma Butterworth IIR 2nd-order (biquad) — sama seperti Audacity.

Usage:
    python lowpass_presets.py input.wav [--outdir OUTPUT_FOLDER] [--list]

Options:
    --outdir   Folder output (default: folder yang sama dengan input file)
    --list     Tampilkan daftar 18 preset tanpa memproses

Example:
    python lowpass_presets.py vocals.wav
    python lowpass_presets.py vocals.wav --outdir hasil_lowpass
"""

import wave
import struct
import argparse
import os
import sys
import math
import time

# ─────────────────────────────────────────────────────────────────────────────
# 18 PRESET LOW-PASS
# (slug, label, cutoff_hz, order)
#
# cutoff_hz : frekuensi potong — semua di atas ini akan diredam
# order     : 1 = lembut, 2 = standard Butterworth, 4 = steep
# ─────────────────────────────────────────────────────────────────────────────
PRESETS = [
    # ── Telephone / Lo-Fi ──
    ("01_telephone",         "Telephone",          800,    2),
    ("02_am_radio",          "AM Radio",           3000,   2),
    ("03_old_cassette",      "Old Cassette",       6000,   2),

    # ── Warmth / Vintage ──
    ("04_vintage_warm",      "Vintage Warm",       4000,   1),
    ("05_vinyl_warmth",      "Vinyl Warmth",       8000,   2),
    ("06_tube_warmth",       "Tube Warmth",        10000,  1),

    # ── Gentle Roll-off ──
    ("07_air_cut_soft",      "Air Cut Soft",       12000,  1),
    ("08_air_cut_medium",    "Air Cut Medium",     10000,  2),
    ("09_presence_tame",     "Presence Tame",      8000,   1),

    # ── Mid Scoop / Body ──
    ("10_body_focus",        "Body Focus",         5000,   2),
    ("11_mud_maker",         "Mud Maker",          2500,   2),
    ("12_bass_only",         "Bass Only",          300,    2),

    # ── Steep / Aggressive ──
    ("13_steep_8k",          "Steep 8kHz",         8000,   4),
    ("14_steep_5k",          "Steep 5kHz",         5000,   4),
    ("15_steep_3k",          "Steep 3kHz",         3000,   4),

    # ── Sub / Ultra Low ──
    ("16_sub_rumble",        "Sub Rumble",         120,    2),
    ("17_deep_bass",         "Deep Bass",          200,    2),

    # ── Full Range Reference ──
    ("18_silky_top",         "Silky Top",          15000,  2),
]

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_wav(path: str):
    with wave.open(path, "rb") as wf:
        params = wf.getparams()
        raw    = wf.readframes(params.nframes)

    sw    = params.sampwidth
    nc    = params.nchannels
    nf    = params.nframes
    total = nf * nc

    if sw == 1:
        samples_raw   = struct.unpack(f"{total}B", raw)
        samples_float = [(s - 128) / 128.0 for s in samples_raw]
    elif sw == 2:
        samples_raw   = struct.unpack(f"<{total}h", raw)
        samples_float = [s / 32768.0 for s in samples_raw]
    elif sw == 3:
        samples_float = []
        for i in range(0, len(raw), 3):
            val = int.from_bytes(raw[i:i+3], byteorder="little", signed=True)
            samples_float.append(val / 8388608.0)
    elif sw == 4:
        samples_raw   = struct.unpack(f"<{total}i", raw)
        samples_float = [s / 2147483648.0 for s in samples_raw]
    else:
        sys.exit(f"[ERROR] Unsupported sample width: {sw} bytes")

    channels = [samples_float[ch::nc] for ch in range(nc)]
    return params, channels, sw


def write_wav(path: str, params, channels, sampwidth: int):
    nc = len(channels)
    nf = len(channels[0])

    interleaved = []
    for i in range(nf):
        for ch in range(nc):
            interleaved.append(channels[ch][i])

    interleaved = [max(-1.0, min(1.0, s)) for s in interleaved]

    if sampwidth == 1:
        raw_samples = [int(s * 127 + 128) & 0xFF for s in interleaved]
        raw = struct.pack(f"{len(raw_samples)}B", *raw_samples)
    elif sampwidth == 2:
        raw_samples = [int(s * 32767) for s in interleaved]
        raw = struct.pack(f"<{len(raw_samples)}h", *raw_samples)
    elif sampwidth == 3:
        raw_parts = []
        for s in interleaved:
            val = max(-8388608, min(8388607, int(s * 8388607)))
            raw_parts.append(val.to_bytes(3, byteorder="little", signed=True))
        raw = b''.join(raw_parts)
    elif sampwidth == 4:
        raw_samples = [int(s * 2147483647) for s in interleaved]
        raw = struct.pack(f"<{len(raw_samples)}i", *raw_samples)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(nc)
        wf.setsampwidth(sampwidth)
        wf.setframerate(params.framerate)
        wf.writeframes(raw)


# ─────────────────────────────────────────────────────────────────────────────
# DSP — Butterworth Biquad Low-Pass Filter
# ─────────────────────────────────────────────────────────────────────────────

def butter_lowpass_coeffs(cutoff_hz: float, framerate: int) -> tuple:
    """
    Hitung koefisien biquad IIR Butterworth low-pass 2nd-order.
    Rumus: bilinear transform dari analog Butterworth prototype.
    Sama seperti yang dipakai Audacity / scipy.signal.butter.

    Returns (b0, b1, b2, a1, a2) — normalized (a0 = 1)
    """
    # Bilinear transform pre-warp
    w0  = 2.0 * math.pi * cutoff_hz / framerate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha  = sin_w0 / math.sqrt(2.0)   # Q = 1/sqrt(2) = Butterworth

    b0 =  (1.0 - cos_w0) / 2.0
    b1 =   1.0 - cos_w0
    b2 =  (1.0 - cos_w0) / 2.0
    a0 =   1.0 + alpha
    a1 =  -2.0 * cos_w0
    a2 =   1.0 - alpha

    return (b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)


def apply_biquad(channel: list, b0: float, b1: float, b2: float,
                 a1: float, a2: float) -> list:
    """Terapkan satu tahap biquad IIR ke channel (direct form II transposed)."""
    out = [0.0] * len(channel)
    w1 = 0.0
    w2 = 0.0
    for i, x in enumerate(channel):
        y   = b0 * x + w1
        w1  = b1 * x - a1 * y + w2
        w2  = b2 * x - a2 * y
        out[i] = y
    return out


def apply_lowpass(channel: list, framerate: int,
                  cutoff_hz: float, order: int) -> list:
    """
    Low-pass Butterworth order 1, 2, atau 4.
    Order 1  : satu biquad (soft roll-off  ~6 dB/oct)
    Order 2  : satu biquad Butterworth     (~12 dB/oct) — default Audacity
    Order 4  : dua biquad cascaded         (~24 dB/oct, steep)
    """
    # Clamp cutoff supaya tidak melewati Nyquist
    nyquist    = framerate / 2.0
    cutoff_hz  = min(cutoff_hz, nyquist * 0.999)

    coeffs = butter_lowpass_coeffs(cutoff_hz, framerate)
    b0, b1, b2, a1, a2 = coeffs

    if order == 1:
        # Single-pole RC approximation lewat 1 pass biquad dengan Q tinggi
        # (semua koefisien b2/a2 di-nol-kan → 1st order)
        w0     = 2.0 * math.pi * cutoff_hz / framerate
        alpha  = 1.0 / (1.0 + 1.0 / math.tan(w0 / 2.0))
        b0_1   = alpha
        b1_1   = alpha
        a1_1   = -(1.0 - 2.0 * alpha)
        # Jalankan sebagai biquad dengan b2=0, a2=0
        out = [0.0] * len(channel)
        x_prev = 0.0
        y_prev = 0.0
        for i, x in enumerate(channel):
            y      = b0_1 * x + b1_1 * x_prev - a1_1 * y_prev
            out[i] = y
            x_prev = x
            y_prev = y
        return out

    elif order == 2:
        return apply_biquad(channel, b0, b1, b2, a1, a2)

    elif order == 4:
        # Cascade dua biquad identik → slope 2x
        stage1 = apply_biquad(channel,  b0, b1, b2, a1, a2)
        stage2 = apply_biquad(stage1,   b0, b1, b2, a1, a2)
        return stage2

    else:
        # Fallback: order 2
        return apply_biquad(channel, b0, b1, b2, a1, a2)


# ─────────────────────────────────────────────────────────────────────────────
# Batch
# ─────────────────────────────────────────────────────────────────────────────

def print_preset_table():
    print()
    print("┌────┬─────────────────────────┬────────────┬───────┐")
    print("│ No │ Preset Name             │ Cutoff Hz  │ Order │")
    print("├────┼─────────────────────────┼────────────┼───────┤")
    for i, (slug, label, cutoff, order) in enumerate(PRESETS, 1):
        print(f"│ {i:2d} │ {label:<23} │  {cutoff:>7} Hz │   {order}   │")
    print("└────┴─────────────────────────┴────────────┴───────┘")
    print()


def batch_process(input_path: str, outdir: str):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Input     : {input_path}")
    print(f"  Output dir: {outdir}")
    print(f"{'='*62}\n")

    print("Membaca file input …")
    t0 = time.time()
    params, channels, sampwidth = read_wav(input_path)
    read_time = time.time() - t0

    duration_s = params.nframes / params.framerate
    print(f"  Channels  : {params.nchannels}")
    print(f"  Rate      : {params.framerate} Hz")
    print(f"  Bit depth : {sampwidth * 8}-bit")
    print(f"  Duration  : {duration_s:.2f} s  ({params.nframes} frames)")
    print(f"  Read time : {read_time:.2f} s\n")

    total_start = time.time()
    success = 0

    for idx, (slug, label, cutoff, order) in enumerate(PRESETS, 1):
        out_filename = f"{base_name}_{slug}.wav"
        out_path     = os.path.join(outdir, out_filename)

        bar = f"[{idx:2d}/18]"
        print(f"{bar} {label:<23}", end="  ", flush=True)

        t1 = time.time()
        processed = []
        for ch_data in channels:
            processed.append(
                apply_lowpass(ch_data, params.framerate, cutoff, order)
            )
        write_wav(out_path, params, processed, sampwidth)
        elapsed = time.time() - t1

        print(f"cutoff={cutoff} Hz  order={order}  →  {out_filename}  ({elapsed:.1f}s)")
        success += 1

    total_time = time.time() - total_start
    print()
    print(f"{'='*62}")
    print(f"  Selesai! {success}/18 file berhasil dibuat.")
    print(f"  Total waktu proses : {total_time:.1f} s")
    print(f"  Folder output      : {os.path.abspath(outdir)}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-apply 18 low-pass filter presets ke satu WAV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="?", help="File WAV input")
    p.add_argument("--outdir", default=None,
                   help="Folder output (default: folder yang sama dengan file input)")
    p.add_argument("--list", action="store_true",
                   help="Tampilkan tabel 18 preset dan keluar")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print_preset_table()
        return

    if not args.input:
        print("Usage: python lowpass_presets.py input.wav [--outdir FOLDER] [--list]")
        sys.exit(1)

    if not os.path.isfile(args.input):
        sys.exit(f"[ERROR] File tidak ditemukan: {args.input}")

    if not args.input.lower().endswith(".wav"):
        sys.exit("[ERROR] File harus berformat WAV.")

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.input))

    print_preset_table()
    batch_process(args.input, outdir)


if __name__ == "__main__":
    main()