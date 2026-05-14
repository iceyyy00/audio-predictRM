"""
highpass_presets.py — Batch-apply 18 high-pass filter presets to a WAV file.

Sekali input, langsung output 18 file WAV dengan preset high-pass berbeda-beda.
Menggunakan algoritma Butterworth IIR biquad — sama seperti Audacity.

Usage:
    python highpass_presets.py input.wav [--outdir OUTPUT_FOLDER] [--list]

Options:
    --outdir   Folder output (default: folder yang sama dengan input file)
    --list     Tampilkan daftar 18 preset tanpa memproses

Example:
    python highpass_presets.py vocals.wav
    python highpass_presets.py drums.wav --outdir hasil_highpass
"""

import wave
import struct
import argparse
import os
import sys
import math
import time

# ─────────────────────────────────────────────────────────────────────────────
# 18 PRESET HIGH-PASS
# (slug, label, cutoff_hz, order)
#
# cutoff_hz : frekuensi potong — semua di BAWAH ini akan diredam
# order     : 1=lembut (~6dB/oct), 2=standard Butterworth (~12dB/oct),
#             4=steep cascade (~24dB/oct)
# ─────────────────────────────────────────────────────────────────────────────
PRESETS = [
    # ── Sub-bass / Rumble Removal ───────────────────────────────────────────
    ("01_subsonic_cut",       "Subsonic Cut",        20,    2),
    # Buang infrasonic/subsonic — standard mastering, hampir tak terdengar

    ("02_rumble_remove",      "Rumble Remove",       40,    2),
    # Buang getaran lantai/AC/mic stand rumble

    ("03_dc_offset_cut",      "DC Offset Cut",       10,    4),
    # Steep cut sangat rendah untuk buang DC offset

    # ── Sub Cleanup ─────────────────────────────────────────────────────────
    ("04_sub_clean_soft",     "Sub Clean Soft",      60,    1),
    # Lembut, buang sub tapi tetap hangat

    ("05_sub_clean_medium",   "Sub Clean Medium",    80,    2),
    # Standard sub cleanup untuk vokal/gitar

    ("06_sub_clean_steep",    "Sub Clean Steep",     80,    4),
    # Steep version — benar-benar bersih dari sub

    # ── Vokal / Recording ───────────────────────────────────────────────────
    ("07_vocal_hpf_soft",     "Vocal HPF Soft",      100,   1),
    # Soft HP untuk vokal — buang proximity effect mic

    ("08_vocal_hpf_standard", "Vocal HPF Standard",  120,   2),
    # Standard setting banyak preamp/channel strip vokal

    ("09_vocal_hpf_tight",    "Vocal HPF Tight",     150,   2),
    # Lebih ketat — untuk vokal tipis/bright

    # ── Instrumen ───────────────────────────────────────────────────────────
    ("10_guitar_hpf",         "Guitar HPF",          80,    2),
    # Buang muddiness bawah gitar elektrik/akustik

    ("11_piano_hpf",          "Piano HPF",           40,    2),
    # Buang sub piano, tetap pertahankan bass nada rendah

    ("12_snare_hpf",          "Snare HPF",           200,   2),
    # Potong low-end snare — buang kick bleed

    # ── Mid/High Focus ──────────────────────────────────────────────────────
    ("13_presence_focus",     "Presence Focus",      300,   2),
    # Fokus ke mid ke atas — efek "hadir" di mix

    ("14_top_end_only",       "Top End Only",        500,   2),
    # Hanya mid-high ke atas — untuk efek kreatif/foley

    ("15_air_band",           "Air Band",            8000,  2),
    # Hanya frekuensi "udara" — untuk layer/parallel processing

    # ── Steep / Agresif ─────────────────────────────────────────────────────
    ("16_steep_200hz",        "Steep 200Hz",         200,   4),
    # Potong tegas semua di bawah 200 Hz

    ("17_steep_500hz",        "Steep 500Hz",         500,   4),
    # Potong tegas di bawah 500 Hz — efek lo-fi/telephone atas

    # ── Broadcast / Mastering ───────────────────────────────────────────────
    ("18_broadcast_hpf",      "Broadcast HPF",       100,   2),
    # Standard HPF broadcast/FM radio — EBU R68
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
# DSP — Butterworth Biquad High-Pass Filter
# ─────────────────────────────────────────────────────────────────────────────

def _hp_coeffs(cutoff_hz: float, framerate: int) -> tuple:
    """
    Koefisien biquad IIR Butterworth high-pass 2nd-order.
    Bilinear transform dari analog Butterworth prototype.
    """
    w0     = 2.0 * math.pi * cutoff_hz / framerate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha  = sin_w0 / math.sqrt(2.0)   # Q = 1/sqrt(2) = Butterworth

    b0 =  (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 =  (1.0 + cos_w0) / 2.0
    a0 =   1.0 + alpha
    a1 =  -2.0 * cos_w0
    a2 =   1.0 - alpha

    return (b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)


def _apply_biquad(channel: list, b0: float, b1: float, b2: float,
                  a1: float, a2: float) -> list:
    """Direct Form II Transposed biquad IIR."""
    out = [0.0] * len(channel)
    w1  = 0.0
    w2  = 0.0
    for i, x in enumerate(channel):
        y      = b0 * x + w1
        w1     = b1 * x - a1 * y + w2
        w2     = b2 * x - a2 * y
        out[i] = y
    return out


def apply_highpass(channel: list, framerate: int,
                   cutoff_hz: float, order: int) -> list:
    """
    High-pass Butterworth order 1, 2, atau 4.

    order 1 : single-pole RC (~6 dB/oktaf)  — paling lembut
    order 2 : biquad Butterworth (~12 dB/oktaf) — default Audacity
    order 4 : dua biquad cascade (~24 dB/oktaf) — paling steep
    """
    nyquist   = framerate / 2.0
    cutoff_hz = max(1.0, min(cutoff_hz, nyquist * 0.999))

    if order == 1:
        # Single-pole high-pass (1st order RC analog equivalent)
        w0    = 2.0 * math.pi * cutoff_hz / framerate
        alpha = 1.0 / (1.0 + math.tan(w0 / 2.0))
        out    = [0.0] * len(channel)
        x_prev = 0.0
        y_prev = 0.0
        for i, x in enumerate(channel):
            y       = alpha * (y_prev + x - x_prev)
            out[i]  = y
            x_prev  = x
            y_prev  = y
        return out

    coeffs = _hp_coeffs(cutoff_hz, framerate)

    if order == 2:
        return _apply_biquad(channel, *coeffs)

    # order 4 — cascade dua biquad identik
    stage1 = _apply_biquad(channel, *coeffs)
    stage2 = _apply_biquad(stage1,  *coeffs)
    return stage2


# ─────────────────────────────────────────────────────────────────────────────
# Batch
# ─────────────────────────────────────────────────────────────────────────────

def print_preset_table():
    print()
    print("┌────┬──────────────────────────┬────────────┬───────┐")
    print("│ No │ Preset Name              │  Cutoff Hz │ Order │")
    print("├────┼──────────────────────────┼────────────┼───────┤")
    for i, (slug, label, cutoff, order) in enumerate(PRESETS, 1):
        slope = {1: "~6 dB/oct", 2: "~12 dB/oct", 4: "~24 dB/oct"}[order]
        print(f"│ {i:2d} │ {label:<24} │  {cutoff:>6} Hz  │ {order} ({slope}) │")
    print("└────┴──────────────────────────┴────────────┴───────┘")
    print()


def batch_process(input_path: str, outdir: str):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Input     : {input_path}")
    print(f"  Output dir: {outdir}")
    print(f"{'='*64}\n")

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
    success     = 0

    for idx, (slug, label, cutoff, order) in enumerate(PRESETS, 1):
        out_filename = f"{base_name}_{slug}.wav"
        out_path     = os.path.join(outdir, out_filename)

        bar = f"[{idx:2d}/18]"
        print(f"{bar} {label:<24}", end="  ", flush=True)

        t1 = time.time()
        processed = []
        for ch_data in channels:
            processed.append(
                apply_highpass(ch_data, params.framerate, cutoff, order)
            )
        write_wav(out_path, params, processed, sampwidth)
        elapsed = time.time() - t1

        slope = {1: "6", 2: "12", 4: "24"}[order]
        print(f"cutoff={cutoff} Hz  order={order} ({slope} dB/oct)  →  {out_filename}  ({elapsed:.1f}s)")
        success += 1

    total_time = time.time() - total_start
    print()
    print(f"{'='*64}")
    print(f"  Selesai! {success}/18 file berhasil dibuat.")
    print(f"  Total waktu proses : {total_time:.1f} s")
    print(f"  Folder output      : {os.path.abspath(outdir)}")
    print(f"{'='*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-apply 18 high-pass filter presets ke satu WAV file.",
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
        print("Usage: python highpass_presets.py input.wav [--outdir FOLDER] [--list]")
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