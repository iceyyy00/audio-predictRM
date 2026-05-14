"""
bandpass_presets.py — Batch-apply 18 band-pass filter presets to a WAV file.

Sekali input, langsung output 18 file WAV dengan preset band-pass berbeda-beda.
Menggunakan algoritma Butterworth IIR biquad (cascade LP + HP) — seperti Audacity.

Usage:
    python bandpass_presets.py input.wav [--outdir OUTPUT_FOLDER] [--list]

Options:
    --outdir   Folder output (default: folder yang sama dengan input file)
    --list     Tampilkan daftar 18 preset tanpa memproses

Example:
    python bandpass_presets.py vocals.wav
    python bandpass_presets.py drums.wav --outdir hasil_bandpass
"""

import wave
import struct
import argparse
import os
import sys
import math
import time

# ─────────────────────────────────────────────────────────────────────────────
# 18 PRESET BAND-PASS
# (slug, label, low_hz, high_hz, order)
#
# low_hz  : frekuensi batas bawah (high-pass cutoff)
# high_hz : frekuensi batas atas  (low-pass  cutoff)
# order   : 1=lembut, 2=standard Butterworth, 4=steep
#
# Karakter preset masing-masing dijelaskan di komentar
# ─────────────────────────────────────────────────────────────────────────────
PRESETS = [
    # ── Telephone / Lo-Fi ──────────────────────────────────────────────────
    ("01_telephone",         "Telephone",          300,    3400,   2),
    # Range PSTN klasik — karakter telpon/radio walkie-talkie

    ("02_am_radio",          "AM Radio",           200,    5000,   2),
    # Lebih lebar dari telpon, tapi tetap lo-fi

    ("03_walkie_talkie",     "Walkie-Talkie",      500,    2500,   4),
    # Sempit & tajam, karakter komunikasi radio lapangan

    # ── Vokal / Presence ───────────────────────────────────────────────────
    ("04_vocal_body",        "Vocal Body",         200,    2000,   2),
    # Fokus ke tubuh/isi vokal, potong udara & sub

    ("05_vocal_presence",    "Vocal Presence",     800,    5000,   2),
    # Fokus ke presence/kejelasan vokal

    ("06_vocal_fullrange",   "Vocal Full Range",   80,     12000,  1),
    # Band-pass lebar, cuma potong sub & udara ekstrem

    # ── Instrumen Spesifik ─────────────────────────────────────────────────
    ("07_snare_body",        "Snare Body",         150,    8000,   2),
    # Tubuh snare drum — potong sub boom & cymbal bleed

    ("08_kick_punch",        "Kick Punch",         60,     200,    2),
    # Fokus ke punch/attack kick drum

    ("09_bass_guitar",       "Bass Guitar",        40,     600,    2),
    # Range fundamental bass gitar/bass synth

    ("10_guitar_mid",        "Guitar Mid",         250,    5000,   2),
    # Mid-range gitar elektrik — karakter Marshall

    ("11_piano_body",        "Piano Body",         80,     8000,   1),
    # Range lebar piano akustik

    # ── Efek / Karakter Kreatif ────────────────────────────────────────────
    ("12_presence_boost",    "Presence Boost",     2000,   8000,   2),
    # Hanya presence & treble — untuk efek "hadir" di mix

    ("13_sub_bass",          "Sub Bass",           20,     120,    4),
    # Hanya sub-bass — untuk efek rumble/cinematic

    ("14_mid_scoop",         "Mid Scoop",          500,    4000,   4),
    # Band sempit mid — untuk efek "boxy" atau nasal

    ("15_high_shelf_band",   "High Shelf Band",    5000,   16000,  2),
    # Hanya treble & udara — brightness/air

    # ── Vintage / Analog ───────────────────────────────────────────────────
    ("16_vinyl_cut",         "Vinyl Cut",          30,     16000,  1),
    # Simulasi cutting head vinyl (potong sub & ultra-high)

    ("17_tape_warmth",       "Tape Warmth",        60,     10000,  2),
    # Simulasi karakter head tape — potong sub & udara

    ("18_broadcast_std",     "Broadcast Standard", 100,    15000,  2),
    # Standard broadcast audio — seperti FM radio hi-fi
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
# DSP — Butterworth Biquad (Low-Pass & High-Pass)
# Band-pass = cascade HP biquad lalu LP biquad
# ─────────────────────────────────────────────────────────────────────────────

def _lp_coeffs(cutoff_hz: float, framerate: int) -> tuple:
    """Koefisien biquad Butterworth low-pass 2nd-order."""
    w0     = 2.0 * math.pi * cutoff_hz / framerate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha  = sin_w0 / math.sqrt(2.0)   # Q = 1/sqrt(2)

    b0 = (1.0 - cos_w0) / 2.0
    b1 =  1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 =  1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 =  1.0 - alpha

    return (b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)


def _hp_coeffs(cutoff_hz: float, framerate: int) -> tuple:
    """Koefisien biquad Butterworth high-pass 2nd-order."""
    w0     = 2.0 * math.pi * cutoff_hz / framerate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha  = sin_w0 / math.sqrt(2.0)

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
    w1 = 0.0
    w2 = 0.0
    for i, x in enumerate(channel):
        y      = b0 * x + w1
        w1     = b1 * x - a1 * y + w2
        w2     = b2 * x - a2 * y
        out[i] = y
    return out


def _apply_lp(channel: list, cutoff_hz: float, framerate: int, order: int) -> list:
    nyquist   = framerate / 2.0
    cutoff_hz = min(cutoff_hz, nyquist * 0.999)
    coeffs    = _lp_coeffs(cutoff_hz, framerate)
    result    = _apply_biquad(channel, *coeffs)
    if order == 4:
        result = _apply_biquad(result, *coeffs)   # cascade kedua
    return result


def _apply_hp(channel: list, cutoff_hz: float, framerate: int, order: int) -> list:
    nyquist   = framerate / 2.0
    cutoff_hz = min(cutoff_hz, nyquist * 0.999)
    if order == 1:
        # Single-pole HP (1st order)
        w0    = 2.0 * math.pi * cutoff_hz / framerate
        alpha = 1.0 / (1.0 + math.tan(w0 / 2.0))
        out   = [0.0] * len(channel)
        x_prev = 0.0
        y_prev = 0.0
        for i, x in enumerate(channel):
            y       = alpha * (y_prev + x - x_prev)
            out[i]  = y
            x_prev  = x
            y_prev  = y
        return out
    coeffs = _hp_coeffs(cutoff_hz, framerate)
    result = _apply_biquad(channel, *coeffs)
    if order == 4:
        result = _apply_biquad(result, *coeffs)
    return result


def apply_bandpass(channel: list, framerate: int,
                   low_hz: float, high_hz: float, order: int) -> list:
    """
    Band-pass = High-Pass(low_hz) → Low-Pass(high_hz), cascade.
    Hanya frekuensi antara low_hz dan high_hz yang lolos.
    """
    if low_hz >= high_hz:
        sys.exit(f"[ERROR] low_hz ({low_hz}) harus lebih kecil dari high_hz ({high_hz})")

    # Clamp ke Nyquist
    nyquist  = framerate / 2.0
    low_hz   = max(1.0, low_hz)
    high_hz  = min(high_hz, nyquist * 0.999)

    # 1. High-pass untuk potong frekuensi BAWAH low_hz
    after_hp = _apply_hp(channel, low_hz,  framerate, order)
    # 2. Low-pass untuk potong frekuensi ATAS high_hz
    after_lp = _apply_lp(after_hp, high_hz, framerate, order)

    return after_lp


# ─────────────────────────────────────────────────────────────────────────────
# Batch
# ─────────────────────────────────────────────────────────────────────────────

def print_preset_table():
    print()
    print("┌────┬──────────────────────────┬──────────┬──────────┬───────┐")
    print("│ No │ Preset Name              │  Low Hz  │  High Hz │ Order │")
    print("├────┼──────────────────────────┼──────────┼──────────┼───────┤")
    for i, (slug, label, lo, hi, order) in enumerate(PRESETS, 1):
        print(f"│ {i:2d} │ {label:<24} │ {lo:>6} Hz │ {hi:>6} Hz │   {order}   │")
    print("└────┴──────────────────────────┴──────────┴──────────┴───────┘")
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

    for idx, (slug, label, lo, hi, order) in enumerate(PRESETS, 1):
        out_filename = f"{base_name}_{slug}.wav"
        out_path     = os.path.join(outdir, out_filename)

        bar = f"[{idx:2d}/18]"
        print(f"{bar} {label:<24}", end="  ", flush=True)

        t1 = time.time()
        processed = []
        for ch_data in channels:
            processed.append(
                apply_bandpass(ch_data, params.framerate, lo, hi, order)
            )
        write_wav(out_path, params, processed, sampwidth)
        elapsed = time.time() - t1

        bw = hi - lo
        print(f"[{lo}–{hi} Hz | BW={bw} Hz | order={order}]  →  {out_filename}  ({elapsed:.1f}s)")
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
        description="Batch-apply 18 band-pass filter presets ke satu WAV file.",
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
        print("Usage: python bandpass_presets.py input.wav [--outdir FOLDER] [--list]")
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