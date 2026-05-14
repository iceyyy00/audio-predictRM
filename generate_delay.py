"""
delay_presets.py — Batch-apply 18 Audacity-style delay presets to a WAV file.

Sekali input, langsung output 18 file WAV dengan preset berbeda-beda.

Usage:
    python delay_presets.py input.wav [--outdir OUTPUT_FOLDER] [--list]

Options:
    --outdir   Folder output (default: folder yang sama dengan input file)
    --list     Tampilkan daftar 18 preset tanpa memproses

Example:
    python delay_presets.py vocals.wav
    python delay_presets.py vocals.wav --outdir hasil_delay
"""

import wave
import struct
import argparse
import os
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# 18 PRESET — mirip preset bawaan Audacity delay/echo effect
# Setiap preset: (nama_file_suffix, label, delay_ms, feedback, mix, repeats)
# ─────────────────────────────────────────────────────────────────────────────
PRESETS = [
    # ── Slapback (echo pendek 1x, rock/country/rockabilly) ──
    ("01_slapback_short",       "Slapback Short",        80,   0.0,  0.35,  1),
    ("02_slapback_medium",      "Slapback Medium",       120,  0.0,  0.40,  1),
    ("03_slapback_long",        "Slapback Long",         180,  0.0,  0.45,  1),

    # ── Room / Small Space ──
    ("04_room_small",           "Room Small",            60,   0.25, 0.30,  4),
    ("05_room_medium",          "Room Medium",           120,  0.30, 0.35,  5),
    ("06_room_large",           "Room Large",            200,  0.35, 0.40,  6),

    # ── Studio / Tracking ──
    ("07_studio_tight",         "Studio Tight",          30,   0.15, 0.25,  3),
    ("08_studio_warm",          "Studio Warm",           250,  0.40, 0.45,  5),

    # ── Hall / Ambience ──
    ("09_hall_small",           "Hall Small",            300,  0.45, 0.50,  6),
    ("10_hall_large",           "Hall Large",            500,  0.55, 0.55,  7),

    # ── Rhythmic / Tempo-synced (berasa berdenyut) ──
    ("11_eighth_note_120bpm",   "Eighth Note @120bpm",   250,  0.50, 0.50,  6),
    ("12_quarter_note_120bpm",  "Quarter Note @120bpm",  500,  0.50, 0.50,  5),
    ("13_dotted_eighth_120bpm", "Dotted Eighth @120bpm", 375,  0.45, 0.50,  5),

    # ── Long / Washy ──
    ("14_long_wash",            "Long Wash",             700,  0.60, 0.55,  8),
    ("15_infinite_drift",       "Infinite Drift",        900,  0.75, 0.60,  9),

    # ── Special / Karakter Unik ──
    ("16_ping_pong_sim",        "Ping-Pong Sim",         450,  0.55, 0.65,  6),
    ("17_canyon_echo",          "Canyon Echo",           1200, 0.65, 0.50,  5),
    ("18_tape_echo",            "Tape Echo",             340,  0.48, 0.52,  7),
]

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_wav(path: str):
    with wave.open(path, "rb") as wf:
        params = wf.getparams()
        raw    = wf.readframes(params.nframes)

    sw = params.sampwidth
    nc = params.nchannels
    nf = params.nframes
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
    nc     = len(channels)
    nf     = len(channels[0])
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
# Core DSP
# ─────────────────────────────────────────────────────────────────────────────

def apply_delay(channel: list,
                framerate: int,
                delay_ms: float,
                feedback: float,
                mix: float,
                repeats: int) -> list:
    """
    Multi-tap delay line.
    out[n] = (1-mix)*dry[n] + mix * Σ(k=1..repeats) feedback^(k-1) * dry[n - k*delay_samples]
    """
    delay_samples = int(framerate * delay_ms / 1000)
    n   = len(channel)
    out = [0.0] * n

    for i in range(n):
        dry = channel[i]
        wet = 0.0
        gain = 1.0
        for k in range(1, repeats + 1):
            tap = i - k * delay_samples
            if tap >= 0:
                wet += gain * channel[tap]
            gain *= feedback
        out[i] = (1.0 - mix) * dry + mix * wet

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Batch processor
# ─────────────────────────────────────────────────────────────────────────────

def print_preset_table():
    print()
    print("┌────┬─────────────────────────────┬──────────┬──────────┬──────┬─────────┐")
    print("│ No │ Preset Name                 │ Delay ms │ Feedback │  Mix │ Repeats │")
    print("├────┼─────────────────────────────┼──────────┼──────────┼──────┼─────────┤")
    for i, (slug, label, dms, fb, mx, rep) in enumerate(PRESETS, 1):
        print(f"│ {i:2d} │ {label:<27} │  {dms:>6} │   {fb:.2f}   │ {mx:.2f} │    {rep:2d}   │")
    print("└────┴─────────────────────────────┴──────────┴──────────┴──────┴─────────┘")
    print()


def batch_process(input_path: str, outdir: str):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Input     : {input_path}")
    print(f"  Output dir: {outdir}")
    print(f"{'='*60}\n")

    # Baca file sekali saja
    print("Membaca file input …")
    t0 = time.time()
    params, channels, sampwidth = read_wav(input_path)
    read_time = time.time() - t0

    duration_s = params.nframes / params.framerate
    print(f"  Channels  : {params.nchannels}")
    print(f"  Rate      : {params.framerate} Hz")
    print(f"  Bit depth : {sampwidth * 8}-bit")
    print(f"  Duration  : {duration_s:.2f} s  ({params.nframes} frames)")
    print(f"  Read time : {read_time:.2f} s")
    print()

    total_start = time.time()
    success = 0

    for idx, (slug, label, delay_ms, feedback, mix, repeats) in enumerate(PRESETS, 1):
        out_filename = f"{base_name}_{slug}.wav"
        out_path     = os.path.join(outdir, out_filename)

        bar = f"[{idx:2d}/18]"
        print(f"{bar} {label:<27}", end="  ", flush=True)

        t1 = time.time()
        processed = []
        for ch_data in channels:
            processed.append(
                apply_delay(ch_data, params.framerate,
                            delay_ms, feedback, mix, repeats)
            )
        write_wav(out_path, params, processed, sampwidth)
        elapsed = time.time() - t1

        print(f"delay={delay_ms}ms fb={feedback} mix={mix} rep={repeats}  →  {out_filename}  ({elapsed:.1f}s)")
        success += 1

    total_time = time.time() - total_start
    print()
    print(f"{'='*60}")
    print(f"  Selesai! {success}/18 file berhasil dibuat.")
    print(f"  Total waktu proses : {total_time:.1f} s")
    print(f"  Folder output      : {os.path.abspath(outdir)}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-apply 18 Audacity-style delay presets ke satu WAV file.",
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
        print("Usage: python delay_presets.py input.wav [--outdir FOLDER] [--list]")
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