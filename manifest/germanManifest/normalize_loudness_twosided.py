#!/usr/bin/env python3
"""
Two-sided loudness normalization for the wav files listed in a manifest column.

Unlike a "boost-only" pass, this brings EVERY clip toward a single target
loudness (--target-lufs) while enforcing a true-peak ceiling (--target-tp):

    gain_db = min(target_lufs - measured_lufs,   # toward target loudness
                  target_tp   - measured_truepeak)  # never exceed TP ceiling

Crucially the gain may be NEGATIVE: clips that are too hot / clipped get pulled
DOWN below the true-peak ceiling (giving headroom and shrinking the huge CQT
activations that NaN the discriminator), and clips that are too quiet get pushed
UP. A single linear gain is applied (ffmpeg `volume`), so SNR is preserved — no
compression or limiting.

Why: the CQT/mel discriminator works in log/energy space, where amplitude
extremes (near-silence -> log(0)/divide-by-0, or clipped peaks -> overflow)
become inf/0 -> NaN in bf16. Normalizing both tails to a consistent target with
true-peak headroom removes both failure modes.

NOTE: attenuating an already-clipped clip lowers its level but cannot restore
the squared-off samples; the level reduction is what prevents the NaN. If the
pre-conversion float32 originals exist, re-converting with headroom is cleaner.

In place + atomic: each wav is written to a temp file then os.replace()'d over
the original. A failed ffmpeg leaves the ORIGINAL untouched.

Idempotent: after a pass, a clip already at the target measures gain ~0 and is
skipped, so re-running (e.g. after a timeout) only touches the remainder.

CPU ONLY (ffmpeg audio filters have no GPU path) — but it's fast (~minutes) and
can be run on a GPU node's CPU cores to skip the CPU queue.

Usage:
    python normalize_loudness_twosided.py \
        --manifest /data/.../LRS3-english/manifestLRS3-cluster-full.tsv \
        --col 1 --has-header \
        --target-lufs -23 --target-tp -2 --workers 32
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_TARGET_LUFS = -23.0
_TARGET_TP = -2.0
_SKIP_GAIN = 0.3   # |gain_db| below this -> already on target, skip (idempotent)


def _init(target_lufs, target_tp, skip_gain):
    global _TARGET_LUFS, _TARGET_TP, _SKIP_GAIN
    _TARGET_LUFS, _TARGET_TP, _SKIP_GAIN = target_lufs, target_tp, skip_gain


_LUFS_RE = re.compile(r"Input Integrated:\s*(-?[\d.]+)\s*LUFS")
_TP_RE = re.compile(r"Input True Peak:\s*(-?[\d.]+)\s*dBTP")


def _measure(path):
    """(lufs, truepeak_dBTP) or (None, None). ffmpeg loudnorm analysis only."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
         "-af", "loudnorm=print_format=summary", "-f", "null", "/dev/null"],
        capture_output=True, text=True)
    blob = r.stderr + r.stdout
    ml = _LUFS_RE.search(blob)
    mt = _TP_RE.search(blob)
    lufs = float(ml.group(1)) if ml else None
    tp = float(mt.group(1)) if mt else None
    return lufs, tp


def _normalize(path):
    """status: ok_boost|ok_atten|skip|measure_fail|ffmpeg_fail|missing."""
    if not os.path.isfile(path):
        return ("missing", path, 0.0)
    lufs, tp = _measure(path)
    if lufs is None or tp is None:
        return ("measure_fail", path, 0.0)
    # -inf / unreadable loudness (true silence) -> can't normalize meaningfully.
    if lufs <= -70:
        return ("measure_fail", path, 0.0)

    gain = min(_TARGET_LUFS - lufs, _TARGET_TP - tp)
    if abs(gain) < _SKIP_GAIN:
        return ("skip", path, gain)

    tmp = f"{path}.norm.{os.getpid()}.tmp.wav"
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-af", f"volume={gain:.4f}dB", "-c:a", "pcm_s16le", tmp],
        capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        if os.path.isfile(tmp):
            os.remove(tmp)
        return ("ffmpeg_fail", path, gain)
    os.replace(tmp, path)
    return (("ok_boost" if gain > 0 else "ok_atten"), path, gain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--col", type=int, default=1, help="0-based wav column")
    ap.add_argument("--has-header", action="store_true")
    ap.add_argument("--target-lufs", type=float, default=-23.0)
    ap.add_argument("--target-tp", type=float, default=-2.0)
    ap.add_argument("--skip-gain", type=float, default=0.3,
                    help="Skip clips already within this |dB| of target.")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths, seen = [], set()
    with open(args.manifest) as f:
        if args.has_header:
            f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            c = line.split("\t")
            if len(c) <= args.col:
                continue
            p = c[args.col]
            if p not in seen:
                seen.add(p)
                paths.append(p)

    n = len(paths)
    print(f"[loudnorm] {n} unique wavs | target {args.target_lufs} LUFS / "
          f"{args.target_tp} dBTP | workers={args.workers}", flush=True)
    if args.dry_run:
        for p in paths[:5]:
            print("  ", p)
        print("  ... (dry run)")
        return

    counts = {"ok_boost": 0, "ok_atten": 0, "skip": 0,
              "measure_fail": 0, "ffmpeg_fail": 0, "missing": 0}
    problems = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init,
                             initargs=(args.target_lufs, args.target_tp,
                                       args.skip_gain)) as ex:
        for i, (st, p, g) in enumerate(ex.map(_normalize, paths, chunksize=8), 1):
            counts[st] += 1
            if st in ("measure_fail", "ffmpeg_fail", "missing"):
                problems.append((st, p))
            if i % args.progress_every == 0 or i == n:
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (n - i) / rate if rate else 0
                print(f"  {i}/{n} | boost={counts['ok_boost']} "
                      f"atten={counts['ok_atten']} skip={counts['skip']} "
                      f"fail={counts['measure_fail']+counts['ffmpeg_fail']} | "
                      f"{rate:.1f} wav/s ETA {eta/60:.1f} min", flush=True)

    print("=" * 64)
    print(f"done in {(time.time()-t0)/60:.1f} min | {counts}")
    if problems:
        log = args.manifest + ".loudnorm_problems.txt"
        with open(log, "w") as fh:
            for st, p in problems:
                fh.write(f"{st}\t{p}\n")
        print(f"{len(problems)} problem files (originals untouched) -> {log}")
    sys.exit(1 if (counts["ffmpeg_fail"] or counts["measure_fail"]) else 0)


if __name__ == "__main__":
    main()
