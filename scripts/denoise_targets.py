#!/usr/bin/env python3
"""Denoise the TARGET wavs of a manifest, writing '<stem>_denoise.wav' NEXT TO each original.

Nothing is overwritten and nothing is deleted -- the originals stay exactly as they are, so
you can listen to both and decide afterwards whether to keep or discard the denoised copies.

Based on scripts/clean_speech.py (same audio-separator model), with three differences:
  * reads the target column of a manifest instead of a flat path list
  * writes in place as '<stem>_denoise.wav' rather than into a separate output tree
  * converts the separator's stem back to the ORIGINAL file's sample rate / channels /
    subtype, so a later overwrite is format-identical and the scorers read them the same

The separator emits its stems into a scratch directory; only the clean ('vocals') side is
kept and converted, the rest is removed.
"""

import argparse
import glob
import logging
import os
import shutil
import subprocess
import tempfile

# audio-separator shells out to ffmpeg. On the cluster ffmpeg lives in the conda env's
# bin, which is NOT on PATH inside a SLURM job -- without this every separation fails.
_ENV_BIN = os.path.join(os.path.dirname(os.path.realpath(__import__("sys").executable)))
os.environ["PATH"] = _ENV_BIN + os.pathsep + os.environ.get("PATH", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("denoise")

DEFAULT_MANIFEST = ("/data/fs201163/eo49197/VoiceConversion-fwf/"
                    "dub-autoalign-healthyYT-trim/testPATH-HE.tsv")
MODEL = "vocals_mel_band_roformer.ckpt"

# WHICH STEM IS THE SPEECH?
# vocals_mel_band_roformer writes TWO files per input:
#     <stem>_(vocals)_vocals_mel_band_roformer.wav   <- the SPEECH  (keep this)
#     <stem>_(other)_vocals_mel_band_roformer.wav    <- the noise   (discard)
# The MODEL NAME also contains "vocals", so a plain substring search for "vocals" matches
# BOTH filenames. Match on the parenthesised stem LABEL only.
CLEAN_LABEL = "(vocals)"
NOISE_LABEL = "(other)"


def read_targets(manifest, col):
    """Distinct existing paths from `col` (0-based) of the manifest, in first-seen order."""
    seen, out, missing = set(), [], 0
    with open(manifest) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= col or "pathological.wav" not in cols[1]:
                continue                      # skips the '/' root line and malformed rows
            p = cols[col]
            if p in seen:
                continue
            seen.add(p)
            if os.path.isfile(p):
                out.append(p)
            else:
                missing += 1
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--col", type=int, default=2,
                    help="0-based target column (2 = file_path).")
    ap.add_argument("--suffix", default="_denoise")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N files (0 = all).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Redo files whose _denoise output already exists.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets, missing = read_targets(args.manifest, args.col)
    if args.limit:
        targets = targets[:args.limit]
    logger.info(f"manifest {args.manifest}")
    logger.info(f"targets: {len(targets)} distinct existing files "
                f"({missing} listed but not on disk)")

    todo = [p for p in targets
            if args.overwrite or not os.path.isfile(
                os.path.join(os.path.dirname(p),
                             os.path.splitext(os.path.basename(p))[0] + args.suffix + ".wav"))]
    logger.info(f"to process: {len(todo)}  (already done: {len(targets) - len(todo)})")

    if args.dry_run:
        for p in todo[:5]:
            stem = os.path.splitext(os.path.basename(p))[0]
            logger.info(f"  {p}\n    -> {os.path.join(os.path.dirname(p), stem + args.suffix + '.wav')}")
        logger.info("  ... (dry run, nothing written)")
        return
    if not todo:
        logger.info("nothing to do")
        return

    import soundfile as sf
    import numpy as np

    # Fail loudly and immediately if ffmpeg is unreachable: without it EVERY separation
    # fails, and a per-file try/except would quietly log 1600 failures instead of one
    # actionable error. (That is exactly what went wrong on the first run.)
    try:
        subprocess.check_output(["ffmpeg", "-version"], text=True,
                                stderr=subprocess.STDOUT)
    except Exception as e:
        raise SystemExit(f"ffmpeg not runnable ({e}). PATH={os.environ['PATH'][:200]}")
    logger.info(f"ffmpeg OK (PATH head: {_ENV_BIN})")

    from audio_separator.separator import Separator

    scratch = tempfile.mkdtemp(prefix="denoise_")
    logger.info(f"scratch dir: {scratch}")
    separator = Separator(output_dir=scratch)
    separator.load_model(model_filename=args.model)

    done = failed = 0
    for i, src in enumerate(todo, 1):
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(os.path.dirname(src), stem + args.suffix + ".wav")
        try:
            info = sf.info(src)                       # original format, to match exactly
            outs = separator.separate(src)
            outs = [o if os.path.isabs(o) else os.path.join(scratch, o) for o in outs]
            outs = [o for o in outs if os.path.isfile(o)]
            if not outs:                              # fall back to whatever landed on disk
                outs = sorted(glob.glob(os.path.join(scratch, "*.wav")))
            if not outs:
                raise RuntimeError(f"separator produced no files for {src}")

            # Pick by the parenthesised stem LABEL, never a bare 'vocals' substring: the
            # model name contains 'vocals' too, so a substring match hits BOTH stems and
            # can silently select the noise side (that produced ~1.6%-RMS silent output).
            clean = next((o for o in outs
                          if CLEAN_LABEL in os.path.basename(o).lower()), None)
            if clean is None:
                raise RuntimeError(
                    f"no '{CLEAN_LABEL}' stem among {[os.path.basename(o) for o in outs]}")

            data, sr = sf.read(clean, dtype="float32", always_2d=True)
            # Guard against the failure we already hit once: a near-silent 'clean' stem
            # means the wrong side was kept. Better to abort than to write 1600 dud files.
            if float(np.abs(data).max()) < 0.005:
                raise RuntimeError(
                    f"selected stem {os.path.basename(clean)} is near-silent "
                    f"(peak {float(np.abs(data).max()):.5f}) -- wrong stem?")
            if data.shape[1] != info.channels:        # stereo stem -> mono original
                data = data.mean(axis=1, keepdims=True) if info.channels == 1 else data
            if sr != info.samplerate:
                import librosa
                data = librosa.resample(data.T, orig_sr=sr, target_sr=info.samplerate).T
            sf.write(dst, np.squeeze(data, axis=1) if info.channels == 1 else data,
                     info.samplerate, subtype=info.subtype)
            done += 1

            for o in outs:                            # drop every stem, clean included
                if os.path.isfile(o):
                    os.remove(o)
            if i % 50 == 0 or i == len(todo):
                logger.info(f"  {i}/{len(todo)} done={done} failed={failed}")
        except Exception as e:
            failed += 1
            logger.warning(f"  [{i}] FAILED {src}: {e}")
            # A genuinely odd input can fail; a broken ENVIRONMENT fails everything. If the
            # first handful all fail, stop rather than logging 1600 warnings and "finishing".
            if failed >= 5 and done == 0:
                raise SystemExit(
                    f"aborting: first {failed} files all failed -- this is a setup problem, "
                    f"not bad audio. Last error: {e}")

    shutil.rmtree(scratch, ignore_errors=True)
    logger.info(f"finished: wrote {done}, failed {failed}")
    if done:
        ex = todo[0]
        logger.info("listen to a pair, e.g.:")
        logger.info(f"  original: {ex}")
        logger.info(f"  denoised: {os.path.join(os.path.dirname(ex), os.path.splitext(os.path.basename(ex))[0] + args.suffix + '.wav')}")


if __name__ == "__main__":
    main()
