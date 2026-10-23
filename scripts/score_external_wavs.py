#!/usr/bin/env python3
"""Score wavs produced by an EXTERNAL voice-conversion model with this repo's stages 3-5.

Why this script exists
----------------------
Stages 3-5 (score_mos.py / score_secs.py / score_asr.py) already work on any folder of
wavs -- they are model-agnostic. What they are NOT agnostic about is the FILENAME: every
scorer identifies an utterance by the prediction being named '<uid>.wav', where

    uid = <speaker>-<COND>-<utterance>        e.g. 0010-BM-common-dialogs_0001

and rebuilds uid -> (source, target) from the manifest. An external model names its
outputs however it likes, so its folder cannot be fed to the scorers directly.

This script is the adapter: it builds a symlink tree whose names ARE the uids, then calls
the existing scorers on that tree. The scoring code stays single-sourced (no forked copy
that can drift), and run_eval_pipeline.py is not touched at all.

Mapping file
------------
--map is a file describing which external wav corresponds to which manifest row. Two
formats, auto-detected:

  1. 'title|src|tgt'  (QuickVC's convert_eval.txt)
     The uid is rebuilt from `src` using the same _uid() the scorers use, so the pairing
     is taken from the converter's own record rather than guessed from filenames.
     The external file is '<title>.wav' inside --wav-dir.

  2. '<external_name>\t<uid>'  (plain TSV, for any other model)

Either way the mapping is EXPLICIT. Nothing is inferred from filename patterns -- a wrong
target silently corrupts SECS (and only SECS: WER/CER/MOS would look normal), so guessing
is not acceptable here.

Example
-------
  python scripts/score_external_wavs.py \
      --wav-dir  /data/.../QuickVC-VoiceConversion/dumpedquickvc \
      --map      /data/.../QuickVC-VoiceConversion/convert_eval.txt \
      --out-root /data/.../DumpedMels_quickvc_test \
      --tag      quickvc
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("score-external")

SCRIPTS = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MANIFEST = ("/data/fs201163/eo49197/VoiceConversion-fwf/"
                    "dub-autoalign-healthyYT-trim/testPATH-HE.tsv")
DEFAULT_TRANSCRIPTS = "/data/fs201163/eo49197/VoiceConversion-fwf/transcripts.csv"
DEFAULT_ASR_MODEL = "openai/whisper-large-v3"


def _uid(input_wav):
    """MUST match dump_mels_eval._uid / score_secs._uid: <speaker>-<COND>-<utterance>.

    .../0019-processed-MP/IS0019shortstories_0008/pathological.wav
      -> '0019-MP-shortstories_0008'
    """
    d = os.path.dirname(input_wav)
    parts = d.rstrip("/").split("/")
    if len(parts) < 2:
        return parts[-1]
    folder1, folder2 = parts[-2], parts[-1]
    m = re.match(r"^(\d+)-processed-(.+)$", folder1)
    if not m:
        return folder2
    spk, cond = m.group(1), m.group(2)
    utt = re.sub(r"^[A-Za-z]*" + re.escape(spk), "", folder2)
    return f"{spk}-{cond}-{utt}"


def load_manifest_uids(manifest):
    """uid -> (source, target) for every manifest row, same filter the scorers use."""
    m = {}
    with open(manifest) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3 or "pathological.wav" not in cols[1]:
                continue          # skips the '/' root line and malformed rows
            m[_uid(cols[1])] = (cols[1], cols[2])
    return m


def parse_map(map_path, wav_dir):
    """-> list of (external_wav_basename, uid). Auto-detects the two formats."""
    pairs = []
    with open(map_path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            if "|" in line:                                  # title|src|tgt
                parts = line.split("|")
                if len(parts) < 2:
                    raise SystemExit(f"{map_path}:{lineno}: expected 'title|src|tgt'")
                title, src = parts[0], parts[1]
                pairs.append((f"{title}.wav", _uid(src)))
            else:                                            # external_name<TAB>uid
                parts = line.split("\t")
                if len(parts) < 2:
                    raise SystemExit(
                        f"{map_path}:{lineno}: expected '<external_name>\\t<uid>' "
                        f"or 'title|src|tgt'")
                name = parts[0] if parts[0].endswith(".wav") else parts[0] + ".wav"
                pairs.append((name, parts[1]))
    return pairs


def build_link_tree(pairs, wav_dir, link_dir, manifest_uids, overwrite):
    """Symlink <link_dir>/<uid>.wav -> <wav_dir>/<external>. Returns (n_linked, problems)."""
    os.makedirs(link_dir, exist_ok=True)
    problems = {"missing_wav": [], "unknown_uid": [], "dup_uid": []}
    seen, linked = set(), 0

    for external, uid in pairs:
        src = os.path.join(wav_dir, external)
        if not os.path.isfile(src):
            problems["missing_wav"].append(external)
            continue
        if uid not in manifest_uids:
            problems["unknown_uid"].append(uid)
            continue
        if uid in seen:
            problems["dup_uid"].append(uid)
            continue
        seen.add(uid)

        dst = os.path.join(link_dir, uid + ".wav")
        if os.path.islink(dst) or os.path.exists(dst):
            if not overwrite:
                linked += 1
                continue
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        linked += 1

    problems["uncovered_manifest"] = sorted(set(manifest_uids) - seen)
    return linked, problems


def run(cmd):
    logger.info("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True, help="Folder of the external model's wavs.")
    ap.add_argument("--map", required=True,
                    help="'title|src|tgt' file (e.g. convert_eval.txt) or "
                         "'<external_name>\\t<uid>' TSV.")
    ap.add_argument("--out-root", required=True, help="Where results are written.")
    ap.add_argument("--tag", default="external",
                    help="Name of the results subfolder under --out-root.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    ap.add_argument("--language", default="de", help="Whisper decode language.")
    ap.add_argument("--stage", default="all",
                    help="'all' or a comma list from 3,4,5 (3=MOS, 4=SECS, 5=ASR).")
    ap.add_argument("--score-batch-size", type=int, default=48)
    ap.add_argument("--baselines", action="store_true",
                    help="Also score the manifest floor/ceiling (MOS + ASR).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    stages = ([3, 4, 5] if args.stage == "all"
              else [int(s) for s in args.stage.split(",") if s.strip()])
    ow = ["--overwrite"] if args.overwrite else []
    asr_tag = args.asr_model.split("/")[-1]

    wav_dir = args.wav_dir.rstrip("/")
    link_dir = os.path.join(args.out_root, args.tag)

    for p in (wav_dir, args.map, args.manifest, args.transcripts):
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")

    # ---------------------------------------------------------------- link tree
    manifest_uids = load_manifest_uids(args.manifest)
    pairs = parse_map(args.map, wav_dir)
    logger.info(f"manifest rows {len(manifest_uids)} | map entries {len(pairs)}")

    linked, problems = build_link_tree(pairs, wav_dir, link_dir, manifest_uids,
                                       args.overwrite)
    logger.info(f"linked {linked} wavs -> {link_dir}")
    for key in ("missing_wav", "unknown_uid", "dup_uid", "uncovered_manifest"):
        bad = problems[key]
        if bad:
            logger.warning(f"  {key}: {len(bad)}  e.g. {bad[:3]}")
    if linked == 0:
        raise SystemExit("nothing linked -- check --wav-dir and --map")

    # ---------------------------------------------------------------- stages
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("=" * 72)
    logger.info(f"[external] tag={args.tag} stages={stages} out_root={args.out_root} "
                f"asr={args.asr_model} lang={args.language}  ({ts})")
    logger.info("=" * 72)

    if 3 in stages:
        run([sys.executable, os.path.join(SCRIPTS, "score_mos.py"), link_dir] + ow)
        if args.baselines:
            base = os.path.join(args.out_root, "mos_baseline")
            for role in ("source", "healthy"):
                run([sys.executable, os.path.join(SCRIPTS, "score_mos.py"),
                     "--role", role, "--manifest", args.manifest,
                     "--out-dir", base] + ow)

    if 4 in stages:
        run([sys.executable, os.path.join(SCRIPTS, "score_secs.py"), link_dir,
             "--manifest", args.manifest,
             "--batch-size", str(args.score_batch_size)] + ow)

    if 5 in stages:
        asr = os.path.join(SCRIPTS, "score_asr.py")
        common = ["--manifest", args.manifest, "--transcripts", args.transcripts,
                  "--model", args.asr_model, "--language", args.language,
                  "--batch-size", str(args.score_batch_size)]
        run([sys.executable, asr, link_dir, "--role", "pred",
             "--out-name", f"asr_{asr_tag}.txt"] + common + ow)
        if args.baselines:
            base = os.path.join(args.out_root, f"asr_baseline_{asr_tag}")
            for role in ("healthy", "source"):
                run([sys.executable, asr, "--role", role, "--out-dir", base] + common + ow)

    logger.info("=" * 72)
    logger.info(f"[external] ALL DONE -> {link_dir}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
