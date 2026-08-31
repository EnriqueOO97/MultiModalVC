"""
Stage 5 (NeMo variant): ASR intelligibility (WER / CER) with an NVIDIA NeMo German model.

Same job and OUTPUT FORMAT as score_asr.py (uid pairing, German normalizer, jiwer WER/CER,
asr_<role>.txt), but the transcriber is a NeMo model (default nvidia/stt_de_conformer_ctc_large,
a CTC German conformer) instead of HuggingFace Whisper -- so its numbers are directly
comparable to the Whisper ceilings computed by score_asr.py.

Needs `nemo_toolkit[asr]` installed in the env (NOT a dependency of score_asr.py):
    pip install "nemo_toolkit[asr]"
Weights are fetched from HF on first run (needs internet), then cached.

  # ceiling over the healthy GT (writes asr_healthy.txt into OUT_DIR):
  python scripts/score_asr_nemo.py --role healthy --out-dir /some/dir \
      --manifest /.../testPATH-HE.tsv --transcripts /.../transcripts.csv

  # predictions (writes asr_scores.txt inside each wav folder):
  python scripts/score_asr_nemo.py <wav_dir> [<wav_dir> ...] --role pred \
      --manifest /.../testPATH-HE.tsv --transcripts /.../transcripts.csv
"""

import os
import re
import sys
import csv
import argparse
import logging

import torch

logger = logging.getLogger("asr-nemo")

DEFAULT_MODEL = "nvidia/stt_de_conformer_ctc_large"
DEFAULT_MANIFEST = ("/data/fs201163/eo49197/VoiceConversion-fwf/"
                    "dub-autoalign-healthyYT-trim/testPATH-HE.tsv")
DEFAULT_TRANSCRIPTS = "/data/fs201163/eo49197/VoiceConversion-fwf/transcripts.csv"


# ---- pairing / text helpers (kept identical to score_asr.py so results are comparable) ----
def _uid(input_wav):
    d = os.path.dirname(input_wav)
    parts = d.rstrip("/").split("/")
    if len(parts) < 2:
        return parts[-1]
    folder1, folder2 = parts[-2], parts[-1]
    m = re.match(r"^(\d+)-processed-(.+)$", folder1)
    if not m:
        return f"{folder1}__{folder2}"
    spk, cond = m.group(1), m.group(2)
    utt = re.sub(r"^[A-Za-z]*" + re.escape(spk), "", folder2)
    return f"{spk}-{cond}-{utt}"


def _utt_of(uid):
    parts = uid.split("-", 2)
    return parts[2] if len(parts) == 3 else uid


def build_uid_map(manifest):
    m = {}
    with open(manifest) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3 or "pathological.wav" not in cols[1]:
                continue
            m[_uid(cols[1])] = {"source": cols[1], "target": cols[2]}
    return m


def load_transcripts(path):
    txt = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            fn = row[0].strip()
            utt = fn[:-4] if fn.lower().endswith(".wav") else fn
            txt[utt] = row[1]
    return txt


_PUNCT = re.compile(r"[^\wäöüßÄÖÜ\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(s):
    s = s.lower().replace("_", " ")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    t = torch.tensor(xs)
    return t.mean().item(), (t.std(unbiased=False).item() if len(xs) > 1 else 0.0)


# ---- NeMo transcription ----
def nemo_transcribe(model, paths, batch_size):
    """Return list[str] hypotheses. Handles NeMo versions that return plain strings,
    Hypothesis objects, or a (best, all) tuple."""
    out = model.transcribe(paths, batch_size=batch_size)
    if isinstance(out, tuple):            # some versions: (best_hyps, all_hyps)
        out = out[0]
    hyps = []
    for h in out:
        hyps.append(h.text if hasattr(h, "text") else str(h))
    return hyps


def score_and_write(uids, paths, refs, model, args, out_path):
    import jiwer
    logger.info(f"=== {out_path}  ({len(uids)} wavs)")
    hyps = nemo_transcribe(model, paths, args.batch_size)
    ref_n = [normalize(r) for r in refs]
    hyp_n = [normalize(h) for h in hyps]

    wers, cers = [], []
    for r, h in zip(ref_n, hyp_n):
        r_ = r if r else " "
        wers.append(jiwer.wer(r_, h if h else " "))
        cers.append(jiwer.cer(r_, h if h else " "))

    by_spk, by_cond = {}, {}
    for uid, w in zip(uids, wers):
        by_spk.setdefault(uid.split("-")[0], []).append(w)
        by_cond.setdefault(uid.split("-")[1], []).append(w)

    mw, sw = _mean_std(wers)
    mc, sc = _mean_std(cers)
    corpus_w = jiwer.wer([r if r else " " for r in ref_n], [h if h else " " for h in hyp_n])
    corpus_c = jiwer.cer([r if r else " " for r in ref_n], [h if h else " " for h in hyp_n])

    with open(out_path, "w") as fh:
        fh.write("uid\tWER\tCER\n")
        for uid, w, c in zip(uids, wers, cers):
            fh.write(f"{uid}\t{w:.4f}\t{c:.4f}\n")
        fh.write(f"MEAN\t{mw:.4f}\t{mc:.4f}\n")
        fh.write(f"STD\t{sw:.4f}\t{sc:.4f}\n")
        fh.write(f"CORPUS\t{corpus_w:.4f}\t{corpus_c:.4f}\n")
        fh.write(f"N\t{len(uids)}\n")
        fh.write("# per-speaker mean WER\n")
        for k in sorted(by_spk):
            m_, _ = _mean_std(by_spk[k])
            fh.write(f"#SPK\t{k}\t{m_:.4f}\t{len(by_spk[k])}\n")
        fh.write("# per-condition mean WER\n")
        for k in sorted(by_cond):
            m_, _ = _mean_std(by_cond[k])
            fh.write(f"#COND\t{k}\t{m_:.4f}\t{len(by_cond[k])}\n")
    logger.info(f"    -> {out_path}   WER(mean)={mw:.3f}  CER(mean)={mc:.3f}  "
                f"WER(corpus)={corpus_w:.3f}  N={len(uids)}")


def collect_from_dir(wav_dir, transcripts):
    uids, paths, refs, missing = [], [], [], []
    for f in sorted(x for x in os.listdir(wav_dir) if x.endswith(".wav")):
        uid = f[:-4]
        ref = transcripts.get(_utt_of(uid))
        if ref is None:
            missing.append(uid)
            continue
        uids.append(uid)
        paths.append(os.path.join(wav_dir, f))
        refs.append(ref)
    return uids, paths, refs, missing


def collect_from_manifest(uid_map, transcripts, key):
    uids, paths, refs, missing = [], [], [], []
    for uid, rec in uid_map.items():
        ref = transcripts.get(_utt_of(uid))
        if ref is None:
            missing.append(uid)
            continue
        uids.append(uid)
        paths.append(rec[key])
        refs.append(ref)
    return uids, paths, refs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_dirs", nargs="*", help="Prediction folders (role=pred only).")
    ap.add_argument("--role", choices=["pred", "healthy", "source"], default="pred")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--out-name", default="asr_scores.txt", help="Output filename (role=pred).")
    ap.add_argument("--out-dir", default=None, help="Where to write asr_<role>.txt (baselines).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    import jiwer  # noqa: F401  (fail fast if missing)
    import nemo.collections.asr as nemo_asr
    logger.info(f"[nemo] loading {args.model} -> {device}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
    model = model.to(device).eval()

    uid_map = build_uid_map(args.manifest)
    transcripts = load_transcripts(args.transcripts)
    logger.info(f"[manifest] {len(uid_map)} uids   [transcripts] {len(transcripts)} utterances")

    if args.role == "pred":
        if not args.wav_dirs:
            ap.error("role=pred needs at least one wav_dir")
        for wav_dir in args.wav_dirs:
            wav_dir = wav_dir.rstrip("/")
            out_path = os.path.join(wav_dir, args.out_name)
            if os.path.exists(out_path) and not args.overwrite:
                logger.info(f"[skip] {out_path} exists (use --overwrite)")
                continue
            uids, paths, refs, missing = collect_from_dir(wav_dir, transcripts)
            if missing:
                logger.warning(f"[{os.path.basename(wav_dir)}] {len(missing)} wavs had no "
                               f"transcript (e.g. {missing[:3]}) -- skipped")
            if not uids:
                logger.warning(f"[{os.path.basename(wav_dir)}] nothing to score; skipping")
                continue
            score_and_write(uids, paths, refs, model, args, out_path)
    else:
        key = "target" if args.role == "healthy" else "source"
        out_dir = args.out_dir or os.path.dirname(args.manifest)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"asr_{args.role}.txt")
        if os.path.exists(out_path) and not args.overwrite:
            logger.info(f"[skip] {out_path} exists (use --overwrite)")
        else:
            uids, paths, refs, missing = collect_from_manifest(uid_map, transcripts, key)
            if missing:
                logger.warning(f"[{args.role}] {len(missing)} uids had no transcript "
                               f"(e.g. {missing[:3]}) -- skipped")
            score_and_write(uids, paths, refs, model, args, out_path)

    logger.info("[asr-nemo] ALL DONE")


if __name__ == "__main__":
    main()
