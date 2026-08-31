"""
Stage 5: ASR intelligibility (WER / CER) with Whisper-small over wav sets.

Transcribes wavs with openai/whisper-small (German) and scores them against the
per-utterance reference transcripts with jiwer. Writes ONE text file per wav set:
    uid   WER   CER
    ...one row per wav...
    MEAN    <per-sample mean WER>   <per-sample mean CER>
    STD     <...>                   <...>
    CORPUS  <aggregate WER>         <aggregate CER>     # edits/words over the whole set
    N       <n>
    # per-speaker and per-condition mean WER
    #SPK  ...    #COND ...

Three roles give the intelligibility story (same ASR, same reference text):
  * pred   : the prediction wavs in the given folder(s) <uid>.wav      -> the result
  * healthy: the GT target wav (manifest col 'file_path'), per uid     -> ceiling
  * source : the input pathological wav (manifest col 'pathological')  -> floor
WER should go source (high) -> pred (lower) -> healthy (lowest).

Pairing: uid = '<speaker>-<COND>-<utterance>' (same _uid() as dump_mels_eval.py). The
utterance part (uid.split('-', 2)[2], maxsplit=2 because utterances contain hyphens,
e.g. 'common-dialogs_0020') keys into transcripts.csv (filename column minus '.wav').

  # predictions (writes asr_scores.txt inside each folder):
  python scripts/score_asr.py <wav_dir> [<wav_dir> ...] \
      --manifest /.../testPATH-HE.tsv --transcripts /.../transcripts.csv --role pred

  # baselines (writes asr_<role>.txt into --out-dir):
  python scripts/score_asr.py --role healthy --out-dir /some/dir \
      --manifest /.../testPATH-HE.tsv --transcripts /.../transcripts.csv

Needs internet on first run to fetch the Whisper weights (cached afterwards).
"""

import os
import re
import sys
import argparse
import logging

import torch
import torchaudio
import soundfile as sf

logger = logging.getLogger("asr")

SR = 16000
DEFAULT_MODEL = "openai/whisper-small"
DEFAULT_MANIFEST = ("/data/fs201163/eo49197/VoiceConversion-fwf/"
                    "dub-autoalign-healthyYT-trim/testPATH-HE.tsv")
DEFAULT_TRANSCRIPTS = "/data/fs201163/eo49197/VoiceConversion-fwf/transcripts.csv"


def _uid(input_wav):
    """MUST match dump_mels_eval._uid: <speaker>-<COND>-<utterance> from the two input
    folders. Kept in sync by hand (copied verbatim)."""
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
    """Utterance part of a uid. maxsplit=2 keeps hyphens inside the utterance."""
    parts = uid.split("-", 2)
    return parts[2] if len(parts) == 3 else uid


def build_uid_map(manifest):
    """uid -> {'target': col3 (file_path), 'source': col2 (pathological)} per manifest row."""
    m = {}
    with open(manifest) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3 or "pathological.wav" not in cols[1]:
                continue  # skip header line and malformed rows
            m[_uid(cols[1])] = {"source": cols[1], "target": cols[2]}
    return m


def load_transcripts(path):
    """utterance -> reference text, from a 'filename,transcript' csv (quoted transcripts)."""
    import csv
    txt = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
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
    """Shared hyp/ref normalizer: lowercase, drop punctuation (keep German letters),
    collapse whitespace. Underscores are word-internal noise here -> spaces."""
    s = s.lower().replace("_", " ")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def load_wav16(path):
    # Read with soundfile (NOT torchaudio.load: torchaudio 2.9 routes through torchcodec,
    # whose native lib fails to load on the cluster -- missing libnppicc.so.12).
    data, sr = sf.read(path, dtype="float32", always_2d=True)   # (time, channels)
    w = torch.from_numpy(data.T)                                # (channels, time)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.squeeze(0)


@torch.no_grad()
def transcribe(model, processor, paths, device, language, batch_size):
    """Whisper transcription per path, batched. Returns list[str] hypotheses."""
    hyps = []
    forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    for i in range(0, len(paths), batch_size):
        wavs = [load_wav16(p).numpy() for p in paths[i:i + batch_size]]
        feats = processor(wavs, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(device, dtype=model.dtype)
        gen = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=440)
        hyps.extend(processor.batch_decode(gen, skip_special_tokens=True))
    return hyps


def _mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    t = torch.tensor(xs)
    return t.mean().item(), (t.std(unbiased=False).item() if len(xs) > 1 else 0.0)


def score_and_write(uids, paths, refs, model, processor, device, args, out_path):
    import jiwer
    logger.info(f"=== {out_path}  ({len(uids)} wavs)")
    hyps = transcribe(model, processor, paths, device, args.language, args.batch_size)
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


def collect_from_dir(wav_dir, uid_map, transcripts):
    """(uids, paths, refs) for prediction wavs '<uid>.wav' in wav_dir that have a ref."""
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
    """(uids, paths, refs) for a baseline role: wav path = manifest col `key`."""
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
    ap.add_argument("--language", default="de")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    import jiwer  # noqa: F401  (fail fast if missing)
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    logger.info(f"[whisper] loading {args.model} -> {device}")
    processor = WhisperProcessor.from_pretrained(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

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
            uids, paths, refs, missing = collect_from_dir(wav_dir, uid_map, transcripts)
            if missing:
                logger.warning(f"[{os.path.basename(wav_dir)}] {len(missing)} wavs had no "
                               f"transcript (e.g. {missing[:3]}) -- skipped")
            if not uids:
                logger.warning(f"[{os.path.basename(wav_dir)}] nothing to score; skipping")
                continue
            score_and_write(uids, paths, refs, model, processor, device, args, out_path)
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
            score_and_write(uids, paths, refs, model, processor, device, args, out_path)

    logger.info("[asr] ALL DONE")


if __name__ == "__main__":
    main()
