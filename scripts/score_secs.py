"""
Stage 4: speaker-similarity (SECS) scoring with ECAPA-TDNN over vocoded wav folders.

For each prediction wav folder, writes ONE text file (default: secs_scores.txt) INSIDE it:
    uid   cos_target   cos_source
    ...one row per wav...
    MEAN  <m>          <m>
    STD   <s>          <s>
    N     <n>
    # per-speaker and per-condition mean cos_target breakdown

Metric = cosine similarity between L2-comparable ECAPA embeddings (the standard SECS
metric; ECAPA is trained with an angular-margin loss, so cosine is its native distance).
  * cos_target : prediction  vs  its GT target  (manifest col3 = file_path; healthyTrim.wav
                 for 0010/0012 BM/PA, healthy.wav elsewhere). This is the headline SECS.
  * cos_source : prediction  vs  the INPUT pathological wav (manifest col2). Leakage check --
                 if this is ~= cos_target, the model is copying the source, not converting.

Pairing: the prediction filename is the uid '<speaker>-<COND>-<utterance>.wav'. We rebuild
uid -> (target, source) from the manifest with the SAME _uid() logic dump_mels_eval.py uses
to name the mels, so the pairing is exact (no string guessing).

Embeddings are computed online (never written to disk): each is used once, extraction is the
only cost (~1-2 min for the whole test set on one GPU), so persisting them would only add IO.

    python scripts/score_secs.py <wav_dir> [<wav_dir> ...] --manifest <PATH-HE.tsv>

Needs internet on first run to fetch the ECAPA weights (cached afterwards).
"""

import os
import re
import sys
import argparse
import logging

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

logger = logging.getLogger("secs")

SR = 16000
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_MANIFEST = ("/data/fs201163/eo49197/VoiceConversion-fwf/"
                    "dub-autoalign-healthyYT-trim/testPATH-HE.tsv")
DEFAULT_SAVEDIR = "/data/fs201163/eo49197/MultiModalVC/pretrained_models/ecapa-voxceleb"


def _uid(input_wav):
    """MUST match dump_mels_eval._uid: <speaker>-<COND>-<utterance> from the two input
    folders. e.g. .../0019-processed-MP/IS0019shortstories_0008/pathological.wav
      -> '0019-MP-shortstories_0008'. Kept in sync by hand (copied verbatim)."""
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


def build_uid_map(manifest):
    """uid -> {'target': col3, 'source': col2} for every manifest row."""
    m = {}
    with open(manifest) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3 or "pathological.wav" not in cols[1]:
                continue  # skip the '/' root header line and any malformed row
            uid = _uid(cols[1])
            m[uid] = {"source": cols[1], "target": cols[2]}
    return m


def load_wav16(path):
    """1-D float32 tensor at 16 kHz mono."""
    # Read with soundfile (NOT torchaudio.load: torchaudio 2.9 routes through torchcodec,
    # whose native lib libtorchcodec_core*.so fails to load on the cluster -- missing
    # libnppicc.so.12). soundfile reads any sr; we resample below with pure-torch ops.
    data, sr = sf.read(path, dtype="float32", always_2d=True)   # (time, channels)
    w = torch.from_numpy(data.T)                                # (channels, time)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.squeeze(0)


@torch.no_grad()
def embed_paths(model, paths, device, batch_size=16):
    """ECAPA embedding per path. Batched with wav_lens so zero-padding is masked out of
    the time pooling (unbiased) and the GPU stays busy. Returns (len(paths), 192)."""
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = [load_wav16(p) for p in paths[i:i + batch_size]]
        lens = torch.tensor([w.numel() for w in chunk], dtype=torch.float)
        maxlen = int(lens.max())
        batch = torch.zeros(len(chunk), maxlen)
        for j, w in enumerate(chunk):
            batch[j, : w.numel()] = w
        rel = (lens / maxlen).to(device)
        emb = model.encode_batch(batch.to(device), wav_lens=rel)   # (b, 1, 192)
        out.append(emb.squeeze(1).cpu())
    return torch.cat(out, 0)


def _mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    t = torch.tensor(xs)
    return t.mean().item(), (t.std(unbiased=False).item() if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_dirs", nargs="+", help="One or more folders of prediction .wav files.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-name", default="secs_scores.txt")
    ap.add_argument("--savedir", default=DEFAULT_SAVEDIR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-source", action="store_true", help="Skip the similarity-to-source check.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    from speechbrain.inference.speaker import EncoderClassifier
    logger.info(f"[ecapa] loading {ECAPA_SOURCE} -> {device}")
    model = EncoderClassifier.from_hparams(
        source=ECAPA_SOURCE, savedir=args.savedir, run_opts={"device": device})
    model.eval()

    uid_map = build_uid_map(args.manifest)
    logger.info(f"[manifest] {args.manifest}: {len(uid_map)} uid->target entries")

    for wav_dir in args.wav_dirs:
        wav_dir = wav_dir.rstrip("/")
        out_path = os.path.join(wav_dir, args.out_name)
        if os.path.exists(out_path) and not args.overwrite:
            logger.info(f"[skip] {out_path} exists (use --overwrite)")
            continue

        files = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))
        uids, preds, tgts, srcs, missing = [], [], [], [], []
        for f in files:
            uid = f[:-4]
            rec = uid_map.get(uid)
            if rec is None:
                missing.append(uid)
                continue
            uids.append(uid)
            preds.append(os.path.join(wav_dir, f))
            tgts.append(rec["target"])
            srcs.append(rec["source"])
        if missing:
            logger.warning(f"[{os.path.basename(wav_dir)}] {len(missing)} wavs had no manifest "
                           f"match (e.g. {missing[:3]}) -- skipped")
        if not uids:
            logger.warning(f"[{os.path.basename(wav_dir)}] no matched wavs; skipping")
            continue
        logger.info(f"=== {wav_dir}  ({len(uids)} matched wavs)")

        e_pred = embed_paths(model, preds, device, args.batch_size)
        e_tgt = embed_paths(model, tgts, device, args.batch_size)
        cos_t = F.cosine_similarity(e_pred, e_tgt).tolist()
        if args.no_source:
            cos_s = [float("nan")] * len(uids)
        else:
            e_src = embed_paths(model, srcs, device, args.batch_size)
            cos_s = F.cosine_similarity(e_pred, e_src).tolist()

        # group means (per speaker, per condition) on cos_target
        by_spk, by_cond = {}, {}
        for uid, ct in zip(uids, cos_t):
            spk, cond = uid.split("-")[0], uid.split("-")[1]
            by_spk.setdefault(spk, []).append(ct)
            by_cond.setdefault(cond, []).append(ct)

        mt, st = _mean_std(cos_t)
        ms, ss = _mean_std([c for c in cos_s if c == c])  # drop NaNs
        with open(out_path, "w") as fh:
            fh.write("uid\tcos_target\tcos_source\n")
            for uid, ct, cs in zip(uids, cos_t, cos_s):
                fh.write(f"{uid}\t{ct:.4f}\t{cs:.4f}\n")
            fh.write(f"MEAN\t{mt:.4f}\t{ms:.4f}\n")
            fh.write(f"STD\t{st:.4f}\t{ss:.4f}\n")
            fh.write(f"N\t{len(uids)}\n")
            fh.write("# per-speaker mean cos_target\n")
            for k in sorted(by_spk):
                m_, s_ = _mean_std(by_spk[k])
                fh.write(f"#SPK\t{k}\t{m_:.4f}\t{len(by_spk[k])}\n")
            fh.write("# per-condition mean cos_target\n")
            for k in sorted(by_cond):
                m_, s_ = _mean_std(by_cond[k])
                fh.write(f"#COND\t{k}\t{m_:.4f}\t{len(by_cond[k])}\n")
        logger.info(f"    -> {out_path}   SECS(cos_target)={mt:.3f}+-{st:.3f}  "
                    f"cos_source={ms:.3f}  N={len(uids)}")

    logger.info("[secs] ALL DONE")


if __name__ == "__main__":
    main()
