"""
Stage 3a: neural MOS scoring (DNSMOS Pro + UTMOS) over wav folders.

For each wav folder, writes ONE text file (default: mos_scores.txt) INSIDE that folder:
    uid   dnsmos   utmos
    ...one row per wav...
    MEAN  <mean>   <mean>
    STD   <std>    <std>
    N     <count>

Both predictors output a single MOS (~1-5). DNSMOS Pro: local TorchScript model
(runs/<variant>/model_best.pt) + its STFT, replicated INLINE below with the exact
training params so we do NOT have to import their gin-decorated utils module (i.e. no
`gin` dependency). UTMOS: the packaged UTMOS22-strong via torch.hub (tarepan/SpeechMOS)
-- self-contained, no fairseq/lightning. Both run in torchEnv on one GPU.

Per-file (not batched) on purpose: MOS predictors pool over time, so zero-padding a
batch biases the score. For a few hundred clips the per-file cost is trivial (<~2 min).

    python scripts/score_mos.py <wav_dir> [<wav_dir> ...] [--dnsmos-variant BVCC]

Needs internet on first run to fetch the UTMOS weights (cached afterwards).
"""

import os
import re
import sys
import argparse
import logging

import numpy as np
import torch
import librosa
import soundfile as sf

logger = logging.getLogger("mos")

DNSMOS_DIR = "/data/fs201163/eo49197/DNSMOSPro"
SR = 16000
DEFAULT_MANIFEST = ("/data/fs201163/eo49197/VoiceConversion-fwf/"
                    "dub-autoalign-healthyYT-trim/testPATH-HE.tsv")


def _uid(input_wav):
    """MUST match dump_mels_eval._uid / score_asr._uid: <speaker>-<COND>-<utterance>
    from the two input folders. Kept in sync by hand (copied verbatim)."""
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
    """uid -> {'target': col3 (healthy file_path), 'source': col2 (pathological)}.
    Reference-free MOS needs no transcripts; the role just picks which wav to score."""
    m = {}
    with open(manifest) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3 or "pathological.wav" not in cols[1]:
                continue  # skip header line and malformed rows
            m[_uid(cols[1])] = {"source": cols[1], "target": cols[2]}
    return m


def dnsmos_stft(samples):
    """Exact replica of DNSMOSPro/utils.stft defaults (win/hop/n_fft=320/160/320,
    log-magnitude). Same math as their module -- copied inline so we don't import
    utils.py, which does `import gin` at the top (gin is not installed / not needed)."""
    spec = librosa.stft(y=samples, win_length=320, hop_length=160, n_fft=320).T
    spec = np.abs(spec)
    spec = np.clip(spec, 1e-7, 1e7)
    return np.log10(spec)


def load_wav16(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    return w


def score_and_write(items, dnsmos, utmos, device, out_path):
    """Score a list of (uid, wav_path) with DNSMOS Pro + UTMOS; write the mos file."""
    logger.info(f"=== {out_path}  ({len(items)} wavs)")
    rows, dns_all, ut_all = [], [], []
    for uid, path in items:
        w = load_wav16(path)
        with torch.no_grad():
            spec = torch.FloatTensor(dnsmos_stft(w)).to(device)      # (T, F)
            d = float(dnsmos(spec[None, None, ...])[:, 0].item())     # mean MOS
            wt = torch.from_numpy(w).float().unsqueeze(0).to(device)  # (1, T)
            u = float(utmos(wt, SR).item())
        rows.append((uid, d, u))
        dns_all.append(d)
        ut_all.append(u)

    with open(out_path, "w") as fh:
        fh.write("uid\tdnsmos\tutmos\n")
        for uid, d, u in rows:
            fh.write(f"{uid}\t{d:.4f}\t{u:.4f}\n")
        md = sum(dns_all) / len(dns_all) if dns_all else float("nan")
        mu = sum(ut_all) / len(ut_all) if ut_all else float("nan")
        sd = float(np.std(dns_all)) if dns_all else float("nan")   # population std (ddof=0)
        su = float(np.std(ut_all)) if ut_all else float("nan")
        fh.write(f"MEAN\t{md:.4f}\t{mu:.4f}\n")
        fh.write(f"STD\t{sd:.4f}\t{su:.4f}\n")
        fh.write(f"N\t{len(rows)}\t{len(rows)}\n")
    logger.info(f"    -> {out_path}   mean dnsmos={md:.3f}  utmos={mu:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_dirs", nargs="*", help="Folders of <uid>.wav (role=pred only).")
    ap.add_argument("--role", choices=["pred", "healthy", "source"], default="pred",
                    help="pred: score each wav_dir. healthy/source: MOS ceiling/floor from "
                         "the manifest (col3 healthy target / col2 pathological input).")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", default=None, help="Where to write mos_<role>.txt (baselines).")
    ap.add_argument("--dnsmos-dir", default=DNSMOS_DIR)
    ap.add_argument("--dnsmos-variant", default="BVCC", choices=["BVCC", "NISQA", "VCC2018"])
    ap.add_argument("--out-name", default="mos_scores.txt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # --- DNSMOS Pro (local TorchScript) ---
    dns_path = os.path.join(args.dnsmos_dir, "runs", args.dnsmos_variant, "model_best.pt")
    logger.info(f"[dnsmos] loading {dns_path}")
    dnsmos = torch.jit.load(dns_path, map_location=device).eval()

    # --- UTMOS22-strong (torch.hub, self-contained) ---
    logger.info("[utmos] loading tarepan/SpeechMOS utmos22_strong via torch.hub ...")
    utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
    utmos = utmos.to(device).eval()

    if args.role == "pred":
        if not args.wav_dirs:
            ap.error("role=pred needs at least one wav_dir")
        for wav_dir in args.wav_dirs:
            wav_dir = wav_dir.rstrip("/")
            out_path = os.path.join(wav_dir, args.out_name)
            if os.path.exists(out_path) and not args.overwrite:
                logger.info(f"[skip] {out_path} exists (use --overwrite)")
                continue
            items = [(f[:-4], os.path.join(wav_dir, f))
                     for f in sorted(x for x in os.listdir(wav_dir) if x.endswith(".wav"))]
            score_and_write(items, dnsmos, utmos, device, out_path)
    else:
        key = "target" if args.role == "healthy" else "source"   # healthy=ceiling, source=floor
        out_dir = args.out_dir or os.path.dirname(args.manifest)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"mos_{args.role}.txt")
        if os.path.exists(out_path) and not args.overwrite:
            logger.info(f"[skip] {out_path} exists (use --overwrite)")
        else:
            uid_map = build_uid_map(args.manifest)
            items = [(uid, rec[key]) for uid, rec in uid_map.items() if os.path.isfile(rec[key])]
            logger.info(f"[{args.role}] {len(items)} wavs from manifest col '{key}'")
            score_and_write(items, dnsmos, utmos, device, out_path)

    logger.info("[mos] ALL DONE")


if __name__ == "__main__":
    main()
