"""
MelVC validation / evaluation — mel-domain metrics on a list of checkpoints.

Reproduces the EXACT training-validation pipeline: it loads the 'valid' split via
the task (same dataset class), batches it with fairseq's own batch iterator using
the checkpoint's max_tokens/batch_size and size-sorted ordered_indices, runs the
model under no_grad/eval (modality dropout off), and computes metrics PER BATCH
then averages them UNWEIGHTED over batches — exactly as criterionMelVC.reduce_metrics
does. So the clean-condition loss_mel / ssim land on the numbers in your training
logs. Single GPU, single process, multi-worker data loading (fast).

Per checkpoint x condition it reports:
    * loss_mel  — L1 vs GT log-mel (training objective)            [matches training]
    * ssim      — global SSIM (criterionMelVC._global_ssim)        [matches training]
    * mcd       — TRUTHFUL MCD in dB (orthonormal DCT, c1..c13). NOT the inflated
                  ~165-200 metric the training plots showed (79 coeffs + un-normed DCT).
    * cos       — per-frame cosine sim of the linear mel envelopes
    * lsd       — log-spectral distance in dB
    * rmse      — Mel-L2: RMSE on the log-mel
    * sc        — spectral convergence on the linear mel

All metrics are MEL-DOMAIN — no vocoder. (Pitch metrics like F0/GPE/VDE are NOT
here: F0 only exists on a waveform, which would require vocoding the predicted mel
through BigVGAN and confound this mel-generator ablation.)

Conditions (run together so you get baseline + noisy side by side):
    * clean      — no noise. Reproduces the training validation numbers.
    * babble_NdB — babble mixed into the INPUT audio at a FIXED SNR via the dataset's
                   own add_noise (target stays clean). The "is video useful" probe.

TWO CSVs are written every run:
    * --out         summary  : one row per checkpoint x condition (batch-averaged,
                               matches training-validation). Feeds the BAR plots.
    * --out-perutt  per-utt  : one row per checkpoint x condition x UTTERANCE
                               (the raw distribution). Feeds the BOX plots.

Whisper weights come ONLY from the checkpoint ([whisper-verify] enforces it).
Results stream to the CSVs, flushed after EVERY checkpoint, so a wall-clock timeout
never loses the checkpoints that already finished.

Edit CHECKPOINTS / DEFAULT_MANIFEST / DEFAULT_NOISE_WAV below, then run it. For all
four conditions in one launch:
    python evaluate_melvc.py --conditions clean,0,-5,-10
"""

import os
import sys
import csv
import math
import logging
import argparse

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from fairseq import checkpoint_utils

# --- PYTHONPATH + fairseq registration + no-stock-whisper monkeypatch ---------
current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path)
repo_root = os.path.dirname(scripts_dir)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Importing inference_melvc sets up sys.path, registers the fairseq model/criterion,
# and installs the monkeypatch that makes Whisper load from the checkpoint only.
import inference_melvc as inf  # noqa: F401  (side effects: sys.path, registration, patch)
from src.bigvgan_mel import mel_spectrogram, resample
from src.dataset_pathological_finetune import mms_pathological_finetune_dataset

logger = logging.getLogger("melvc-eval")


# ===========================================================================
# EDIT HERE. Paste the checkpoints to evaluate (swap freely between runs).
# ===========================================================================
CHECKPOINTS = [
    "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_audioonly_v1_headsalvage_complete/checkpoints/checkpoint_best.pt",
    "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_dropout_noise_v2_headsalvage_complete/checkpoints/checkpoint_best.pt",
    "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_de_stage2_from_en_adv/checkpoints/checkpoint31.pt",
    "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_de_stage2_from_en/checkpoints/checkpoint46.pt",
    "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_de_stage2_from_en_OGM/checkpoints/checkpoint71.pt",
    "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_p2h_audioonly_baseline/checkpoints/checkpoint120.pt",
    "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_p2h_audioonly_samelosses/checkpoints/checkpoint120.pt",
    "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_p2h_fullopen_aug_reg/checkpoints/checkpoint120.pt"
]

# The manifest the trainings validated on (split name -> validPATH-HE.tsv).
DEFAULT_MANIFEST = "/data/fs201163/eo49197/VoiceConversion-fwf/pre-trainVoxcelebYoutube/validPATH-HE.tsv"

# Babble noise file used during training (16 kHz mono). Same one the runs used.
DEFAULT_NOISE_WAV = os.path.join(repo_root, "noise", "babble_noise.wav")

# Noise conditions, as (label, snr_dB | None-for-clean). Papers' headline noisy
# condition is babble @ 0 dB (mWhisper-Flamingo & MMS-LLaMA).
DEFAULT_CONDITIONS = [("clean", None), ("babble_0dB", 0.0)]

MCD_NUM_COEF = 13   # keep c1..c13 (drop c0); classic MattShannon/mcd value
SOURCE_SR = 16000
TARGET_SR = 22050
HOP = 256
MEL_CFG = dict(sampling_rate=TARGET_SR, num_mels=80, n_fft=1024, hop_size=HOP,
               win_size=1024, fmin=0, fmax=8000)   # == criterionMelVC.mel_cfg
METRIC_KEYS = ("l1", "ssim", "mcd", "cos", "lsd", "rmse", "sc")


# ===========================================================================
# Metrics — mel domain. x (pred), y (gt): (B, bands, T) log-mel; mask: (B,1,T).
# Each returns ONE per-batch scalar; aggregation averages these UNWEIGHTED over
# batches, matching criterionMelVC.reduce_metrics (sum(vals)/len(vals)).
# ===========================================================================
def global_ssim(x, y, mask):
    """Identical to criterionMelVC._global_ssim (global SSIM over valid frames)."""
    m = mask.expand_as(x)
    n = m.sum(dim=(1, 2)).clamp(min=1)
    mx = (x * m).sum(dim=(1, 2)) / n
    my = (y * m).sum(dim=(1, 2)) / n
    vx = ((x - mx[:, None, None]) ** 2 * m).sum(dim=(1, 2)) / n
    vy = ((y - my[:, None, None]) ** 2 * m).sum(dim=(1, 2)) / n
    cov = ((x - mx[:, None, None]) * (y - my[:, None, None]) * m).sum(dim=(1, 2)) / n
    L = 14.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / \
           ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2) + 1e-8)
    return ssim.mean().item()


_ORTHO_DCT = {}


def _ortho_dct_basis(bands, device):
    """Orthonormal DCT-II basis (K=bands, M=bands). Cached per (bands, device)."""
    key = (bands, str(device))
    if key not in _ORTHO_DCT:
        n = torch.arange(bands, dtype=torch.float32)
        k = n.unsqueeze(1)
        basis = torch.cos(math.pi / bands * (n.unsqueeze(0) + 0.5) * k)  # (K, M)
        # Orthonormal scaling: c0 *= sqrt(1/N), ck *= sqrt(2/N). The training metric
        # was missing this (and kept all 79 coeffs) -> ~6.3x inflation.
        scale = torch.full((bands, 1), math.sqrt(2.0 / bands))
        scale[0, 0] = math.sqrt(1.0 / bands)
        _ORTHO_DCT[key] = (basis * scale).to(device)
    return _ORTHO_DCT[key]


def truthful_mcd(x, y, mask, num_coef, device):
    """MCD (dB): orthonormal DCT-II of log-mel, keep c1..c{num_coef} (drop c0).

    MCD = (10/ln10) * sqrt( 2 * sum_d (cx_d - cy_d)^2 ), averaged over valid frames.
    """
    bands = x.size(1)
    B = _ortho_dct_basis(bands, device)
    cx = torch.einsum("km,bmt->bkt", B, x)
    cy = torch.einsum("km,bmt->bkt", B, y)
    hi = min(num_coef + 1, bands)
    diff2 = (cx[:, 1:hi, :] - cy[:, 1:hi, :]) ** 2
    diff2 = diff2.sum(dim=1)
    valid = mask.squeeze(1)
    per_frame = (10.0 / math.log(10.0)) * torch.sqrt((2.0 * diff2).clamp(min=0))
    return ((per_frame * valid).sum() / valid.sum().clamp(min=1)).item()


def frame_cosine(x, y, mask):
    """Mean per-frame cosine similarity of the LINEAR mel envelopes (exp of log-mel)."""
    lx, ly = torch.exp(x), torch.exp(y)
    num = (lx * ly).sum(dim=1)
    den = lx.norm(dim=1) * ly.norm(dim=1) + 1e-8
    valid = mask.squeeze(1)
    return (((num / den) * valid).sum() / valid.sum().clamp(min=1)).item()


def log_spectral_distance(x, y, mask):
    """LSD in dB. Mel is natural-log magnitude -> dB = (20/ln10)*lnmag."""
    d2 = ((20.0 / math.log(10.0)) * (x - y)) ** 2
    per_frame = torch.sqrt(d2.mean(dim=1))
    valid = mask.squeeze(1)
    return ((per_frame * valid).sum() / valid.sum().clamp(min=1)).item()


def mel_rmse(x, y, mask):
    """RMSE on the log-mel (Mel-L2)."""
    per_frame = torch.sqrt(((x - y) ** 2).mean(dim=1))
    valid = mask.squeeze(1)
    return ((per_frame * valid).sum() / valid.sum().clamp(min=1)).item()


def spectral_convergence(x, y, mask):
    """Spectral convergence ||gt-pred||_F / ||gt||_F over the LINEAR mel spectrogram."""
    lx, ly = torch.exp(x), torch.exp(y)
    num = torch.sqrt((((ly - lx) * mask) ** 2).sum())
    den = torch.sqrt(((ly * mask) ** 2).sum()) + 1e-8
    return (num / den).item()


def per_sample_metrics(x, y, mask, num_coef, device):
    """SAME formulas as the per-batch metrics above, but WITHOUT the final reduction
    over the batch — returns a dict of (B,) tensors, one value per utterance. Feeds
    the per-utterance CSV (box plots). The per-batch functions above are untouched so
    the summary CSV keeps matching the training-validation numbers exactly.
    """
    bands = x.size(1)
    valid = mask.squeeze(1)                        # (B, m)
    nfr = valid.sum(dim=1).clamp(min=1)            # (B,)

    # l1 (== loss_mel): mean |x-y| over valid frames x bands, per sample
    l1 = ((x - y).abs() * mask).sum(dim=(1, 2)) / (nfr * bands)

    # global SSIM per sample (same math as global_ssim, sans .mean())
    m = mask.expand_as(x)
    n = m.sum(dim=(1, 2)).clamp(min=1)
    mx = (x * m).sum(dim=(1, 2)) / n
    my = (y * m).sum(dim=(1, 2)) / n
    vx = ((x - mx[:, None, None]) ** 2 * m).sum(dim=(1, 2)) / n
    vy = ((y - my[:, None, None]) ** 2 * m).sum(dim=(1, 2)) / n
    cov = ((x - mx[:, None, None]) * (y - my[:, None, None]) * m).sum(dim=(1, 2)) / n
    L = 14.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / \
           ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2) + 1e-8)

    # truthful MCD per sample
    Bd = _ortho_dct_basis(bands, device)
    cx = torch.einsum("km,bmt->bkt", Bd, x)
    cy = torch.einsum("km,bmt->bkt", Bd, y)
    hi = min(num_coef + 1, bands)
    diff2 = ((cx[:, 1:hi, :] - cy[:, 1:hi, :]) ** 2).sum(dim=1)      # (B, m)
    mcd_pf = (10.0 / math.log(10.0)) * torch.sqrt((2.0 * diff2).clamp(min=0))
    mcd = (mcd_pf * valid).sum(dim=1) / nfr

    # frame cosine per sample
    lx, ly = torch.exp(x), torch.exp(y)
    num = (lx * ly).sum(dim=1)
    den = lx.norm(dim=1) * ly.norm(dim=1) + 1e-8
    cos = ((num / den) * valid).sum(dim=1) / nfr

    # LSD per sample
    lsd_pf = torch.sqrt((((20.0 / math.log(10.0)) * (x - y)) ** 2).mean(dim=1))
    lsd = (lsd_pf * valid).sum(dim=1) / nfr

    # RMSE per sample
    rmse_pf = torch.sqrt(((x - y) ** 2).mean(dim=1))
    rmse = (rmse_pf * valid).sum(dim=1) / nfr

    # spectral convergence per sample
    num_sc = torch.sqrt((((ly - lx) * mask) ** 2).sum(dim=(1, 2)))
    den_sc = torch.sqrt(((ly * mask) ** 2).sum(dim=(1, 2))) + 1e-8
    sc = num_sc / den_sc

    return {"l1": l1, "ssim": ssim, "mcd": mcd, "cos": cos,
            "lsd": lsd, "rmse": rmse, "sc": sc}


# ===========================================================================
# Loading + datasets
# ===========================================================================
def load_model_and_cfg(ckpt, manifest_dir, device):
    """Verified MelVC load (mirrors inference_melvc.load_model) returning cfg + task."""
    w2v_path = os.path.join(repo_root, "pretrained_models/avhubert/large_vox_iter5.pt")
    overrides = {
        "task": {"data": manifest_dir, "label_dir": manifest_dir},
        "model": {"w2v_path": w2v_path, "stage1_checkpoint": "", "vocoder_checkpoint": ""},
    }
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck_cfg = raw.get("cfg", {})
    mc = ck_cfg.get("model", {}) if isinstance(ck_cfg, dict) else getattr(ck_cfg, "model", {})
    md = mc if isinstance(mc, dict) else OmegaConf.to_container(mc, resolve=True)
    if "use_cqt" not in md:
        overrides["model"]["use_cqt"] = False
    if "upsampling_method" not in md:
        overrides["model"]["upsampling_method"] = "interpolation"
    sd = raw.get("model", {})
    if any(k.startswith("upsample_conv1.") for k in sd):
        overrides["model"]["transconv_layers"] = (
            4 if any(k.startswith("upsample_conv4.") for k in sd)
            else 3 if any(k.startswith("upsample_conv3.") for k in sd) else 2)
    whisper_keys = [k for k in sd if k.startswith("whisper.")]
    if len(whisper_keys) < 300:
        raise RuntimeError(f"Whisper weights missing from checkpoint: only "
                           f"{len(whisper_keys)} 'whisper.*' keys. Refusing to run.")
    del raw, sd

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ckpt], arg_overrides=overrides, strict=False)
    model = models[0].to(device).eval()
    if not hasattr(model, "mel_head"):
        raise RuntimeError("Loaded model has no mel_head — is this a MelVC checkpoint?")
    logger.info(f"[whisper-verify] {len(whisper_keys)} 'whisper.*' keys present — OK")
    return model, cfg, task


def build_noisy_dataset(cfg_task, manifest_dir, snr_db, noise_wav):
    """Valid dataset with fixed-SNR babble on the INPUT (target stays clean).

    The dataset only injects noise on subset_name=='train', so we set that to open
    the gate (image_aug forced off, shuffle off -> same ordered_indices as 'valid').
    noise_prob=1.0 + snr_levels=[snr] -> every sample gets babble at exactly snr.
    """
    return mms_pathological_finetune_dataset(
        manifest_path=f"{manifest_dir}/validPATH-HE.tsv",
        sample_rate=cfg_task.sample_rate,
        max_sample_size=cfg_task.max_sample_size,
        shuffle=False,
        normalize=cfg_task.normalize,
        image_mean=cfg_task.image_mean,
        image_std=cfg_task.image_std,
        image_crop_size=cfg_task.image_crop_size,
        image_aug=False,
        modalities=cfg_task.modalities,
        subset_name="train",
        number_of_synths=cfg_task.number_of_synths,
        noise_wav=noise_wav,
        noise_prob=1.0,
        snr_levels=[snr_db],
    )


def make_iter(task, cfg, dataset, num_workers):
    """fairseq batch iterator over `dataset`, batched like training validation."""
    ds = cfg.dataset
    max_tokens = getattr(ds, "max_tokens_valid", None) or getattr(ds, "max_tokens", None)
    max_sentences = getattr(ds, "batch_size_valid", None) or getattr(ds, "batch_size", None)
    bi = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        max_positions=task.max_positions(),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=getattr(ds, "required_batch_size_multiple", 1),
        seed=getattr(getattr(cfg, "common", object()), "seed", 1),
        num_workers=num_workers,
        epoch=1,
        disable_iterator_cache=True,
    )
    return bi.next_epoch_itr(shuffle=False)


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


# ===========================================================================
def run_condition(model, itr, device, desc):
    """One pass. Returns:
        agg      — per-batch metrics averaged UNWEIGHTED over batches (== reduce_metrics),
                   for the summary/bar-plot CSV (matches training-validation).
        nsamples — number of utterances seen.
        per_utt  — list of {utt_id, l1, ssim, mcd, cos, lsd, rmse, sc} dicts, one per
                   utterance, for the box-plot CSV.
    """
    acc = {k: 0.0 for k in METRIC_KEYS}
    nbatches = 0
    nsamples = 0
    per_utt = []
    for sample in tqdm(itr, desc=desc, leave=False):
        if not sample:
            continue
        net_input = _to_device(sample["net_input"], device)
        source = net_input["source"]
        source["spk_embeddings"] = sample["spk_embeddings"].to(device)
        gt_wav = sample["target_waveform"].to(device).float()
        wav_lens = sample["waveform_lengths"].to(device)

        with torch.no_grad():
            gt_mel = mel_spectrogram(resample(gt_wav, SOURCE_SR, TARGET_SR), **MEL_CFG)
            T_mel = gt_mel.size(-1)
            mel_lengths = torch.clamp(
                ((wav_lens * TARGET_SR) // SOURCE_SR - HOP) // HOP + 1, min=1, max=T_mel)
            source["mel_target_lengths"] = mel_lengths
            net_input["source"] = source
            out = model(**net_input)
            pred = out["melspec"].transpose(1, 2).float()      # (B,80,T)

        m = min(pred.size(-1), gt_mel.size(-1))
        p, g = pred[..., :m], gt_mel[..., :m]
        idx = torch.arange(m, device=device).unsqueeze(0)
        mask = (idx < mel_lengths.clamp(max=m).unsqueeze(1)).float().unsqueeze(1)  # (B,1,m)

        # --- summary (batch-averaged, unchanged: matches training-validation) ---
        denom = mask.sum() * p.size(1) + 1e-8
        acc["l1"] += (((p - g).abs() * mask).sum() / denom).item()
        acc["ssim"] += global_ssim(p, g, mask)
        acc["mcd"] += truthful_mcd(p, g, mask, MCD_NUM_COEF, device)
        acc["cos"] += frame_cosine(p, g, mask)
        acc["lsd"] += log_spectral_distance(p, g, mask)
        acc["rmse"] += mel_rmse(p, g, mask)
        acc["sc"] += spectral_convergence(p, g, mask)
        nbatches += 1

        # --- per-utterance (raw distribution for box plots) ---
        ps = per_sample_metrics(p, g, mask, MCD_NUM_COEF, device)
        ids = sample.get("id")
        if torch.is_tensor(ids):
            ids = ids.tolist()
        else:
            ids = list(range(nsamples, nsamples + g.size(0)))
        for i in range(g.size(0)):
            per_utt.append({"utt_id": ids[i],
                            **{k: ps[k][i].item() for k in METRIC_KEYS}})

        nsamples += g.size(0)

    nb = max(nbatches, 1)
    return {k: acc[k] / nb for k in METRIC_KEYS}, nsamples, per_utt


def main():
    ap = argparse.ArgumentParser(description="MelVC mel-domain evaluation over checkpoints")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--noise-wav", default=DEFAULT_NOISE_WAV)
    ap.add_argument("--conditions", default=None,
                    help="Comma list run in ONE launch, e.g. 'clean,0,-5,-10'. 'clean' = "
                         "no noise; numbers = babble SNR in dB. Overrides --snr/--snr-list.")
    ap.add_argument("--snr", type=float, default=None,
                    help="Override: single SNR (dB) for the noisy condition (default 0).")
    ap.add_argument("--snr-list", default=None,
                    help="Comma-separated SNRs (dB), e.g. '-5,-10'. Evaluates ONLY those "
                         "noisy levels (no clean, no default 0 dB). Overrides --snr.")
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--noisy-only", action="store_true")
    ap.add_argument("--force-modality", default=None,
                    choices=["av", "audio_only", "video_only"])
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(scripts_dir, "melvc_eval_results.csv"),
                    help="Summary CSV (batch-averaged, one row per checkpoint x condition).")
    ap.add_argument("--out-perutt", default=None,
                    help="Per-utterance CSV for box plots. Default: <out>_perutt.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    conditions = list(DEFAULT_CONDITIONS)
    if args.conditions is not None:
        # Explicit list, clean + any noisy levels, all in one launch.
        conditions = []
        for tok in args.conditions.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower() == "clean":
                conditions.append(("clean", None))
            else:
                s = float(tok)
                conditions.append((f"babble_{s:g}dB", s))
    elif args.snr_list is not None:
        # ONLY the requested noisy levels — no clean, no default 0 dB.
        snrs = [float(s) for s in args.snr_list.split(",") if s.strip() != ""]
        conditions = [(f"babble_{s:g}dB", s) for s in snrs]
    elif args.snr is not None:
        conditions = [("clean", None), (f"babble_{args.snr:g}dB", args.snr)]
    if args.clean_only:
        conditions = [c for c in conditions if c[1] is None]
    if args.noisy_only:
        conditions = [c for c in conditions if c[1] is not None]

    manifest_dir = os.path.dirname(args.manifest)
    mcd_key = f"mcd_c1_{MCD_NUM_COEF}"
    cols = ["checkpoint", "modality", "condition", "n", "loss_mel", "ssim",
            mcd_key, "cos", "lsd", "rmse", "sc"]

    # Per-utterance CSV (box plots): one row per checkpoint x condition x utterance.
    out_perutt = args.out_perutt or (os.path.splitext(args.out)[0] + "_perutt.csv")
    perutt_cols = ["checkpoint", "modality", "condition", "utt_id", "loss_mel", "ssim",
                   mcd_key, "cos", "lsd", "rmse", "sc"]

    # Stream both CSVs, flushing per checkpoint, so a timeout never wipes finished work.
    csv_f = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=cols)
    writer.writeheader()
    csv_f.flush()
    pu_f = open(out_perutt, "w", newline="")
    pu_writer = csv.DictWriter(pu_f, fieldnames=perutt_cols)
    pu_writer.writeheader()
    pu_f.flush()
    logger.info(f"[eval] conditions={[c[0] for c in conditions]}  MCD=c1..c{MCD_NUM_COEF}  "
                f"workers={args.num_workers}")
    logger.info(f"[eval] summary   -> {args.out} (flushed per checkpoint)")
    logger.info(f"[eval] per-utt   -> {out_perutt} (flushed per checkpoint)")

    # Datasets are checkpoint-independent -> build once from the first cfg, reuse.
    ref_task = ref_cfg = None
    datasets = {}
    all_rows = []
    for ckpt in CHECKPOINTS:
        if not os.path.isfile(ckpt):
            logger.warning(f"missing checkpoint, skipping: {ckpt}")
            continue
        name = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
        model, cfg, task = load_model_and_cfg(ckpt, manifest_dir, device)
        if args.force_modality:
            model.modality_mode = args.force_modality
        modality = getattr(model, "modality_mode", "av")
        logger.info(f"[eval] {name}: modality_mode={modality}")

        if ref_task is None:
            ref_task, ref_cfg = task, cfg
            for label, snr in conditions:
                if snr is None:
                    task.load_dataset("valid")        # clean, identical to training valid
                    datasets[label] = task.dataset("valid")
                else:
                    # task.cfg (not cfg.task): the task's own config object holds
                    # sample_rate/image_* etc. — the same fields task.load_dataset
                    # uses. cfg.task is the raw struct-mode dict and lacks them.
                    datasets[label] = build_noisy_dataset(task.cfg, manifest_dir, snr,
                                                           args.noise_wav)

        rows = []
        pu_rows = []
        for label, _snr in conditions:
            itr = make_iter(ref_task, ref_cfg, datasets[label], args.num_workers)
            metrics, n, per_utt = run_condition(model, itr, device, f"{name}/{label}")
            row = {"checkpoint": name, "modality": modality, "condition": label, "n": n,
                   "loss_mel": metrics["l1"], "ssim": metrics["ssim"],
                   mcd_key: metrics["mcd"], "cos": metrics["cos"],
                   "lsd": metrics["lsd"], "rmse": metrics["rmse"], "sc": metrics["sc"]}
            rows.append(row)
            for u in per_utt:
                pu_rows.append({"checkpoint": name, "modality": modality,
                                "condition": label, "utt_id": u["utt_id"],
                                "loss_mel": u["l1"], "ssim": u["ssim"],
                                mcd_key: u["mcd"], "cos": u["cos"], "lsd": u["lsd"],
                                "rmse": u["rmse"], "sc": u["sc"]})
            logger.info(f"[eval] {name}/{label} n={n} "
                        f"L1={row['loss_mel']:.4f} SSIM={row['ssim']:.4f} "
                        f"MCD={row[mcd_key]:.2f} LSD={row['lsd']:.2f}")
        writer.writerows(rows)
        csv_f.flush()
        os.fsync(csv_f.fileno())
        pu_writer.writerows(pu_rows)
        pu_f.flush()
        os.fsync(pu_f.fileno())
        all_rows += rows
        logger.info(f"[eval] {name} done — {len(all_rows)} summary rows, "
                    f"{len(pu_rows)} per-utt rows this checkpoint")

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    csv_f.close()
    pu_f.close()

    if not all_rows:
        logger.error("no results produced.")
        return
    print("\n" + "=" * 118)
    print(f"{'checkpoint':42s} {'modality':11s} {'cond':11s} {'n':>6s} "
          f"{'L1':>7s} {'SSIM':>7s} {'MCD':>7s} {'cos':>6s} {'LSD':>7s} {'RMSE':>7s} {'SC':>6s}")
    print("-" * 118)
    for r in all_rows:
        print(f"{r['checkpoint'][:42]:42s} {r['modality']:11s} {r['condition']:11s} "
              f"{r['n']:>6d} {r['loss_mel']:>7.4f} {r['ssim']:>7.4f} "
              f"{r[mcd_key]:>7.2f} {r['cos']:>6.3f} {r['lsd']:>7.3f} "
              f"{r['rmse']:>7.3f} {r['sc']:>6.3f}")
    print("=" * 118)
    logger.info(f"[eval] wrote {args.out} and {out_perutt}")


if __name__ == "__main__":
    main()
