"""
Dump teacher-forced predicted mels for BigVGAN GTA fine-tuning.

Runs the FROZEN MelVC checkpoint over a pathological->healthy manifest and writes,
for every (pathological-input, healthy-target) pair, the model's PREDICTED log-mel
as a float32 .npy placed RIGHT NEXT TO the healthy target wav, same basename:

    .../segment65_Healthy.wav   (the GT target, column `file_path`)
    .../segment65_Healthy.npy   (<- written here)

Why offline: the acoustic model is frozen, so its predicted mel for a clip is
deterministic. Computing it once and caching beats recomputing it every BigVGAN
step/epoch (which would redundantly re-run the ~850M-param AV forward + video
decode dozens of times). This is exactly BigVGAN's native `--fine_tuning` contract
(precomputed mels loaded from disk).

Pipeline (identical teacher-forcing to training/eval, reusing evaluate_melvc):
  * load checkpoint (Whisper weights enforced from the ckpt),
  * build the manifest's dataset CLEAN (no babble) with number_of_synths=0 so ONLY
    the real healthy target is used (one mel per manifest entry),
  * fairseq batch iterator (size-sorted, multi-worker) — same batching as training,
  * per sample: derive the GT-mel length T (resample target->22050 + BigVGAN mel),
    teacher-force the model to T, save pred_mel[:, :T] as <target_stem>.npy.

The saved mel has exactly T = floor(L_22k / hop) frames. The companion 22050 Hz
target wav (made later) must be trimmed to T*hop samples — and the resample step
should read T back from this .npy to guarantee frame-exact <mel, wav> alignment
for BigVGAN's `audio_len == mel_frames * hop` assert.

NO metrics, NO vocoder, NO GT mel saved — only the predicted mel.

    python scripts/dump_mels_gta.py --manifest <path/to/trainPATH-HE.tsv> [--checkpoint ...]
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path)
repo_root = os.path.dirname(scripts_dir)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Reuse the verified loader / iterator / teacher-forcing constants from the eval
# script (importing it also registers the fairseq model+criterion and installs the
# Whisper-from-checkpoint monkeypatch).
import evaluate_melvc as ev
from src.bigvgan_mel import mel_spectrogram, resample
from src.dataset_pathological_finetune import mms_pathological_finetune_dataset

logger = logging.getLogger("melvc-dump")

SOURCE_SR = ev.SOURCE_SR        # 16000
TARGET_SR = ev.TARGET_SR        # 22050
HOP = ev.HOP                    # 256
MEL_CFG = ev.MEL_CFG            # == criterionMelVC.mel_cfg (80/1024/256/1024/0/8000 @22050)

# Best P2H checkpoint — the model BigVGAN will be GTA-paired with.
DEFAULT_CHECKPOINT = ("/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_p2h_fullopen_aug_reg/checkpoints/checkpoint120.pt")


def build_clean_dataset(cfg_task, manifest_path):
    """Manifest dataset, CLEAN input, healthy target only (number_of_synths=0)."""
    return mms_pathological_finetune_dataset(
        manifest_path=manifest_path,
        sample_rate=cfg_task.sample_rate,
        max_sample_size=cfg_task.max_sample_size,
        shuffle=False,
        normalize=cfg_task.normalize,
        image_mean=cfg_task.image_mean,
        image_std=cfg_task.image_std,
        image_crop_size=cfg_task.image_crop_size,
        image_aug=False,
        modalities=cfg_task.modalities,
        subset_name="valid",        # noise gate CLOSED (only opens on 'train')
        number_of_synths=0,         # ONLY the real healthy target (column file_path)
        noise_wav=None,
        noise_prob=0.0,
    )


def target_npy_path(ds, dataset_index):
    """Absolute <healthy_target_stem>.npy for a given dataset (input,target) index."""
    entry_idx, target_idx = ds.index_map[dataset_index]
    tpath = ds._resolve(ds.target_lists[entry_idx][target_idx])
    return os.path.splitext(tpath)[0] + ".npy"


def main():
    ap = argparse.ArgumentParser(description="Dump teacher-forced predicted mels for GTA")
    ap.add_argument("--manifest", required=True,
                    help="Full path to a *PATH-HE.tsv split (e.g. trainPATH-HE.tsv).")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-dump even if the .npy already exists (default: skip).")
    ap.add_argument("--teacher-force-duration", action="store_true",
                    help="Dump-only: disable the duration predictor so the mel is generated "
                         "at GT (mel_target_lengths) length instead of the predicted length. "
                         "NEVER the default; opt-in here only. Does not touch the model on disk.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    manifest_dir = os.path.dirname(args.manifest)
    logger.info(f"[dump] checkpoint={args.checkpoint}")
    logger.info(f"[dump] manifest={args.manifest}")

    model, cfg, task = ev.load_model_and_cfg(args.checkpoint, manifest_dir, device)
    modality = getattr(model, "modality_mode", "av")
    logger.info(f"[dump] modality_mode={modality} (eval -> dropout off)")

    # Teacher forcing (dump-only, opt-in): nulling the in-memory duration_predictor
    # makes _maybe_predict_duration early-return target_lengths (== mel_target_lengths,
    # the GT length passed below), so the mel is emitted at GT length and frame-aligns
    # with the GT audio for GTA. This mutates only THIS process's model object; the
    # checkpoint on disk, training, and inference_melvc are untouched.
    if args.teacher_force_duration:
        model.duration_predictor = None
        logger.info("[dump] TEACHER FORCING on: duration predictor disabled "
                    "-> mels generated at GT (mel_target_lengths) length")

    ds = build_clean_dataset(task.cfg, args.manifest)
    itr = ev.make_iter(task, cfg, ds, args.num_workers)

    written = skipped = 0
    for sample in tqdm(itr, desc=os.path.basename(args.manifest), leave=False):
        if not sample:
            continue
        net_input = ev._to_device(sample["net_input"], device)
        source = net_input["source"]
        source["spk_embeddings"] = sample["spk_embeddings"].to(device)
        gt_wav = sample["target_waveform"].to(device).float()
        wav_lens = sample["waveform_lengths"].to(device)
        ids = sample["id"].tolist()

        with torch.no_grad():
            # GT mel only to derive the teacher-forcing length T (not saved).
            gt_mel = mel_spectrogram(resample(gt_wav, SOURCE_SR, TARGET_SR), **MEL_CFG)
            T_mel = gt_mel.size(-1)
            mel_lengths = torch.clamp(
                ((wav_lens * TARGET_SR) // SOURCE_SR - HOP) // HOP + 1, min=1, max=T_mel)
            source["mel_target_lengths"] = mel_lengths
            net_input["source"] = source
            out = model(**net_input)
            pred = out["melspec"].transpose(1, 2).float()    # (B, 80, T_pred)

        Tpred = pred.size(-1)
        for b, dataset_index in enumerate(ids):
            Ti = int(min(int(mel_lengths[b].item()), Tpred))
            mel_i = pred[b, :, :Ti].contiguous().cpu().numpy().astype(np.float32)  # (80, Ti)
            out_npy = target_npy_path(ds, dataset_index)
            if os.path.exists(out_npy) and not args.overwrite:
                skipped += 1
                continue
            np.save(out_npy, mel_i)
            written += 1

    logger.info(f"[dump] done: wrote {written} mels, skipped {skipped} existing "
                f"(total pairs {len(ds)}).")


if __name__ == "__main__":
    main()
