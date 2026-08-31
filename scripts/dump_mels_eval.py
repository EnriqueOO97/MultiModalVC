"""
Dump GENUINE-INFERENCE predicted mels for the evaluation pipeline (Stage 1).

Unlike dump_mels_gta.py (which teacher-forces to the GT-target length for BigVGAN
GTA), this script runs each checkpoint the way real inference runs: the output mel
length is derived from the INPUT (pathological) audio — never from the target wav,
which is supervision-only and never touches inference.

  * free mode  : duration predictor decides the length  (N = L_in * ratio).
                 The genuine voice-conversion output for a duration checkpoint.
  * tf   mode  : duration predictor disabled -> length = INPUT length.
                 (Classic checkpoints have no predictor, so this is their ONLY mode.)

Length source: we pass NO mel_target_lengths. forward_speech then derives the mel
frame count from source['audio_lengths'] (the input, already in the collater), i.e.
the exact path scripts/inference_melvc.py uses. out['target_lengths'] gives each
sample's real output length (predicted N in free mode, input length in tf mode).

Layout (father folder = the checkpoint tag; one npy folder per mode/modality inside):
    <out_root>/<tag>/<tag>__<mode>__<modality>/<uid>.npy    # (80, T) float32 log-mel
Re-dumping the same checkpoint with another mode/modality lands in the SAME father,
so Stage 2 vocodes everything for a checkpoint in one place (wavs go beside the npy).
Only .npy files are written; the uid (<speaker>-<COND>-<utterance>, from the input) maps
each mel back to the eval manifest for Stage 3, so no index file is needed.

Combinatorial: each checkpoint dumps  modes x modalities  folders.
  modes      : 'free' (predicted length) | 'tf' (input length; classic's only mode)
  modalities : 'av' | 'video_only' (audio masked) | 'audio_only' (video masked*)
  classic checkpoint  -> modes collapse to {tf}   -> up to 3 folders
  duration checkpoint -> modes {free, tf}         -> up to 6 folders
  (*audio_only uses the UNTRAINED video_mask_emb -> expected to be poor; by request.)

Config (JSON): a list of checkpoints, each with modes AND modalities to dump.
    [
      {"checkpoint": "/.../adv2_cont/checkpoints/checkpoint_best.pt",
       "modes": ["free", "tf"], "modalities": ["av", "video_only", "audio_only"],
       "tag": "adv2_cont_best"},
      {"checkpoint": "/.../melvc_p2h_from_de_stage2_OGM/checkpoints/checkpoint284.pt",
       "modes": ["tf"], "modalities": ["av", "video_only"], "tag": "classic_ogm"}
    ]
'tag' optional (default "<run_dir_name>_<ckpt_stem>"); 'modes'/'modalities' default
to ["free"]/["av"]. Classic checkpoints ignore 'free' (warned).

    python scripts/dump_mels_eval.py --config eval_checkpoints.json \
        --manifest /.../testPATH-HE.tsv --out-root /.../eval/mels
"""

import os
import re
import sys
import json
import argparse
import logging

import numpy as np
import torch
from tqdm import tqdm

_scripts = os.path.dirname(os.path.abspath(__file__))
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

import evaluate_melvc as ev
from src.dataset_pathological_finetune import mms_pathological_finetune_dataset

logger = logging.getLogger("melvc-dump-eval")


def build_clean_dataset(cfg_task, manifest_path):
    """Manifest dataset, CLEAN input (no babble), healthy target only."""
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
        subset_name="valid",        # noise gate CLOSED
        number_of_synths=0,         # only the real target
        noise_wav=None,
        noise_prob=0.0,
    )


def _paths_for(ds, dataset_index):
    """(input_wav, target_wav) absolute paths for a flat dataset index."""
    entry_idx, target_idx = ds.index_map[dataset_index]
    inp = ds._resolve(ds.patho_paths[entry_idx])
    tgt = ds._resolve(ds.target_lists[entry_idx][target_idx])
    return inp, tgt


def _uid(input_wav):
    """Human-readable id from the two input-path folders: <speaker>-<COND>-<utterance>.

    e.g. .../RECS/0019-processed-MP/IS0019shortstories_0008/pathological.wav
      folder1 '0019-processed-MP'      -> speaker '0019', COND 'MP'
      folder2 'IS0019shortstories_0008'-> strip leading '<letters><speaker>' id ('IS0019')
                                          -> utterance 'shortstories_0008'
      => '0019-MP-shortstories_0008'
    The id prefix (letters) is variable length (IS/MO/HE/MMY...), so we strip
    '^[A-Za-z]*<speaker>' rather than a fixed 6 chars. Collision-free: (speaker, COND,
    utt-folder) is the on-disk path, unique per utterance. Falls back to the raw
    'folder1__folder2' join if folder1 isn't the '<digits>-processed-<COND>' shape.
    """
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


def _default_tag(ckpt):
    run = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))  # <run>/checkpoints/x.pt
    return run  # father folder = the run name (parent of /checkpoints), no ckpt stem


def _ckpt_id(ckpt):
    """Checkpoint id used inside the npy folder name: the number from 'checkpointN.pt'
    ('250','500'...) or 'best' for 'checkpoint_best.pt'. Falls back to the stem."""
    stem = os.path.splitext(os.path.basename(ckpt))[0]      # e.g. checkpoint250 / checkpoint_best
    return re.sub(r"^checkpoint_?", "", stem) or stem


def dump_one_mode(model, ds, itr, device, out_dir, mode, modality, overwrite):
    """Run one inference pass in (`mode`, `modality`) and write ONLY the mel .npy files.

    mode     : 'free' (duration predictor decides length) | 'tf' (length = input length).
    modality : 'av' | 'video_only' (audio masked) | 'audio_only' (video masked).

    The .npy filename is the utterance uid (<speaker>-<COND>-<utterance>, from the input),
    which maps back to the eval manifest for Stage 3 -- so no index file is written.
    """
    os.makedirs(out_dir, exist_ok=True)

    # tf: disable duration predictor -> length comes from source['audio_lengths']
    # (the INPUT length). free: leave it as loaded. Restored below.
    saved_dp = getattr(model, "duration_predictor", None)
    if mode == "tf":
        model.duration_predictor = None

    # Modality: forward() reads self.modality_mode deterministically at inference
    # (dropout sampling only fires in the training forward), so setting it here picks
    # the mask. Restored below. NOTE: 'audio_only' uses video_mask_emb, which was NEVER
    # trained (p_audio_only=0 in training) -> those mels are expected to be poor.
    saved_mode = getattr(model, "modality_mode", "av")
    model.modality_mode = modality

    written = skipped = 0
    uid_seen = set()
    for sample in tqdm(itr, desc=os.path.basename(out_dir), leave=False):
        if not sample:
            continue
        net_input = ev._to_device(sample["net_input"], device)
        net_input["source"]["spk_embeddings"] = sample["spk_embeddings"].to(device)
        # NOTE: intentionally NO mel_target_lengths -> genuine inference length.
        ids = sample["id"].tolist()

        with torch.no_grad():
            out = model(**net_input)
            pred = out["melspec"].transpose(1, 2).float()      # (B, 80, Tmax)
            tl = out["target_lengths"].tolist()                # per-sample real length

        for b, dataset_index in enumerate(ids):
            T = int(min(tl[b], pred.size(-1)))
            inp, _ = _paths_for(ds, dataset_index)
            uid = _uid(inp)
            if uid in uid_seen:                                # ultra-rare safety net
                uid = f"{uid}__{dataset_index}"
            uid_seen.add(uid)

            out_npy = os.path.join(out_dir, uid + ".npy")
            if os.path.exists(out_npy) and not overwrite:
                skipped += 1
                continue
            mel_i = pred[b, :, :T].contiguous().cpu().numpy().astype(np.float32)
            np.save(out_npy, mel_i)
            written += 1

    model.duration_predictor = saved_dp                        # restore for next pass
    model.modality_mode = saved_mode
    logger.info(f"[{mode}/{modality}] {out_dir}: wrote {written} mels, skipped {skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="JSON list of {checkpoint, modes, modalities, tag?}.")
    ap.add_argument("--manifest", required=True, help="A *PATH-HE.tsv split.")
    ap.add_argument("--out-root", default="/data/fs201163/eo49197/DumpedMels",
                    help="Root dir; one father folder per run, npy folders per checkpoint inside.")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    manifest_dir = os.path.dirname(args.manifest)
    entries = json.load(open(args.config))

    for e in entries:
        ckpt = e["checkpoint"]
        modes = list(e.get("modes", ["free"]))
        modalities = list(e.get("modalities", ["av"]))
        tag = e.get("tag") or _default_tag(ckpt)
        ckpt_id = _ckpt_id(ckpt)
        logger.info(f"=== {tag}  modes={modes}  modalities={modalities}  <- {ckpt}")

        model, cfg, task = ev.load_model_and_cfg(ckpt, manifest_dir, device)
        has_dp = getattr(model, "duration_predictor", None) is not None
        logger.info(f"    duration_predictor present: {has_dp}")

        # Classic (no predictor): 'free' == 'tf'. Collapse to a single 'tf' pass.
        if not has_dp:
            if any(m == "free" for m in modes):
                logger.warning(f"    {tag}: classic checkpoint has no duration predictor; "
                               f"'free' is identical to 'tf' -> dumping 'tf' only.")
            modes = ["tf"]

        modes = [m for m in dict.fromkeys(modes) if m in ("free", "tf")]
        modalities = [m for m in dict.fromkeys(modalities)
                      if m in ("av", "video_only", "audio_only")]

        ds = build_clean_dataset(task.cfg, args.manifest)
        for mode in modes:                          # combinatorial: mode x modality
            for modality in modalities:
                if modality == "audio_only":
                    logger.warning(f"    {tag}: 'audio_only' uses the UNTRAINED video_mask_emb "
                                   f"(training used p_audio_only=0) -> mels expected to be poor.")
                itr = ev.make_iter(task, cfg, ds, args.num_workers)
                # Father folder = the run tag; the npy folder lives inside it, named
                # '<tag>-npy'. When a single (mode, modality) is dumped (the usual eval
                # case) that name is used verbatim. If several combos are dumped for the
                # same checkpoint, the mode/modality is appended so they don't collide.
                single_combo = len(modes) * len(modalities) == 1
                base_npy = f"{tag}-{ckpt_id}-npy"       # <run>-<ckpt#>-npy (or -best-npy)
                npy_name = base_npy if single_combo else f"{base_npy}__{mode}__{modality}"
                out_dir = os.path.join(args.out_root, tag, npy_name)
                dump_one_mode(model, ds, itr, device, out_dir, mode, modality, args.overwrite)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("[dump-eval] all checkpoints done.")


if __name__ == "__main__":
    main()
