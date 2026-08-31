"""
Stage 2: vocode dumped mel .npy folders to 16 kHz mono wavs (batched BigVGAN).

For each npy folder, writes a sibling wav folder INSIDE the same father folder:
    <father>/<npy_folder_name>/<uid>.npy          (input, from Stage 1)
    <father>/<npy_folder_name>__<vocoder_tag>/<uid>.wav   (output, 16 kHz mono)

The wav folder name = <npy_folder_name>__<vocoder_tag>, so it encodes checkpoint +
inference mode + modality (from the npy folder) AND the vocoder used.

BigVGAN outputs 22.05 kHz; every ASR/MOS model wants 16 kHz mono, so each clip is
resampled 22050->16000 and saved PCM_16. Inference is BATCHED (mels padded to the
batch max, vocoded in one pass, each output trimmed to frames*hop before resample) --
much faster than one-by-one, which matters since the vocoder is light.

Config (JSON): a list of vocoder jobs.
    [
      {"generator": "/.../bigvganGTA/p2h_from_de_stage2_OGM_ck500/g_best_mel",
       "tag": "p2h_ck500_gbestmel",
       "npy_folders": [
         "/data/fs201163/eo49197/DumpedMels/classic_ogm_ckpt500/classic_ogm_ckpt500__tf__av"
       ]}
    ]
'generator' is a BigVGAN generator checkpoint file ({"generator": state_dict}); its
folder must contain config.json. 'tag' names the vocoder in the output folder name.

    python scripts/dump_wavs_eval.py --config vocoder_jobs.json [--batch-size 32]
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import torch
import soundfile as sf
import torchaudio

logger = logging.getLogger("melvc-vocode")

DEFAULT_BIGVGAN_CODE_DIR = "/data/fs201163/eo49197/BigVGAN"
BIGVGAN_SR = 22050
OUT_SR = 16000
HOP = 256
EXPECTED_MEL = dict(sampling_rate=22050, num_mels=80, n_fft=1024, hop_size=256, fmax=8000)


def load_bigvgan(generator_file, code_dir, device):
    """Build BigVGAN from the config.json next to `generator_file`, then load that
    generator's weights (GTA-finetuned g_* file, i.e. {"generator": state_dict})."""
    ckpt_dir = os.path.dirname(generator_file)
    if not os.path.isfile(os.path.join(ckpt_dir, "config.json")):
        raise FileNotFoundError(f"config.json not found next to generator: {ckpt_dir}")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import bigvgan as bigvgan_mod

    # Build the architecture straight from config.json, then load the requested generator.
    # (NOT from_pretrained: it also requires a bigvgan_generator.pt in the folder, which some
    # GTA runs don't ship — they only have g_* files.) load_state_dict must happen BEFORE
    # remove_weight_norm (g_* files store weight_g/weight_v params).
    h_cfg = bigvgan_mod.load_hparams_from_json(os.path.join(ckpt_dir, "config.json"))
    voc = bigvgan_mod.BigVGAN(h_cfg, use_cuda_kernel=False)
    g = torch.load(generator_file, map_location="cpu")
    state = g["generator"] if isinstance(g, dict) and "generator" in g else g
    voc.load_state_dict(state)
    voc.remove_weight_norm()
    voc = voc.eval().to(device)
    for p in voc.parameters():
        p.requires_grad = False

    h = voc.h
    got = dict(sampling_rate=h.sampling_rate, num_mels=h.num_mels,
               n_fft=h.n_fft, hop_size=h.hop_size, fmax=h.fmax)
    for k, v in EXPECTED_MEL.items():
        if int(got[k]) != int(v):
            raise RuntimeError(f"BigVGAN mel mismatch on '{k}': {got[k]} vs {v}.")
    logger.info(f"[bigvgan] loaded {os.path.basename(generator_file)}  mel OK: {got}")
    return voc


def vocode_folder(voc, npy_dir, wav_dir, device, resampler, batch_size, overwrite):
    os.makedirs(wav_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(npy_dir) if f.endswith(".npy"))

    # (uid, frames) then sort by length to minimize padding waste.
    items = []
    for f in files:
        uid = f[:-4]
        out_wav = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_wav) and not overwrite:
            continue
        arr = np.load(os.path.join(npy_dir, f))          # (80, T)
        items.append((uid, arr))
    items.sort(key=lambda x: x[1].shape[1])

    written = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        lens = [a.shape[1] for _, a in batch]
        maxT = max(lens)
        x = torch.zeros(len(batch), EXPECTED_MEL["num_mels"], maxT, dtype=torch.float32)
        for b, (_, a) in enumerate(batch):
            x[b, :, :a.shape[1]] = torch.from_numpy(a)
        x = x.to(device)

        with torch.no_grad():
            wav = voc(x).squeeze(1).float()              # (B, maxT*hop) @ 22050
            for b, (uid, _) in enumerate(batch):
                n = lens[b] * HOP
                w22 = wav[b, :n].unsqueeze(0)            # (1, n)
                w16 = resampler(w22).squeeze(0).clamp(-1.0, 1.0).cpu().numpy()
                sf.write(os.path.join(wav_dir, uid + ".wav"), w16, OUT_SR, subtype="PCM_16")
                written += 1

    logger.info(f"    {wav_dir}: wrote {written} wavs "
                f"({len(files)} npy, {len(files) - len(items)} skipped existing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON list of {generator, tag, npy_folders}.")
    ap.add_argument("--bigvgan-code-dir", default=DEFAULT_BIGVGAN_CODE_DIR)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    jobs = json.load(open(args.config))
    resampler = torchaudio.transforms.Resample(BIGVGAN_SR, OUT_SR).to(device)

    for job in jobs:
        gen = job["generator"]
        tag = job["tag"]
        npy_folders = job["npy_folders"]
        logger.info(f"=== vocoder {tag}  <- {gen}  ({len(npy_folders)} npy folders)")
        voc = load_bigvgan(gen, args.bigvgan_code_dir, device)

        for npy_dir in npy_folders:
            npy_dir = npy_dir.rstrip("/")
            father = os.path.dirname(npy_dir)                     # wavs go beside npy
            wav_dir = os.path.join(father, f"{os.path.basename(npy_dir)}__{tag}")
            logger.info(f"  vocoding {npy_dir}  ->  {wav_dir}")
            vocode_folder(voc, npy_dir, wav_dir, device, resampler,
                          args.batch_size, args.overwrite)

        del voc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("[vocode] all jobs done. ALL DONE")


if __name__ == "__main__":
    main()
