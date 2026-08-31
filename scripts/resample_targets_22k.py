"""
Resample P2H healthy targets to 22050 Hz for BigVGAN GTA — CPU only, no GPU.

For every healthy target (column `file_path`) in the P2H train+valid manifests:
  * load the 16 kHz target wav,
  * resample 16000 -> 22050 with torchaudio.transforms.Resample (the SAME resampler
    src/bigvgan_mel.resample uses, so the audio matches how the mel length was set),
  * read T = frame count from the sibling predicted mel <stem>.npy,
  * trim/pad the resampled wav to EXACTLY T * hop(256) samples,
  * write <stem>-22k.wav at 22050 Hz (float32).

The trim-to-(T*256) is the only "trimming": it drops/pads at most 255 sub-frame
samples so BigVGAN's fine_tuning assert `audio_len == mel_frames * 256` holds
frame-exact (BigVGAN does the same hop-alignment in its non-finetuning branch).

Naming: 16 kHz original stays <stem>.wav (untouched); the 22050 copy is
<stem>-22k.wav; the mel is <stem>.npy — all co-located, no conflict.

Run on the LOGIN NODE (no SLURM, no GPU):
    python scripts/resample_targets_22k.py            # both splits
    python scripts/resample_targets_22k.py --manifest /path/to/oneSplit.tsv
"""

import os
import sys
import argparse

import numpy as np
import torch
import torchaudio
import soundfile as sf

SOURCE_SR = 16000
TARGET_SR = 22050
HOP = 256
DEFAULT_DATA = "/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT"
DEFAULT_SPLITS = ["trainPATH-HE", "validPATH-HE"]

# This resample is tiny; torch's default thread count (=cores/2, 88 on the login
# node) makes it ~156x SLOWER via thread-sync overhead: 842ms -> 5.4ms per file.
# Pure perf setting — same resampler/kernel/math as src/bigvgan_mel.resample.
torch.set_num_threads(1)
_resampler = torchaudio.transforms.Resample(SOURCE_SR, TARGET_SR)


def read_targets(manifest_path):
    """Column index 2 (`file_path`, the healthy target) of every data row."""
    targets = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if i == 0:  # header line
                continue
            line = line.rstrip("\n")
            if not line:
                continue
            targets.append(line.split("\t")[2])
    return targets


def process_one(target_wav, overwrite):
    stem = os.path.splitext(target_wav)[0]
    npy_path = stem + ".npy"
    out_wav = stem + "-22k.wav"
    if os.path.exists(out_wav) and not overwrite:
        return "skip"
    if not os.path.exists(npy_path):
        return f"NO_MEL:{npy_path}"

    wav, sr = sf.read(target_wav, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == SOURCE_SR, f"expected {SOURCE_SR}, got {sr} for {target_wav}"

    res = _resampler(torch.from_numpy(wav).float()).numpy()

    T = int(np.load(npy_path, mmap_mode="r").shape[-1])
    need = T * HOP
    if res.shape[0] >= need:
        res = res[:need]
    else:
        res = np.pad(res, (0, need - res.shape[0]), mode="constant")

    sf.write(out_wav, res.astype(np.float32), TARGET_SR, subtype="FLOAT")
    return "ok"


def main():
    ap = argparse.ArgumentParser(description="Resample P2H targets to 22050 for GTA")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--manifest", default=None,
                    help="Single manifest .tsv (default: both train+valid splits).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifests = ([args.manifest] if args.manifest
                 else [os.path.join(args.data, f"{s}.tsv") for s in DEFAULT_SPLITS])

    grand = {"ok": 0, "skip": 0, "nomel": 0}
    for mpath in manifests:
        targets = read_targets(mpath)
        n = len(targets)
        ok = skip = nomel = 0
        for i, t in enumerate(targets):
            r = process_one(t, args.overwrite)
            if r == "ok":
                ok += 1
            elif r == "skip":
                skip += 1
            else:
                nomel += 1
                if nomel <= 5:
                    print(f"  [warn] {r}")
            if (i + 1) % 500 == 0 or (i + 1) == n:
                print(f"{os.path.basename(mpath)}: {i + 1}/{n}  "
                      f"ok={ok} skip={skip} nomel={nomel}", flush=True)
        print(f"== {os.path.basename(mpath)} done: ok={ok} skip={skip} nomel={nomel} (of {n})")
        grand["ok"] += ok
        grand["skip"] += skip
        grand["nomel"] += nomel

    print(f"\nALL DONE: ok={grand['ok']} skip={grand['skip']} nomel={grand['nomel']}")
    if grand["nomel"]:
        print("WARNING: some targets had no sibling .npy — run the mel dump first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
