"""
Generate BigVGAN fine_tuning filelists for the P2H GTA targets.

BigVGAN's get_dataset_filelist builds each wav path as
    os.path.join(input_wavs_dir, <entry> + ".wav")
When <entry> is an ABSOLUTE path, os.path.join ignores input_wavs_dir and returns
the absolute path — which lets BigVGAN read our wavs from their scattered, co-located
locations (no flat input_wavs_dir needed).

So each filelist line is the absolute stem of a resampled target WITH the "-22k"
suffix and WITHOUT extension, e.g.:
    /data/.../segment65_Healthy-22k
BigVGAN then reads  <stem>-22k.wav  (target) and, via the patched mel loader,
<stem>.npy (predicted mel).

Writes train.txt / val.txt into the output dir (default: alongside the manifests).

    python scripts/make_bigvgan_filelists.py --out-dir /data/.../bigvganGTA/filelists
"""

import os
import argparse

DEFAULT_DATA = "/data/fs201163/eo49197/VoiceConversion-fwf/dub-autoalign-healthyYT"
SPLITS = {"trainPATH-HE": "train.txt", "validPATH-HE": "val.txt"}


def targets(manifest_path):
    out = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if i == 0 or not line.strip():
                continue
            out.append(line.rstrip("\n").split("\t")[2])  # file_path (healthy target)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for split, fname in SPLITS.items():
        tgts = targets(os.path.join(args.data, f"{split}.tsv"))
        out_path = os.path.join(args.out_dir, fname)
        n_ok = 0
        with open(out_path, "w") as w:
            for t in tgts:
                stem = os.path.splitext(t)[0] + "-22k"   # absolute "<stem>-22k"
                wav = stem + ".wav"
                if not os.path.exists(wav):
                    raise FileNotFoundError(f"missing resampled wav: {wav}")
                w.write(stem + "\n")
                n_ok += 1
        print(f"{fname}: wrote {n_ok} entries -> {out_path}")


if __name__ == "__main__":
    main()
