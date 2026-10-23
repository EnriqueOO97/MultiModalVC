import os
import shutil
import argparse
import tempfile
import traceback
import time
import torch
import librosa
import soundfile as sf
from audio_separator.separator import Separator

# Define paths
MANIFESTS = [
    "/data/fs201163/eo49197/MultiModalVC/manifest/germanManifest/trainMOS.tsv",
    "/data/fs201163/eo49197/MultiModalVC/manifest/germanManifest/validMOS.tsv",
    "/data/fs201163/eo49197/MultiModalVC/manifest/germanManifest/testMOS.tsv",
]
MODEL = "vocals_mel_band_roformer.ckpt"
MOS_THRESHOLD = 3.35


def parse_manifests(manifest_paths, mos_threshold):
    audio_files = []
    for manifest_path in manifest_paths:
        if not os.path.exists(manifest_path):
            print(f"Manifest not found: {manifest_path}")
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().rsplit(None, 1)
                if len(parts) >= 2:
                    filepath = parts[0]
                    try:
                        mos_score = float(parts[1])
                        if mos_score <= mos_threshold:
                            audio_files.append(filepath)
                    except ValueError:
                        pass
    return audio_files


def format_audio(input_path, output_path, target_sr=16000):
    """Takes left channel of stereo stem (avoids phase cancellation), resamples to target_sr mono."""
    wav, sr = sf.read(input_path, always_2d=True)   # shape: (samples, channels)
    wav = wav[:, 0]                                  # take left channel only, no phase mixing
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, wav, target_sr, format='WAV', subtype='PCM_16')


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0,
                        help="Index of this worker (0-based)")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Total number of parallel workers")
    args = parser.parse_args()

    # 1. Gather all files below or equal to threshold
    print("Parsing manifests...")
    target_files = parse_manifests(MANIFESTS, MOS_THRESHOLD)
    print(f"Found {len(target_files)} initial files with MOS <= {MOS_THRESHOLD}.")

    # 2. Filter out already processed files (resume capability)
    files_to_process = []
    for filepath in target_files:
        if not filepath.endswith(".wav"):
            continue

        base_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)

        old_filepath = os.path.join(base_dir, f"{name}_OLD{ext}")
        if not os.path.exists(old_filepath):
            if os.path.exists(filepath):
                files_to_process.append(filepath)

    print(f"After skipping already processed files, {len(files_to_process)} remain to be processed.")

    # 3. Shard: this worker takes every world_size-th file starting at rank
    files_to_process = files_to_process[args.rank::args.world_size]
    print(f"Worker {args.rank}/{args.world_size} will process {len(files_to_process)} files.")

    if not files_to_process:
        print("Nothing to do.")
        return

    # 4. Pin this worker to its CUDA device (rank 0-3 maps to GPU 0-3 on a 4xH100 node)
    torch.cuda.set_device(args.rank)
    print(f"Worker {args.rank} using CUDA device {args.rank} ({torch.cuda.get_device_name(args.rank)})")

    # 5. Setup separator with a per-process temp dir (safe for parallel runs)
    temp_output_dir = tempfile.mkdtemp(prefix="audio_sep_")
    print(f"Using temp dir: {temp_output_dir}")

    separator = Separator(
        output_dir=temp_output_dir,
        output_single_stem="Vocals",
        sample_rate=44100,
        use_soundfile=True,
        # IMPORTANT: batch_size > 1 causes silent failures (returns unprocessed noisy audio).
        mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 1, "overlap": 8, "pitch_shift": 0}
    )

    print(f"Loading model {MODEL}...")
    separator.load_model(model_filename=MODEL)
    print("Model loaded successfully.")

    # 5. Process files
    for idx, filepath in enumerate(files_to_process):
        print(f"\n[{idx+1}/{len(files_to_process)}] Processing: {filepath}")

        base_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        old_filepath = os.path.join(base_dir, f"{name}_OLD{ext}")

        try:
            output_files = separator.separate(filepath)

            if not output_files:
                print(f"  -> Separator returned nothing for {filepath}.")
                continue

            raw_clean_stem = output_files[0]
            if not os.path.isabs(raw_clean_stem):
                raw_clean_stem = os.path.join(temp_output_dir, raw_clean_stem)

            if not os.path.exists(raw_clean_stem):
                print(f"  -> Expected clean stem not found: {raw_clean_stem}")
                continue

            # Write formatted clean file into the SAME directory as the original
            # so the final swap is a same-filesystem os.rename (atomic).
            temp_formatted_path = os.path.join(base_dir, f"{name}_CLEAN_TEMP{ext}")
            format_audio(raw_clean_stem, temp_formatted_path, target_sr=16000)

            # Atomic swap: both renames stay on the same filesystem
            os.rename(filepath, old_filepath)
            print(f"  -> Original backed up to {old_filepath}")
            os.rename(temp_formatted_path, filepath)
            print(f"  -> Cleaned audio placed at {filepath}")

            try:
                os.remove(raw_clean_stem)
            except OSError:
                pass

        except Exception as e:
            print(f"  !! Error processing {filepath}: {e}")
            traceback.print_exc()

    # Final cleanup of temp dir
    shutil.rmtree(temp_output_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"\n✓ Worker {args.rank} finished. Elapsed time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
