from audio_separator.separator import Separator
import os
import shutil

MANIFEST = "/ceph/home/TUG/olivares-tug/MMS-LLaMA/deNoisingOutputs/noisyAudios.tsv"  # <-- paste your manifest path here

OUTPUT_DIR = "/ceph/home/TUG/olivares-tug/MMS-LLaMA/deNoisingOutputs"

MODELS = [
    "vocals_mel_band_roformer.ckpt"
]

def get_model_tag(model_filename):
    name = os.path.splitext(model_filename)[0]
    return name

def read_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths

def process_with_model(audio_paths, model_filename, output_dir):
    tag = get_model_tag(model_filename)
    model_output_dir = os.path.join(output_dir, tag)
    os.makedirs(model_output_dir, exist_ok=True)

    separator = Separator(output_dir=model_output_dir)
    separator.load_model(model_filename=model_filename)

    for audio_path in audio_paths:
        print(f"[{tag}] Processing: {audio_path}")
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_files = separator.separate(audio_path)

        print(f"  -> Output files: {output_files}")

        # Always copy the original for reference
        orig_dest = os.path.join(model_output_dir, f"{base}__original.wav")
        shutil.copy2(audio_path, orig_dest)
        print(f"  -> Original copied: {orig_dest}")

        # The "clean" stem is the non-noise side:
        # mel_band_roformer produces "(other)" = clean speech, "(crowd)" = noise
        # UVR-MDX-NET_Crowd produces "(No Crowd)" = clean speech, "(Crowd)" = noise
        CLEAN_KEYWORDS = ["No Crowd", "no crowd", "(other)", "Vocals", "vocals"]
        clean_file = next((f for f in output_files if any(k in f for k in CLEAN_KEYWORDS)), None)
        if clean_file is None and output_files:
            clean_file = output_files[0]
            print(f"  -> Warning: could not identify clean stem, using: {clean_file}")

        if clean_file and os.path.exists(clean_file):
            dest = os.path.join(model_output_dir, f"{base}__clean.wav")
            shutil.move(clean_file, dest)
            print(f"  -> Clean stem saved: {dest}")

        # Remove remaining output files (noise stems)
        for f in output_files:
            if f != clean_file and os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    audio_paths = read_manifest(MANIFEST)
    print(f"Loaded {len(audio_paths)} files from manifest.")

    for model in MODELS:
        print(f"\n=== Running model: {model} ===")
        process_with_model(audio_paths, model, OUTPUT_DIR)

    print("\nDone.")
