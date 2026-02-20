"""
Quick Vocoder Inference Script for DiVISe
==========================================
Converts mel spectrograms to waveforms using HiFi-GAN vocoder checkpoints.

Usage: python quickInference.py

Configure the paths below before running.
"""

import os
import sys
import shutil
import argparse
import glob
import torch
import torchaudio
from pathlib import Path

# ==============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ==============================================================================

# Path to TSV manifest file containing samples (uses wav column to find _pred.pt files)
TSV_MANIFEST = "/ceph/home/TUG/olivares-tug/DiVISe/samples/test_inference.tsv"

# Suffix for predicted mel files (appended to wav basename, replaces .wav)
PRED_SUFFIX = "_pred.pt"

# Path to HiFi-GAN vocoder checkpoint (.pt file)
CHECKPOINT_PATH = "/ceph/home/TUG/olivares-tug/DiVISe/exp-vocoder/exp7-finetuning-on-german-EMA-bz-64/model-best.pt"

# Output directory for generated waveforms
OUTPUT_DIR = "/ceph/home/TUG/olivares-tug/DiVISe/samples"

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================

# Add custom_hifigan to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "custom_hifigan"))

from hifigan.generator import HifiganGenerator


def load_vocoder(checkpoint_path: str, device: torch.device, use_ema: bool = False) -> HifiganGenerator:
    """Load HiFi-GAN generator from checkpoint."""
    print(f"Loading vocoder from: {checkpoint_path}")
    if use_ema:
        print("  Using EMA generator weights")
    else:
        print("  Using standard (AdamW) generator weights")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if use_ema and "ema_generator" in checkpoint and checkpoint["ema_generator"] is not None:
        # Load EMA weights
        state_dict = checkpoint["ema_generator"]
    elif "generator" in checkpoint:
        # DiVISe format: checkpoint["generator"]["model"]
        state_dict = checkpoint["generator"]["model"]
    else:
        # Direct state dict format
        state_dict = checkpoint
    
    # Remove "module." prefix if present (from DDP training)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Create generator with default parameters for mel spectrogram mode
    generator = HifiganGenerator()
    generator.load_state_dict(state_dict)
    generator = generator.to(device)
    generator.eval()
    generator.remove_weight_norm()
    
    print("Vocoder loaded successfully!")
    return generator


def load_mel(mel_path: str, device: torch.device) -> torch.Tensor:
    """Load mel spectrogram from .pt file."""
    mel = torch.load(mel_path, map_location=device)
    
    # Ensure correct shape: (batch, mel_bins, time)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # Add batch dimension
    
    # Mels are saved as (time, mel_bins) -> transpose to (mel_bins, time)
    # Then add batch dim for (1, mel_bins, time)
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    
    # Always transpose: (time, 128) -> (128, time)
    mel = mel.transpose(0, 1)
    mel = mel.unsqueeze(0)  # Add batch dim
    
    return mel.to(device)


def parse_tsv_manifest(tsv_path: str) -> list:
    """Parse TSV manifest and return list of (pred_mel_path, original_wav_path) tuples."""
    items = []
    
    with open(tsv_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('/'):
            continue  # Skip header or empty lines
        
        parts = line.split('\t')
        if len(parts) >= 3:
            wav_path = parts[2]  # Third column is wav path
            # Construct pred file path: same folder, same basename + _pred.pt
            pred_path = wav_path.replace('.wav', PRED_SUFFIX)
            
            if os.path.exists(pred_path):
                items.append((pred_path, wav_path))
            else:
                print(f"Warning: Pred file not found: {pred_path}")
    
    return items


def parse_folder_input(input_folder: str) -> list:
    """Find all .pt files in folder and corresponding wavs if they exist."""
    items = []
    # Find all .pt files (assuming they are mel predictions)
    # Filter out model checkpoints if they happen to be there (usually start with model-)
    pt_files = glob.glob(os.path.join(input_folder, "*.pt"))
    
    for pt_file in pt_files:
        if "model-" in Path(pt_file).name:
            continue
            
        # Try to find reference wav
        # Assumption: wav has same basename but with .wav extension
        # If the pt file ends with _pred.pt, we might need to strip that to find the wav
        base_name = Path(pt_file).stem
        if base_name.endswith("_pred"):
             base_wav_name = base_name.replace("_pred", "")
        else:
             base_wav_name = base_name
             
        wav_path = os.path.join(input_folder, f"{base_wav_name}.wav")
        
        if not os.path.exists(wav_path):
            # Try exact match just in case
            wav_path = pt_file.replace(".pt", ".wav")
            
        if not os.path.exists(wav_path):
            # No reference found
            wav_path = None
            
        items.append((pt_file, wav_path))
        
    return items


@torch.inference_mode()
def generate_waveform(generator: HifiganGenerator, mel: torch.Tensor) -> torch.Tensor:
    """Generate waveform from mel spectrogram."""
    wav = generator(mel)
    wav = wav.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    return wav


def main():
    parser = argparse.ArgumentParser(description="Quick HiFi-GAN Inference")
    parser.add_argument("--InputFolder", type=str, help="Path to folder containing .pt files to process", default=None)
    parser.add_argument("--use_ema", action="store_true", help="Use EMA generator weights instead of standard weights")
    args = parser.parse_args()

    # Validate paths
    if args.InputFolder:
        if not os.path.exists(args.InputFolder):
            print(f"Error: InputFolder does not exist: {args.InputFolder}")
            return
    elif not os.path.exists(TSV_MANIFEST):
        print(f"Error: TSV_MANIFEST does not exist: {TSV_MANIFEST}")
        print("Please edit the CONFIGURATION section in this script.")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: CHECKPOINT_PATH does not exist: {CHECKPOINT_PATH}")
        print("Please edit the CONFIGURATION section in this script.")
        return
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load vocoder
    generator = load_vocoder(CHECKPOINT_PATH, device, use_ema=args.use_ema)
    
    # Get list of (pred mel, original wav) pairs
    if args.InputFolder:
        print(f"Scanning folder: {args.InputFolder}")
        file_pairs = parse_folder_input(args.InputFolder)
    else:
        print(f"Using manifest: {TSV_MANIFEST}")
        file_pairs = parse_tsv_manifest(TSV_MANIFEST)
    
    if not file_pairs:
        if args.InputFolder:
            print(f"No .pt files found in folder: {args.InputFolder}")
        else:
             print(f"No predicted mel files found from manifest: {TSV_MANIFEST}")
        return
    
    print(f"Found {len(file_pairs)} samples to process")
    
    # Process each sample
    for mel_file, original_wav in file_pairs:
        print(f"Processing: {mel_file}")
        
        # Copy original reference wav
        if original_wav and os.path.exists(original_wav):
            base_name = Path(original_wav).name
            original_output = os.path.join(OUTPUT_DIR, base_name)
            shutil.copy2(original_wav, original_output)
            print(f"  Copied reference: {original_output}")
        else:
            print(f"  Warning: Original wav not found (reference skipped)")
        
        # Load mel
        mel = load_mel(mel_file, device)
        print(f"  Mel shape: {mel.shape}")
        
        # Generate waveform
        wav = generate_waveform(generator, mel)
        print(f"  Generated waveform shape: {wav.shape}")
        
        # Save waveform - use original basename without _pred suffix
        base_name = Path(mel_file).stem.replace("_pred", "")
        output_name = f"{base_name}_lastEMA.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        # Sample rate is 16kHz for DiVISe vocoders
        torchaudio.save(output_path, wav.unsqueeze(0).cpu(), sample_rate=16000)
        print(f"  Saved to: {output_path}")
    
    print(f"\nDone! Generated {len(file_pairs)} waveform(s) in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
