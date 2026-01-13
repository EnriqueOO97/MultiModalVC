import os
import sys
import torch
import torchaudio
import numpy as np
from fairseq import checkpoint_utils
from transformers import AutoTokenizer, WhisperProcessor

# ==========================================
#           CRITICAL IMPORTS FIX
# ==========================================
# This is required so that 'import src.task' works correctly
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
repo_root = os.path.dirname(src_dir)

if repo_root not in sys.path:
    sys.path.append(repo_root)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Hack to bypass the 'if len(sys.argv) == 1' check in hubert.py
sys.argv.append("dummy_arg")

import src.task   
import src.model  
from src.utils import Compose, Normalize, CenterCrop, load_video

# Clean up
sys.argv.pop()
# ==========================================


# ==========================================
#           CONFIGURATION SECTION
# ==========================================

# Path to your trained model checkpoint (.pt file)
CHECKPOINT_PATH = "/ceph/home/TUG/olivares-tug/MMS-LLaMA/pretrained_models/mms_llama/1759h/ckpt-1759h.pt"

# Path to the input audio file (.wav)
AUDIO_PATH = "/ceph/home/TUG/olivares-tug/datasets/lrs3/lrs3_video_seg24s/test/0Fi83BHQsMA/00002.wav"

# Path to the input video file (.mp4). Set to None if audio-only.
# NOTE: This video must be the pre-processed cropped mouth region.
VIDEO_PATH = "/ceph/home/TUG/olivares-tug/datasets/lrs3/lrs3_video_seg24s/test/0Fi83BHQsMA/00002.mp4" 

# Path or HuggingFace name for the LLM (must match what was used in training)
LLM_PATH = "meta-llama/Llama-3.2-3B"

# Device to run on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Crop size (default is 88 as per dataset.py)
IMAGE_CROP_SIZE = 88

# ==========================================

def process_audio(audio_path, processor):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    wav_numpy = wav.squeeze().numpy()
    input_features = processor(wav_numpy, sampling_rate=16000, return_tensors="pt").input_features
    return input_features

def process_video(video_path, transform):
    if video_path is None:
        return None
    frames = load_video(video_path)
    frames = transform(frames)
    frames = np.expand_dims(frames, axis=-1)
    video_tensor = torch.from_numpy(frames.astype(np.float32))
    video_tensor = video_tensor.permute(3, 0, 1, 2)
    return video_tensor.unsqueeze(0)

def main():
    print(f"Using device: {DEVICE}")

    # --- 1. Load Model ---
    print(f"Loading model from {CHECKPOINT_PATH}...")
    
    model_overrides = {
        "task": {"llm_path": LLM_PATH},
        "model": {"llm_path": LLM_PATH}
    }
    
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [CHECKPOINT_PATH], 
        arg_overrides=model_overrides,
        strict=False
    )
    model = models[0]
    model.eval()
    model.to(DEVICE)
    
    # --- 2. Prepare Processors ---
    print("Initializing processors...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    
    video_transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)),
        Normalize(0.0, 1.0)
    ])

    # --- 3. Process Inputs ---
    print(f"Processing {AUDIO_PATH}...")
    audio_features = process_audio(AUDIO_PATH, whisper_processor).to(DEVICE)
    video_features = process_video(VIDEO_PATH, video_transform)
    
    padding_mask = None
    if video_features is not None:
        video_features = video_features.to(DEVICE)
        T = video_features.shape[2]
        padding_mask = torch.zeros((1, T), dtype=torch.bool).to(DEVICE)

    instruction_text = ""
    print(f"Instruction: {instruction_text}")
    #instruction_text = ""
    instruction_tokens = tokenizer(instruction_text, return_tensors="pt").input_ids
    instruction_tokens = instruction_tokens.to(DEVICE)

    # --- 4. Construct Batch ---
    sample = {
        "net_input": {
            "source": {
                "audio": audio_features,
                "video": video_features,
                "instruction": [instruction_tokens[0]]
            },
            "padding_mask": padding_mask
        }
    }

    # --- 5. Generate ---
    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            **sample["net_input"],
            num_beams=5,
            max_length=200
        )
        
    # --- 6. Decode ---
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n" + "="*30)
    print(f"RESULT: {generated_text}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()