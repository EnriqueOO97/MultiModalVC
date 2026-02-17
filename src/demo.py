import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
from fairseq import checkpoint_utils
from transformers import AutoTokenizer, WhisperProcessor

# We assume PYTHONPATH is set correctly by the shell script, 
# so we don't need the sys.path hacks here.

# However, we still need to import the project modules.
# Since this script runs as a module or script inside src/, 
# we need to ensure imports work relative to the root.
try:
    import src.task   
    import src.model  
    from src.utils import Compose, Normalize, CenterCrop, load_video
except ImportError:
    # Fallback if running directly from src/ without PYTHONPATH set
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import src.task   
    import src.model  
    from src.utils import Compose, Normalize, CenterCrop, load_video

def get_parser():
    parser = argparse.ArgumentParser(description="MMS-LLaMA Single File Inference")
    
    # Model Arguments
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--llm-path", type=str, default="meta-llama/Llama-3.2-3B", help="HuggingFace model name or path")
    parser.add_argument("--sr-predictor-path", type=str, default=None, help="Path to speech rate predictor checkpoint (optional)")
    
    # Input Arguments
    parser.add_argument("--audio-path", type=str, required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--video-path", type=str, default=None, help="Path to input video file (.mp4). If None, audio-only.")
    
    # Generation Arguments
    parser.add_argument("--instruction", type=str, default="Recognize this speech in English. Input : ", help="Instruction prompt")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for generation")
    parser.add_argument("--max-length", type=int, default=200, help="Max generation length")
    parser.add_argument("--image-crop-size", type=int, default=96, help="Crop size for video frames")
    
    return parser

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
    parser = get_parser()
    # Fairseq's checkpoint_utils might parse sys.argv, so we parse known args only
    args, unknown = parser.parse_known_args()
    
    # Hack: Fairseq/Hubert expects arguments, so we inject a dummy one if needed
    # to prevent it from crashing when parsing sys.argv
    if "dummy_arg" not in sys.argv:
        sys.argv.append("dummy_arg")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading model from {args.checkpoint_path}...")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_w2v_path = os.path.join(root_dir, "pretrained_models", "avhubert", "large_vox_iter5.pt")

    model_overrides = {
        "task": {"llm_path": args.llm_path},
        "model": {
            "llm_path": args.llm_path,
            "w2v_path": default_w2v_path,
        },
    }
    
    if args.sr_predictor_path:
        model_overrides["model"]["sr_ckpt_path"] = args.sr_predictor_path

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.checkpoint_path], 
        arg_overrides=model_overrides,
        strict=False
    )
    model = models[0]
    model.eval()
    model.to(device)
    
    # --- 2. Prepare Processors ---
    print("Initializing processors...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    
    video_transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((args.image_crop_size, args.image_crop_size)),
        Normalize(0.0, 1.0)
    ])

    # --- 3. Process Inputs ---
    #print(f"Processing {args.audio_path}...")
    audio_features = process_audio(args.audio_path, whisper_processor).to(device)
    
    video_features = process_video(args.video_path, video_transform)

    padding_mask = None
    if video_features is not None:
        video_features = video_features.to(device)
        T = video_features.shape[2]
        padding_mask = torch.zeros((1, T), dtype=torch.bool).to(device)

    instruction_tokens = tokenizer(args.instruction, return_tensors="pt").input_ids
    instruction_tokens = instruction_tokens.to(device)

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
            num_beams=args.beam_size,
            max_length=args.max_length
        )
        
    # --- 6. Decode ---
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n" + "="*30)
    print(f"RESULT: {generated_text}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()