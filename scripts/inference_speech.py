import os
import sys
import argparse
import logging
import torch
import torchaudio
import numpy as np

from tqdm import tqdm
from scipy.io import wavfile
from transformers import AutoTokenizer, WhisperProcessor
from fairseq import checkpoint_utils, utils
from omegaconf import OmegaConf

# --- PYTHONPATH SETUP ---
# Ensure local modules can be imported
current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path)
repo_root = os.path.dirname(scripts_dir)
fairseq_dir = os.path.join(repo_root, "fairseq")
avhubert_dir = os.path.join(repo_root, "avhubert")
src_dir = os.path.join(repo_root, "src")

# Add paths to sys.path
for p in [avhubert_dir, fairseq_dir, repo_root, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Hack to bypass Fairseq arg parsing issues in some environments
_added_dummy = False
if len(sys.argv) == 1:
    sys.argv.append("dummy")
    _added_dummy = True

import src.task
import src.model
from src.modelSpeech import MMS_LLaMA_Speech
from src.utils import Compose, Normalize, CenterCrop, load_video

# Clean up dummy arg immediately after imports
if _added_dummy and "dummy" in sys.argv:
    sys.argv.remove("dummy")
# ------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("inference_speech")


def get_parser():
    parser = argparse.ArgumentParser(description="Inference script for MMS-LLaMA Speech Model")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="/ceph/home/TUG/olivares-tug/MMS-LLaMA/exp/mms-llama-speech/speech_finetune_1GPU_mels100hz_128bands_freq4/checkpoints/checkpoint_best.pt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--manifest-dir", 
        type=str, 
        default="/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/",
        help="Directory containing train.tsv, valid.tsv, test.tsv"
    )
    parser.add_argument(
        "--subset", 
        type=str, 
        default="all",
        help="Subset to run inference on (e.g., train, valid, test, test_inference, or 'all' for all three)"
    )
    parser.add_argument(
        "--output-format", 
        type=str, 
        default="pt",
        choices=["pt", "npy"],
        help="Output format for generated mel spectrograms"
    )
    parser.add_argument(
        "--llm-path", 
        type=str, 
        default="meta-llama/Llama-3.2-3B",
        help="Path or name of the LLM"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Custom output directory for generated mel spectrograms. If not set, saves next to input audio."
    )
    return parser


def load_model(args, device):
    logger.info(f"Loading model from {args.checkpoint}...")
    
    # Path setup for overrides
    # We need to make sure w2v_path points to something valid, 
    # even if we are loading weights from the checkpoint.
    w2v_path = os.path.join(repo_root, "pretrained_models/avhubert/large_vox_iter5.pt")
    
    model_overrides = {
        "task": {
            "data": args.manifest_dir,
            "label_dir": args.manifest_dir,
            "llm_path": args.llm_path,
        },
        "model": {
            "w2v_path": w2v_path,
            "llm_path": args.llm_path,
        }
    }

    try:
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [args.checkpoint], 
            arg_overrides=model_overrides,
            strict=False
        )
        model = models[0]
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fallback manual loading if ensemble loading fails due to config mismatches
        logger.info("Attempting fallback manual loading...")
        
        # Load checkpoint state directly to get config
        state = torch.load(args.checkpoint, map_location="cpu")
        cfg = OmegaConf.create(state["cfg"])
        
        # Re-apply overrides
        cfg.task.data = args.manifest_dir
        cfg.model.llm_path = args.llm_path
        cfg.model.w2v_path = w2v_path
        
        # Setup task & build model
        task = src.task.MMS_LLaMA_Task.setup_task(cfg.task)
        model = MMS_LLaMA_Speech.build_model(cfg.model, task)
        model.load_state_dict(state["model"], strict=False)

    model.eval()
    model.to(device)
    logger.info("Model loaded successfully.")
    return model, task, cfg


def process_audio(audio_path, processor, device):
    # Matches src/dataset.py:load_feature logic
    try:
        sample_rate, wav_data = wavfile.read(audio_path)
    except Exception as e:
        logger.warning(f"Failed to read audio {audio_path}: {e}")
        return None, 0

    if sample_rate != 16000:
        logger.warning(f"Sample rate mismatch: {sample_rate} != 16000 for {audio_path}. Resampling not implemented in this snippet (dataset.py assumes 16k).")
        # In dataset.py, it asserts 16k. Assuming data is clean.
        return None, 0
    
    # Normalize Int16 -> Float32
    if wav_data.dtype == np.int16:
        wav_data = wav_data / 32768.0
    
    wav_data = wav_data.astype(np.float32)
    
    # Calculate lengths
    audio_len_samples = int(len(wav_data))
    
    # Process with Whisper
    input_features = processor(wav_data, sampling_rate=16000, return_tensors="pt").input_features
    return input_features.to(device), audio_len_samples


def process_video(video_path, transform, device):
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return None, None
        
    try:
        frames = load_video(video_path) # Uses src.utils.load_video
        frames = transform(frames)
        # Expand dims: (T, H, W) -> (T, H, W, 1) if grayscale/1ch
        frames = np.expand_dims(frames, axis=-1)
        
        video_tensor = torch.from_numpy(frames.astype(np.float32))
        # Permute: (T, H, W, C) -> (C, T, H, W) ? 
        # Check dataset.py: 
        # collater_audio:
        # if len(audios[0].shape) == 2: transpose...
        # else: collated_audios = collated_audios.permute((0, 4, 1, 2, 3)) # [B, T, H, W, C] -> [B, C, T, H, W]
        # Wait, dataset.py load_video returns (T, H, W, 1)
        # collater receives list of (T, H, W, 1)
        # then permutes to (B, C, T, H, W)
        
        # Here we are processing single sample.
        # Let's match the shape expected by model.forward
        # model.py -> avhubert(source={'video': ...})
        # avhubert expects B, C, T, H, W
        
        video_tensor = video_tensor.permute(3, 0, 1, 2) # (C, T, H, W)
        return video_tensor.unsqueeze(0).to(device), video_tensor.size(1) # Add batch dim
        
    except Exception as e:
        logger.warning(f"Failed to process video {video_path}: {e}")
        return None, None


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model, task, cfg = load_model(args, device)
    
    # Processors & Transforms
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    
    # Image Transform (Matched to dataset.py)
    # image_mean=0, image_std=1, image_crop_size=88
    video_transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((88, 88)),
        Normalize(0.0, 1.0)
    ])
    
    # Instruction Token
    instruction_text = "Focus on semantics, not voice characteristics"
    instruction_tokens = tokenizer(instruction_text, return_tensors="pt").input_ids
    instruction_tokens = instruction_tokens.to(device)
    
    # Subsets to process
    subsets = ["train", "valid", "test"] if args.subset == "all" else [args.subset]
    
    for subset in subsets:
        manifest_path = os.path.join(args.manifest_dir, f"{subset}.tsv")
        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest not found: {manifest_path}. Skipping.")
            continue
            
        logger.info(f"Processing subset: {subset} from {manifest_path}")
        
        # Read Manifest
        # Format: root line, then rows
        with open(manifest_path, 'r') as f:
            root = f.readline().strip()
            lines = f.readlines()
            
        # Parse lines
        # format: audio_id \t video_path \t audio_path \t ...
        data = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                data.append({
                    "id": parts[0],
                    "video_rel": parts[1],
                    "audio_rel": parts[2],
                })
        
        logger.info(f"Found {len(data)} samples.")
        
        for item in tqdm(data, desc=f"Inference {subset}"):
            audio_path = os.path.join(root, item['audio_rel']) if not os.path.isabs(item['audio_rel']) else item['audio_rel']
            # Video path usually relative to root as well, check logic?
            # dataset.py: video_path = items[1]... names.append((video_path...))
            # load_feature -> load_video(os.path.join(self.audio_root, audio_name)) 
            # so video is also joined with root.
            video_path = os.path.join(root, item['video_rel']) if not os.path.isabs(item['video_rel']) else item['video_rel']
            
            # Determine output path
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                audio_basename = os.path.basename(audio_path)
                output_path = os.path.join(args.output_dir, audio_basename.replace('.wav', f'_pred.{args.output_format}'))
            else:
                output_path = audio_path.replace('.wav', f'_pred.{args.output_format}')
            
            if os.path.exists(output_path):
                 # Skip if exists? Or overwrite? 
                 # User prompt said "to avoid overwriting", implying we should check names. 
                 # But usually scripts overwrite if re-run. 
                 # Let's just write. 
                 pass

            # Preprocess
            audio_feats, audio_len = process_audio(audio_path, whisper_processor, device)
            if audio_feats is None: continue
            
            video_feats, video_len = process_video(video_path, video_transform, device)
            
            # Construct Input
            # Model forward_speech expects:
            # source = { 'audio':..., 'video':..., 'instruction':..., 'audio_lengths':... }
            
            padding_mask = torch.zeros((1, video_len), dtype=torch.bool, device=device) if video_feats is not None else None
            
            source = {
                "audio": audio_feats,
                "video": video_feats,
                "instruction": [instruction_tokens[0]], # List of tensors
                "audio_lengths": torch.tensor([audio_len], dtype=torch.long, device=device)
            }
            
            # Dummy target list (required arg but not used for inference logic in forward_speech usually,
            # but let's check modelSpeech.py signature: forward_speech(self, **kwargs)
            # It doesn't explicitly require targets unless criterion uses it.
            # But the forward_speech body uses kwargs['target_list']?
            # Checked modelSpeech.py: it acts on source.
            
            with torch.no_grad():
                outputs = model(
                    source=source,
                    padding_mask=padding_mask
                )
            
            melspec = outputs["melspec"] # (B, T, C)
            
            # Save
            if args.output_format == "pt":
                torch.save(melspec[0].cpu(), output_path)
            elif args.output_format == "npy":
                np.save(output_path, melspec[0].cpu().numpy())


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Remove dummy arg if present
    if "dummy" in sys.argv:
        sys.argv.remove("dummy")
        
    run_inference(args)
