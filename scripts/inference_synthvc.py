"""
Inference script for MMS_Speech_NoLLM_E2E_SynthVC (voice conversion).

For each manifest entry, runs all 6 synthetic audio variants through the model
using the canonical video + speaker embedding, producing one waveform per variant.

Output naming: pred_SourceSyn{1-6}_{ID}.wav
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np

from tqdm import tqdm
from scipy.io import wavfile
from transformers import WhisperProcessor
from fairseq import checkpoint_utils
from omegaconf import OmegaConf

# --- PYTHONPATH SETUP ---
current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path)
repo_root = os.path.dirname(scripts_dir)
fairseq_dir = os.path.join(repo_root, "fairseq")
avhubert_dir = os.path.join(repo_root, "avhubert")
src_dir = os.path.join(repo_root, "src")

for p in [avhubert_dir, fairseq_dir, repo_root, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Bypass Fairseq arg parsing issues when running directly
_added_dummy = False
if len(sys.argv) == 1:
    sys.argv.append("dummy")
    _added_dummy = True

# Fairseq registration — order matters: base first, then SynthVC variants
import src.task                          # registers base task
import src.model                         # registers base models
import src.task_synthvc                  # registers MMS_LLaMA_training_synthvc
import src.modelSpeechNoLLM_E2E_SynthVC  # registers MMS_Speech_NoLLM_E2E_SynthVC
import src.criterionSpeechE2E_SynthVC    # registers e2e_gan_loss_synthvc

from src.utils import Compose, Normalize, CenterCrop, load_video

if _added_dummy and "dummy" in sys.argv:
    sys.argv.remove("dummy")
# ------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("inference_synthvc")


def get_parser():
    parser = argparse.ArgumentParser(
        description="SynthVC inference: canonical video + synthetic audio + spk_emb → waveform"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/gpfs/data/fs72969/enriqueoo97/exp/mms-speech-NoLLM-E2E-SynthVC/"
                "synthvc_disc_bz5_Mel1_Conv1/checkpoints/checkpoint_best.pt",
        help="Path to the SynthVC model checkpoint",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=os.path.join(repo_root, "manifest/germanManifest/test_inferenceAugmented.tsv"),
        help="Path to the augmented inference manifest (TSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(repo_root, "outputsInference"),
        help="Directory for output WAV files",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Vocoder output sample rate (16000 Hz for this codebase)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)",
    )
    return parser


def load_model(checkpoint_path, manifest_dir, device):
    """Load the SynthVC model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}...")

    w2v_path = os.path.join(repo_root, "pretrained_models/avhubert/large_vox_iter5.pt")

    model_overrides = {
        "task": {
            "data": manifest_dir,
            "label_dir": manifest_dir,
        },
        "model": {
            "w2v_path": w2v_path,
            # The SynthVC checkpoint already contains all weights — skip re-loading
            # stage1 / vocoder from their original cluster paths.
            "stage1_checkpoint": "???",
            "vocoder_checkpoint": "???",
        },
    }

    try:
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path],
            arg_overrides=model_overrides,
            strict=False,
        )
        model = models[0]
    except Exception as e:
        logger.warning(f"Ensemble loading failed: {e}")
        logger.info("Attempting fallback manual loading...")

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(state["cfg"])

        cfg.task.data = manifest_dir
        cfg.task.label_dir = manifest_dir
        cfg.model.w2v_path = w2v_path
        cfg.model.stage1_checkpoint = "???"
        cfg.model.vocoder_checkpoint = "???"

        from src.task_synthvc import MMS_LLaMA_TrainingSynthVCTask
        task = MMS_LLaMA_TrainingSynthVCTask.setup_task(cfg.task)

        from src.modelSpeechNoLLM_E2E_SynthVC import MMS_Speech_NoLLM_E2E_SynthVC
        model = MMS_Speech_NoLLM_E2E_SynthVC.build_model(cfg.model, task)
        model.load_state_dict(state["model"], strict=False)

    model.eval()
    model.to(device)
    logger.info("Model loaded and set to eval mode.")
    return model


def process_audio(audio_path, processor, device):
    """Load a WAV file and convert to Whisper input features."""
    sample_rate, wav_data = wavfile.read(audio_path)
    assert sample_rate == 16000, f"Expected 16kHz, got {sample_rate} for {audio_path}"
    if wav_data.dtype == np.int16:
        wav_data = wav_data / 32768.0
    wav_data = wav_data.astype(np.float32)
    audio_len = int(len(wav_data))
    feats = processor(wav_data, sampling_rate=16000, return_tensors="pt").input_features
    return feats.to(device), audio_len  # (1, 80, 3000), int


def process_video(video_path, transform, device):
    """Load and preprocess video, returning tensor (1, 1, T, H, W) and video length T."""
    frames = load_video(video_path)
    frames = transform(frames)
    frames = np.expand_dims(frames, axis=-1)                          # (T, H, W, 1)
    video_tensor = torch.from_numpy(frames.astype(np.float32))
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)     # (1, 1, T, H, W)
    return video_tensor.to(device), video_tensor.size(2)


def run_inference(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_dir = os.path.dirname(args.manifest)

    model = load_model(args.checkpoint, manifest_dir, device)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

    # Image transform — values from mms-speech-nollm-e2e-synthvc.yaml
    video_transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((88, 88)),
        Normalize(0.421, 0.165),
    ])

    # Parse manifest
    with open(args.manifest, "r") as f:
        _ = f.readline()  # root line (ignored; paths in manifest are absolute)
        lines = f.readlines()

    logger.info(f"Found {len(lines)} entries in manifest → expect {len(lines) * 6} output WAVs")

    total_saved = 0
    for line in tqdm(lines, desc="Entries"):
        parts = line.strip().split("\t")
        video_path   = parts[1]
        canon_path   = parts[2]
        spk_emb_path = parts[3]
        synth_paths  = parts[4:10]   # 6 synthetic audio paths
        sample_id    = os.path.splitext(os.path.basename(canon_path))[0]

        # Load video once per entry (shared across all 6 synthetic variants)
        try:
            video_tensor, video_len = process_video(video_path, video_transform, device)
        except Exception as e:
            logger.warning(f"Skipping {sample_id}: video load failed — {e}")
            continue

        padding_mask = torch.zeros((1, video_len), dtype=torch.bool, device=device)

        # Load speaker embedding once per entry
        try:
            spk_emb = torch.load(spk_emb_path, map_location="cpu", weights_only=True)
            if spk_emb.dim() > 1:
                spk_emb = spk_emb.squeeze()   # → (512,)
            spk_emb = spk_emb.unsqueeze(0).to(device)  # → (1, 512)
        except Exception as e:
            logger.warning(f"Skipping {sample_id}: spk_emb load failed — {e}")
            continue

        # Run all 6 synthetic variants
        for syn_idx, synth_path in enumerate(synth_paths, start=1):
            out_name = f"pred_SourceSyn{syn_idx}_{sample_id}.wav"
            out_path = os.path.join(args.output_dir, out_name)

            try:
                whisper_feats, synth_len = process_audio(synth_path, whisper_processor, device)
            except Exception as e:
                logger.warning(f"Skipping {out_name}: synth audio load failed — {e}")
                continue

            source = {
                "audio":          whisper_feats,                                   # (1, 80, 3000)
                "video":          video_tensor,                                    # (1, 1, T, H, W)
                "synth_audio":    whisper_feats,                                   # drives Whisper encoding
                "audio_lengths":  torch.tensor([synth_len], dtype=torch.long,
                                               device=device),                     # drives output duration
                "spk_embeddings": spk_emb,                                        # (1, 512)
            }

            try:
                with torch.no_grad():
                    output = model(source=source, padding_mask=padding_mask)
                waveform = output["waveform"]  # (1, 1, T_audio)
            except Exception as e:
                logger.warning(f"Skipping {out_name}: model forward failed — {e}")
                continue

            wav_np = waveform[0, 0].cpu().float().numpy()
            wav_np = np.clip(wav_np, -1.0, 1.0)
            wav_int16 = (wav_np * 32767).astype(np.int16)
            wavfile.write(out_path, args.sample_rate, wav_int16)
            total_saved += 1
            logger.debug(f"Saved {out_name} | samples={len(wav_int16)}")

    logger.info(f"Done. Saved {total_saved} WAV files to {args.output_dir}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run_inference(args)
