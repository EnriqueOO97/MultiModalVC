"""
Inference script for MMS_Speech_NoLLM model with modality ablation support.

Modes:
  av         - Use both audio and video (standard inference)
  video_only - Mask audio with learned audio_mask_emb (video-only ablation)
  audio_only - Mask video with learned video_mask_emb (audio-only ablation)

Output naming:
  av:         <basename>_pred.pt
  video_only: <basename>_pred_videoOnly.pt
  audio_only: <basename>_pred_audioOnly.pt
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

# Hack to bypass Fairseq arg parsing issues
_added_dummy = False
if len(sys.argv) == 1:
    sys.argv.append("dummy")
    _added_dummy = True

import src.task
import src.model
from src.modelSpeechNoLLM import MMS_Speech_NoLLM
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
logger = logging.getLogger("inference_speech_nollm")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Inference script for MMS_Speech_NoLLM with modality ablation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/ceph/home/TUG/olivares-tug/MMS-LLaMA/exp/mms-speech-NoLLM/speech_mels100hz_128bands_bz20_48G_freq4_wrmp2000_mask_modalitydrop_norm_fiftyfifty/checkpoints/checkpoint_best.pt",
        help="Path to the NoLLM model checkpoint",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/test_inference.tsv",
        help="Path to the TSV manifest file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="av",
        choices=["av", "video_only", "audio_only", "all"],
        help="Inference mode: 'av', 'video_only', 'audio_only', or 'all' to run all three",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/ceph/home/TUG/olivares-tug/MMS-LLaMA/ablation_test_output",
        help="Output directory for generated mel spectrograms",
    )
    return parser


def load_model(checkpoint_path, manifest_dir, device):
    """Load the NoLLM model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}...")

    w2v_path = os.path.join(repo_root, "pretrained_models/avhubert/large_vox_iter5.pt")

    model_overrides = {
        "task": {
            "data": manifest_dir,
            "label_dir": manifest_dir,
        },
        "model": {
            "w2v_path": w2v_path,
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
        logger.error(f"Ensemble loading failed: {e}")
        logger.info("Attempting fallback manual loading...")

        state = torch.load(checkpoint_path, map_location="cpu")
        cfg = OmegaConf.create(state["cfg"])

        cfg.task.data = manifest_dir
        cfg.model.w2v_path = w2v_path

        task = src.task.MMS_LLaMA_Task.setup_task(cfg.task)
        model = MMS_Speech_NoLLM.build_model(cfg.model, task)
        model.load_state_dict(state["model"], strict=False)

    model.eval()
    model.to(device)

    # Log mask embedding stats
    logger.info(
        f"audio_mask_emb: shape={model.audio_mask_emb.shape}, "
        f"min={model.audio_mask_emb.min().item():.4f}, "
        f"max={model.audio_mask_emb.max().item():.4f}"
    )
    logger.info(
        f"video_mask_emb: shape={model.video_mask_emb.shape}, "
        f"min={model.video_mask_emb.min().item():.4f}, "
        f"max={model.video_mask_emb.max().item():.4f}"
    )

    return model, task, cfg


def process_audio(audio_path, processor, device):
    """Load and preprocess audio for Whisper."""
    try:
        sample_rate, wav_data = wavfile.read(audio_path)
    except Exception as e:
        logger.warning(f"Failed to read audio {audio_path}: {e}")
        return None, 0

    if sample_rate != 16000:
        logger.warning(f"Sample rate {sample_rate} != 16000 for {audio_path}")
        return None, 0

    if wav_data.dtype == np.int16:
        wav_data = wav_data / 32768.0

    wav_data = wav_data.astype(np.float32)
    audio_len_samples = int(len(wav_data))

    input_features = processor(
        wav_data, sampling_rate=16000, return_tensors="pt"
    ).input_features
    return input_features.to(device), audio_len_samples


def process_video(video_path, transform, device):
    """Load and preprocess video."""
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return None, None

    try:
        frames = load_video(video_path)
        frames = transform(frames)
        frames = np.expand_dims(frames, axis=-1)

        video_tensor = torch.from_numpy(frames.astype(np.float32))
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        return video_tensor.unsqueeze(0).to(device), video_tensor.size(1)

    except Exception as e:
        logger.warning(f"Failed to process video {video_path}: {e}")
        return None, None


def apply_modality_mask(model, whisper_enc_out, avhubert_output, mode):
    """
    Apply modality masking using the learned mask embeddings.
    This replicates the exact same masking logic used during training.
    """
    if mode == "video_only":
        # Replace all audio features with the learned audio mask
        whisper_enc_out = (
            model.audio_mask_emb.unsqueeze(0).unsqueeze(0).expand_as(whisper_enc_out)
        )
        logger.debug("Applied audio_mask_emb (video_only mode)")
    elif mode == "audio_only":
        # Replace all video features with the learned video mask
        avhubert_output["encoder_out"] = (
            model.video_mask_emb.unsqueeze(0)
            .unsqueeze(0)
            .expand_as(avhubert_output["encoder_out"])
        )
        logger.debug("Applied video_mask_emb (audio_only mode)")
    # mode == 'av': no masking
    return whisper_enc_out, avhubert_output


def forward_with_ablation(model, source, padding_mask, mode):
    """
    Run the model's forward pass with manual modality masking.
    This replicates forward_speech but allows masking at inference time
    (the original forward_speech only masks during training).
    """
    import torch.nn.functional as F

    # 1. Feature extraction (no grad, same as forward_speech)
    with torch.no_grad():
        whisper_enc_out = model.whisper(source)

        avhubert_source = {"audio": None, "video": source["video"]}
        avhubert_output = model.avhubert(
            source=avhubert_source, padding_mask=padding_mask
        )
        avhubert_output["encoder_out"] = avhubert_output["encoder_out"].transpose(0, 1)

    video_lengths = torch.sum(~avhubert_output["padding_mask"], dim=1).tolist()
    max_vid_len = max(video_lengths)

    # 2. Speech rate predictor
    if model.cfg.use_sr_predictor:
        len_queries, resized_len_list = model.query_length_calculation(
            whisper_enc_out, video_lengths, max_vid_len
        )
    else:
        len_queries = [
            max(int(vl / 25 * model.cfg.queries_per_sec), model.cfg.queries_per_sec)
            for vl in video_lengths
        ]

    # 3. Feature processing
    whisper_enc_out = model.afeat_1d_conv(
        whisper_enc_out.transpose(1, 2)
    ).transpose(1, 2)

    if model.cfg.use_qformer:
        pm = (~avhubert_output["padding_mask"]).long()
        len_feat = video_lengths
    else:
        pm = avhubert_output["padding_mask"][:, 1::2]
        pm = (~pm).long()
        len_feat = torch.sum(pm, dim=1).tolist()
        avhubert_output["encoder_out"] = model.vfeat_1d_conv(
            avhubert_output["encoder_out"].transpose(1, 2)
        ).transpose(1, 2)

    B_dim, T_v, _ = avhubert_output["encoder_out"].size()
    whisper_enc_out = whisper_enc_out[:, :T_v, :]

    # 4. Apply modality masking (the key difference from training forward_speech)
    whisper_enc_out, avhubert_output = apply_modality_mask(
        model, whisper_enc_out, avhubert_output, mode
    )

    # 5. Fusion
    if model.modality_fuse == "concat":
        av_feat = torch.cat(
            [whisper_enc_out, avhubert_output["encoder_out"]], dim=2
        )
    elif model.modality_fuse == "add":
        av_feat = whisper_enc_out + avhubert_output["encoder_out"]
    elif model.modality_fuse == "cross-att":
        av_feat = model.multimodal_attention_layer(
            audio_feature=whisper_enc_out,
            visual_feature=avhubert_output["encoder_out"],
        )
    else:
        raise ValueError(f"Unknown modality fusion: {model.modality_fuse}")

    # 6. Q-Former compression
    if model.cfg.use_qformer:
        query_output = model.compression_using_qformer(
            len_queries, resized_len_list, len_feat, av_feat
        )
        query_output = model.avfeat_to_llm(query_output)
        queries = query_output
        query_lengths = len_queries
    else:
        queries = model.avfeat_to_llm(av_feat)
        query_lengths = len_feat

    # 7. Speech head (proj1 -> interpolation -> proj2 -> conformer -> mel_head)
    B = queries.size(0)
    av_lengths = query_lengths

    max_av_len = max(av_lengths)
    av_hidden_padded = queries.new_zeros((B, max_av_len, queries.size(-1)))
    for i in range(B):
        length = av_lengths[i]
        av_hidden_padded[i, :length, :] = queries[i, :length, :]

    x = model.proj1(av_hidden_padded.to(model.proj1.weight.dtype))
    x = model.ln1(x)

    # Interpolation
    audio_lengths = source.get("audio_lengths", None)
    if audio_lengths is None:
        audio = source.get("audio", None)
        if audio is not None:
            if audio.dim() == 2:
                audio_lengths = torch.full(
                    (audio.size(0),), audio.size(1),
                    device=audio.device, dtype=torch.long,
                )
            elif audio.dim() == 3 and audio.size(1) == 1:
                audio_lengths = torch.full(
                    (audio.size(0),), audio.size(2),
                    device=audio.device, dtype=torch.long,
                )

    if audio_lengths is None:
        raise ValueError("Audio lengths required for interpolation.")

    n_fft = 1024
    hop_length = 160
    pad = (n_fft - hop_length) // 2
    audio_lengths = audio_lengths.to(dtype=torch.long)
    target_lengths = (
        torch.div(audio_lengths + 2 * pad - n_fft, hop_length, rounding_mode="floor")
        + 1
    )
    target_lengths = torch.clamp(target_lengths, min=1)
    max_target_len = int(target_lengths.max().item())

    x = x.transpose(1, 2)  # (B, C, T)
    B_interp, C, T_av = x.size()
    x_up = x.new_zeros((B_interp, C, max_target_len))

    for i in range(B_interp):
        actual_av_len = av_lengths[i]
        x_slice = x[i : i + 1, :, :actual_av_len]
        tgt_len = int(target_lengths[i].item())
        x_i = F.interpolate(x_slice, size=tgt_len, mode="linear", align_corners=False)
        x_up[i, :, :tgt_len] = x_i[0]

    x = x_up.transpose(1, 2)  # (B, T, C)

    x = model.proj2(x)
    x = model.ln2(x)

    x = model.conformer(x)
    x = model.ln3(x)

    melspec = model.mel_head(x)

    return {"melspec": melspec}


def get_output_suffix(mode):
    """Return the file suffix for a given mode."""
    suffixes = {
        "av": "_pred.pt",
        "video_only": "_pred_videoOnly.pt",
        "audio_only": "_pred_audioOnly.pt",
    }
    return suffixes[mode]


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    manifest_dir = os.path.dirname(args.manifest)
    model, task, cfg = load_model(args.checkpoint, manifest_dir, device)

    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

    video_transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((88, 88)),
        Normalize(0.0, 1.0),
    ])

    # Read manifest
    with open(args.manifest, "r") as f:
        root = f.readline().strip()
        lines = f.readlines()

    data = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            data.append({
                "id": parts[0],
                "video_rel": parts[1],
                "audio_rel": parts[2],
            })

    logger.info(f"Found {len(data)} samples in {args.manifest}")

    # Determine which modes to run
    modes = ["av", "video_only", "audio_only"] if args.mode == "all" else [args.mode]

    os.makedirs(args.output_dir, exist_ok=True)

    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running inference in mode: {mode}")
        logger.info(f"{'='*60}")

        suffix = get_output_suffix(mode)

        for item in tqdm(data, desc=f"Inference [{mode}]"):
            audio_path = (
                os.path.join(root, item["audio_rel"])
                if not os.path.isabs(item["audio_rel"])
                else item["audio_rel"]
            )
            video_path = (
                os.path.join(root, item["video_rel"])
                if not os.path.isabs(item["video_rel"])
                else item["video_rel"]
            )

            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(args.output_dir, audio_basename + suffix)

            # Preprocess audio
            audio_feats, audio_len = process_audio(audio_path, whisper_processor, device)
            if audio_feats is None:
                logger.warning(f"Skipping {audio_path} (audio failed)")
                continue

            # Preprocess video
            video_feats, video_len = process_video(video_path, video_transform, device)
            if video_feats is None:
                logger.warning(f"Skipping {video_path} (video failed)")
                continue

            # Build input
            padding_mask = torch.zeros(
                (1, video_len), dtype=torch.bool, device=device
            )

            source = {
                "audio": audio_feats,
                "video": video_feats,
                "audio_lengths": torch.tensor(
                    [audio_len], dtype=torch.long, device=device
                ),
            }

            # Run inference with ablation
            with torch.no_grad():
                outputs = forward_with_ablation(model, source, padding_mask, mode)

            melspec = outputs["melspec"]  # (B, T, 128)

            # Save
            torch.save(melspec[0].cpu(), output_path)
            logger.debug(f"Saved: {output_path} | shape: {melspec[0].shape}")

        logger.info(f"Finished mode '{mode}': {len(data)} samples -> {args.output_dir}")

    logger.info("\nAll done!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if "dummy" in sys.argv:
        sys.argv.remove("dummy")

    run_inference(args)
