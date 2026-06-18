"""
Inference script for pathological → healthy voice conversion.

For each manifest entry, runs the pathological audio+video through the model
using 3 different speaker embeddings (healthy, synth1, synth2), producing
one waveform per target.

Output naming: {sample_id}_target-{healthy|synth1|synth2}.wav
"""

import os
import sys
import argparse
import logging
import random
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

# Fairseq registration
import src.task
import src.model
import src.task_pathological_finetune
import src.modelSpeechNoLLM_E2E_SynthVC
import src.criterionSpeechE2E_SynthVC

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
logger = logging.getLogger("path-to-health-inference")


# ---------------------------------------------------------------------------
# SELF-CONTAINED INFERENCE FIX (no changes to task_pathological_finetune.py).
#
# The fine-tuned Whisper encoder already lives INSIDE the model checkpoint (367
# whisper.* tensors that overwrite the architecture scaffold when the checkpoint
# loads). The task's build-time helper `_load_finetuned_whisper` would instead
# try to read an EXTERNAL safetensors file whose path is baked into the
# checkpoint cfg — that path is typically unreachable on an inference machine,
# and the helper raises when it's missing. We replace it with a no-op so:
#   * inference never touches the external file, and
#   * Whisper weights come EXCLUSIVELY from the checkpoint (zero stock HF weights;
#     the [whisper-verify] checks in load_model() prove this after load).
# This keeps the fix entirely within this script.
# ---------------------------------------------------------------------------
import src.task_pathological_finetune as _tpf_mod


def _skip_external_whisper(self, model):
    logger.info(
        "[inference] external Whisper swap skipped — Whisper weights come from the model checkpoint"
    )


_tpf_mod.MMS_PathologicalFinetuneTask._load_finetuned_whisper = _skip_external_whisper


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pathological → healthy inference: patho audio+video + target xvector → waveform"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/pathological_finetune_v16_concat_dnsmos01/checkpoints/checkpoint_best.pt",
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/data/fs201163/eo49197/VoiceConversion-fwf/dub-healthyTY/testPATH-HE.tsv",
        help="Path to the pathological test manifest (TSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Path-to-healthy-outputs",
        help="Directory for output WAV files",
    )
    parser.add_argument(
        "--sorted",
        type=int,
        default=None,
        metavar="N",
        help="Process the first N entries (sorted order)",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=None,
        metavar="N",
        help="Process N randomly chosen entries",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Vocoder output sample rate (16000 Hz)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="Split inputs longer than SEC into chunks of SEC seconds. "
             "Each chunk produces its own WAV file with a _partN suffix. "
             "Omit to process each sample whole (default).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)",
    )
    return parser


def _xvector_path(wav_path: str) -> str:
    """Convert ``<stem>.wav`` -> ``<stem>_xvector.pt``."""
    if wav_path.endswith(".wav"):
        return wav_path[:-4] + "_xvector.pt"
    raise ValueError(f"Expected .wav path, got: {wav_path}")


def load_model(checkpoint_path, manifest_dir, device):
    """Load the fine-tuned model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}...")

    w2v_path = os.path.join(repo_root, "pretrained_models/avhubert/large_vox_iter5.pt")

    model_overrides = {
        "task": {
            "data": manifest_dir,
            "label_dir": manifest_dir,
        },
        "model": {
            "w2v_path": w2v_path,
            "stage1_checkpoint": "",
            "vocoder_checkpoint": "",
        },
    }

    # Ensure new config fields exist for older checkpoints
    _raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _ckpt_cfg = _raw.get("cfg", {})
    _model_cfg = _ckpt_cfg.get("model", {}) if isinstance(_ckpt_cfg, dict) else getattr(_ckpt_cfg, "model", {})
    _model_dict = _model_cfg if isinstance(_model_cfg, dict) else OmegaConf.to_container(_model_cfg, resolve=True)
    if "use_cqt" not in _model_dict:
        model_overrides["model"]["use_cqt"] = False
    if "upsampling_method" not in _model_dict:
        model_overrides["model"]["upsampling_method"] = "interpolation"
    _sd = _raw.get("model", {})
    _has_conv1 = any(k.startswith("upsample_conv1.") for k in _sd.keys())
    _has_conv3 = any(k.startswith("upsample_conv3.") for k in _sd.keys())
    _has_conv4 = any(k.startswith("upsample_conv4.") for k in _sd.keys())
    if _has_conv1:
        # 4 layers (2x2x2x4=32x), 3 layers (2x2x4=16x), or 2 layers (2x4=8x).
        # NOTE: conv3's kernel differs between 3-layer (k=8) and 4-layer (k=4),
        # so detecting conv4 is required — assuming 3 whenever conv3 exists would
        # build the wrong conv3 kernel for 4-layer checkpoints and fail to load.
        model_overrides["model"]["transconv_layers"] = 4 if _has_conv4 else (3 if _has_conv3 else 2)
    del _raw, _ckpt_cfg, _model_cfg, _model_dict, _sd

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

        _fb = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(_fb["cfg"])
        cfg.task.data = manifest_dir
        cfg.task.label_dir = manifest_dir
        cfg.model.data = manifest_dir
        cfg.model.w2v_path = w2v_path
        cfg.model.stage1_checkpoint = ""
        cfg.model.vocoder_checkpoint = ""

        from src.task_pathological_finetune import MMS_PathologicalFinetuneTask
        task = MMS_PathologicalFinetuneTask.setup_task(cfg.task)

        from src.modelSpeechNoLLM_E2E_SynthVC import MMS_Speech_NoLLM_E2E_SynthVC
        model = MMS_Speech_NoLLM_E2E_SynthVC.build_model(cfg.model, task)

        model.load_state_dict(_fb["model"], strict=False)
        del _fb

    model.eval()

    # --- Verify whisper weights came FROM the checkpoint (no HF fallback) ---
    _raw_for_check = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _ckpt_sd = _raw_for_check.get("model", {})
    _ckpt_whisper_keys = [k for k in _ckpt_sd.keys() if k.startswith("whisper.")]

    if len(_ckpt_whisper_keys) < 300:
        raise RuntimeError(
            f"Whisper weights missing from checkpoint: only {len(_ckpt_whisper_keys)} "
            f"'whisper.*' keys found (expected ~390). Refusing to run inference — "
            f"the model would silently use stock openai/whisper-medium or the external "
            f"safetensors file, neither of which reflects the trained checkpoint."
        )
    logger.info(f"[whisper-verify] checkpoint contains {len(_ckpt_whisper_keys)} 'whisper.*' keys — OK")

    # Sanity: confirm whisper layer 0 of the loaded model matches the checkpoint's stored value
    _ck_w0 = _ckpt_sd.get("whisper.whisper.layers.0.self_attn.q_proj.weight")
    if _ck_w0 is not None:
        _live_w0 = model.whisper.whisper.layers[0].self_attn.q_proj.weight.detach().cpu()
        if not torch.allclose(_ck_w0.float(), _live_w0.float(), atol=1e-5):
            raise RuntimeError(
                "Whisper layer-0 weights in loaded model do NOT match the checkpoint. "
                "Something silently overrode them after checkpoint load."
            )
        logger.info("[whisper-verify] live whisper.layers[0].q_proj matches checkpoint — OK")
    del _raw_for_check, _ckpt_sd

    # Strip discriminators — they are dead weight during inference
    n_disc = 0
    for disc_name in ("mpd", "cqtd", "msstftd"):
        if hasattr(model, disc_name):
            disc_module = getattr(model, disc_name)
            n_disc += sum(p.numel() for p in disc_module.parameters())
            delattr(model, disc_name)
    if n_disc:
        logger.info(f"Removed {n_disc:,} discriminator params from model (not needed for inference).")
        import gc
        gc.collect()

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
    return feats.to(device), audio_len


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
        root = f.readline().strip()
        lines = f.readlines()

    # Resolve paths if relative
    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(root, path)

    # Select entries
    if args.sorted is not None and args.random is not None:
        raise ValueError("Cannot use both --sorted and --random. Choose one.")

    if args.sorted is not None:
        n = min(args.sorted, len(lines))
        selected_lines = lines[:n]
        logger.info(f"Processing first {n} entries (sorted)")
    elif args.random is not None:
        n = min(args.random, len(lines))
        selected_lines = random.sample(lines, n)
        logger.info(f"Processing {n} random entries")
    else:
        selected_lines = lines
        logger.info(f"Processing all {len(lines)} entries")

    # Target labels for outputs per entry — only healthy speaker embedding is used
    target_labels = ["healthy"]

    total_saved = 0
    for line in tqdm(selected_lines, desc="Entries"):
        parts = line.strip().split("\t")
        if len(parts) != 9:
            logger.warning(f"Skipping malformed line (expected 9 columns, got {len(parts)})")
            continue

        video_path   = resolve(parts[0])
        patho_path   = resolve(parts[1])
        healthy_path = resolve(parts[2])
        synth_paths  = [resolve(p) for p in parts[3:9]]  # 6 synthetic audio paths
        # Use parent dir as the unique id (patho file is often generically named, e.g.
        # "pathological.wav", so the directory is what distinguishes entries)
        parent_dir = os.path.basename(os.path.dirname(patho_path))
        patho_basename = os.path.splitext(os.path.basename(patho_path))[0]
        sample_id = f"{parent_dir}__{patho_basename}"

        target_paths = [healthy_path]

        # Load RAW video frames (once per entry, shared across all chunks/targets)
        try:
            raw_frames = load_video(video_path)  # numpy (T, H, W)
        except Exception as e:
            logger.warning(f"Skipping {sample_id}: video load failed — {e}")
            continue

        # Load RAW pathological wav (once per entry)
        try:
            sr_in, wav_data = wavfile.read(patho_path)
            assert sr_in == 16000, f"Expected 16kHz, got {sr_in} for {patho_path}"
            if wav_data.dtype == np.int16:
                wav_data = wav_data / 32768.0
            wav_data = wav_data.astype(np.float32)
        except Exception as e:
            logger.warning(f"Skipping {sample_id}: pathological audio load failed — {e}")
            continue

        # Determine chunk plan. AV-HuBERT video runs at 25 fps; audio at 16 kHz.
        if args.chunk_seconds is not None and len(wav_data) > int(args.chunk_seconds * 16000):
            chunk_audio = int(args.chunk_seconds * 16000)
            chunk_video = int(args.chunk_seconds * 25)
            n_chunks = (len(wav_data) + chunk_audio - 1) // chunk_audio
        else:
            chunk_audio = len(wav_data)
            chunk_video = len(raw_frames)
            n_chunks = 1

        # Preload xvectors once per entry (independent of chunking)
        spk_emb_by_target = {}
        for target_label, target_path in zip(target_labels, target_paths):
            try:
                xvector_path = _xvector_path(target_path)
                spk_emb = torch.load(xvector_path, map_location="cpu", weights_only=True)
                if spk_emb.dim() > 1:
                    spk_emb = spk_emb.squeeze()   # → (512,)
                spk_emb_by_target[target_label] = spk_emb.unsqueeze(0).to(device)  # (1, 512)
            except Exception as e:
                logger.warning(f"Skipping {sample_id} target {target_label}: xvector load failed — {e}")

        # Process each chunk independently. Audio and video are sliced from the same
        # wall-clock window so each chunk's streams stay synchronized.
        for chunk_idx in range(n_chunks):
            a_start = chunk_idx * chunk_audio
            a_end = min(a_start + chunk_audio, len(wav_data))
            v_start = chunk_idx * chunk_video
            v_end = min(v_start + chunk_video, len(raw_frames))

            chunk_wav = wav_data[a_start:a_end]
            chunk_frames = raw_frames[v_start:v_end]

            if len(chunk_wav) == 0 or len(chunk_frames) == 0:
                logger.debug(f"Skipping empty chunk {chunk_idx+1}/{n_chunks} for {sample_id}")
                continue

            # Process video chunk: transform + tensorize
            processed_frames = video_transform(chunk_frames)
            processed_frames = np.expand_dims(processed_frames, axis=-1)         # (T, H, W, 1)
            video_tensor = torch.from_numpy(processed_frames.astype(np.float32))
            video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, 1, T, H, W)
            video_len = video_tensor.size(2)
            padding_mask = torch.zeros((1, video_len), dtype=torch.bool, device=device)

            # Process audio chunk: Whisper features
            whisper_feats = whisper_processor(
                chunk_wav, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            patho_len = int(len(chunk_wav))

            part_suffix = f"_part{chunk_idx+1}" if n_chunks > 1 else ""

            # Run each of the 3 targets on this chunk
            for target_label, target_path in zip(target_labels, target_paths):
                if target_label not in spk_emb_by_target:
                    continue  # xvector failed earlier — already warned
                spk_emb = spk_emb_by_target[target_label]

                out_name = f"{sample_id}_target-{target_label}{part_suffix}.wav"
                out_path = os.path.join(args.output_dir, out_name)

                # Build source dict for inference
                # During inference the model uses source['audio'] for Whisper encoding
                # and source['synth_audio'] if available. We set both to the pathological
                # audio so the model processes the pathological input.
                source = {
                    "audio":          whisper_feats,                                   # (1, 80, 3000)
                    "video":          video_tensor,                                    # (1, 1, T, H, W)
                    "synth_audio":    whisper_feats,                                   # pathological audio
                    "audio_lengths":  torch.tensor([patho_len], dtype=torch.long,
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
