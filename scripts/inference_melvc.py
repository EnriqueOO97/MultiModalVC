"""
MelVC inference: pathological audio+video -> predicted BigVGAN-format mel
-> FROZEN BigVGAN vocoder -> healthy waveform (22.05 kHz).

Pipeline per manifest entry:
    patho audio (Whisper feats) + video + healthy xvector
        -> MelVC model            (melspec, B x T_mel x 80)
        -> transpose + float32     (B x 80 x T_mel)
        -> frozen BigVGAN          (B x 1 x T_audio, @ 22050 Hz)
        -> int16 WAV

Everything the MelVC model needs is inside the model checkpoint (the fine-tuned
Whisper encoder lives there as 367 `whisper.*` tensors; AV-HuBERT backbone is the
frozen structural prior loaded from w2v_path). No stock/external Whisper weights
are used — the [whisper-verify] checks below enforce that.

The BigVGAN vocoder is loaded SEPARATELY and frozen. Its mel format MUST match the
mel the model was trained to emit (22050 / 80-band / n_fft1024 / hop256 / fmax8000)
— this is asserted at load time; a mismatch would silently produce garbage audio.

Output naming: {sample_id}_target-healthy.wav
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

# Fairseq registration (MelVC model + criterion must be imported so the
# @register_model / @register_criterion decorators run).
import src.task
import src.model
import src.task_pathological_finetune
import src.modelSpeechNoLLM_E2E_SynthVC
import src.criterionSpeechE2E_SynthVC
import src.modelSpeechNoLLM_MelVC
import src.criterionMelVC

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
logger = logging.getLogger("melvc-inference")


# ---------------------------------------------------------------------------
# Default paths. Override any of these on the command line.
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "/data/fs201163/eo49197/MultiModalVC/exp/multiModalVC-synth/melvc_v2/checkpoints/checkpoint44.pt"
# Code repo (architecture). Added to sys.path so `import bigvgan` works.
DEFAULT_BIGVGAN_CODE_DIR = "/data/fs201163/eo49197/BigVGAN"
# Weights dir: a local folder holding `config.json` + `bigvgan_generator.pt`
# for `bigvgan_v2_22khz_80band_fmax8k_256x`.
DEFAULT_BIGVGAN_CKPT_DIR = "/data/fs201163/eo49197/MultiModalVC/pretrained_models/BigVGAN"

# Test split to run inference on. PASTE THE PATH OF YOUR TEST MANIFEST HERE
# (or pass --manifest at runtime to override).
DEFAULT_MANIFEST = ""

# Mel format the model was trained to emit — BigVGAN must match these exactly.
EXPECTED_MEL = dict(sampling_rate=22050, num_mels=80, n_fft=1024, hop_size=256, fmax=8000)


# ---------------------------------------------------------------------------
# SELF-CONTAINED INFERENCE FIX: the fine-tuned Whisper lives INSIDE the model
# checkpoint. The task's build-time `_load_finetuned_whisper` would instead try
# to read an external safetensors path baked into the checkpoint cfg (unreachable
# on most machines). Replace it with a no-op so Whisper weights come EXCLUSIVELY
# from the checkpoint; the [whisper-verify] checks prove it after load.
# ---------------------------------------------------------------------------
import src.task_pathological_finetune as _tpf_mod


def _skip_external_whisper(self, model):
    logger.info(
        "[inference] external Whisper swap skipped — Whisper weights come from the model checkpoint"
    )


_tpf_mod.MMS_PathologicalFinetuneTask._load_finetuned_whisper = _skip_external_whisper


def get_parser():
    p = argparse.ArgumentParser(
        description="MelVC inference: patho audio+video + healthy xvector -> mel -> BigVGAN -> wav"
    )
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                   help="Path to the trained MelVC model checkpoint")
    p.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST,
                   help="Pathological test manifest (TSV, 9 columns)")
    p.add_argument("--output-dir", type=str, default="melvc-outputs",
                   help="Directory for output WAV files")
    p.add_argument("--bigvgan-code-dir", type=str, default=DEFAULT_BIGVGAN_CODE_DIR,
                   help="BigVGAN code repo (for `import bigvgan`)")
    p.add_argument("--bigvgan-ckpt-dir", type=str, default=DEFAULT_BIGVGAN_CKPT_DIR,
                   help="Local dir with config.json + bigvgan_generator.pt")
    p.add_argument("--sorted", type=int, default=None, metavar="N",
                   help="Process the first N entries (sorted order)")
    p.add_argument("--random", type=int, default=None, metavar="N",
                   help="Process N randomly chosen entries")
    p.add_argument("--sample-rate", type=int, default=22050,
                   help="Output WAV sample rate (BigVGAN native = 22050)")
    p.add_argument("--chunk-seconds", type=float, default=None, metavar="SEC",
                   help="Split inputs longer than SEC into SEC-second chunks "
                        "(each chunk -> its own _partN WAV). Omit to process whole.")
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: cuda if available, else cpu)")
    return p


def _xvector_path(wav_path: str) -> str:
    """Convert ``<stem>.wav`` -> ``<stem>_xvector.pt``."""
    if wav_path.endswith(".wav"):
        return wav_path[:-4] + "_xvector.pt"
    raise ValueError(f"Expected .wav path, got: {wav_path}")


def load_model(checkpoint_path, manifest_dir, device):
    """Load the trained MelVC model from checkpoint (no vocoder, no disc)."""
    logger.info(f"Loading MelVC model from {checkpoint_path}...")

    w2v_path = os.path.join(repo_root, "pretrained_models/avhubert/large_vox_iter5.pt")

    model_overrides = {
        "task": {"data": manifest_dir, "label_dir": manifest_dir},
        "model": {
            "w2v_path": w2v_path,
            "stage1_checkpoint": "",
            "vocoder_checkpoint": "",   # MelVC deletes the vocoder in __init__
        },
    }

    # Ensure config fields exist + detect transconv depth from the state_dict.
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
        model_overrides["model"]["transconv_layers"] = 4 if _has_conv4 else (3 if _has_conv3 else 2)
    del _raw, _ckpt_cfg, _model_cfg, _model_dict, _sd

    try:
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides=model_overrides, strict=False,
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
        from src.modelSpeechNoLLM_MelVC import MMS_Speech_NoLLM_MelVC
        model = MMS_Speech_NoLLM_MelVC.build_model(cfg.model, task)
        model.load_state_dict(_fb["model"], strict=False)
        del _fb

    model.eval()

    # --- Verify Whisper weights came FROM the checkpoint (no HF fallback) ---
    _raw_for_check = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _ckpt_sd = _raw_for_check.get("model", {})
    _ckpt_whisper_keys = [k for k in _ckpt_sd.keys() if k.startswith("whisper.")]
    if len(_ckpt_whisper_keys) < 300:
        raise RuntimeError(
            f"Whisper weights missing from checkpoint: only {len(_ckpt_whisper_keys)} "
            f"'whisper.*' keys (expected ~367). Refusing to run inference."
        )
    logger.info(f"[whisper-verify] checkpoint has {len(_ckpt_whisper_keys)} 'whisper.*' keys — OK")
    _ck_w0 = _ckpt_sd.get("whisper.whisper.layers.0.self_attn.q_proj.weight")
    if _ck_w0 is not None:
        _live_w0 = model.whisper.whisper.layers[0].self_attn.q_proj.weight.detach().cpu()
        if not torch.allclose(_ck_w0.float(), _live_w0.float(), atol=1e-5):
            raise RuntimeError("Whisper layer-0 weights in loaded model do NOT match the checkpoint.")
        logger.info("[whisper-verify] live whisper.layers[0].q_proj matches checkpoint — OK")
    del _raw_for_check, _ckpt_sd

    # MelVC already deletes the vocoder in __init__ and has no discriminators.
    if hasattr(model, "mel_head"):
        logger.info(f"[melvc] mel_head present: 512 -> {model.mel_head.out_features} bands.")
    else:
        raise RuntimeError("Loaded model has no mel_head — is this a MelVC checkpoint?")

    model.to(device)
    logger.info("MelVC model loaded and set to eval mode.")
    return model


def load_bigvgan(code_dir, ckpt_dir, device):
    """Load and freeze the BigVGAN vocoder from a local code dir + weights dir."""
    if not ckpt_dir:
        raise ValueError(
            "BigVGAN checkpoint dir not set. Download "
            "`bigvgan_v2_22khz_80band_fmax8k_256x` (config.json + bigvgan_generator.pt) "
            "and pass --bigvgan-ckpt-dir <dir> (or edit DEFAULT_BIGVGAN_CKPT_DIR)."
        )
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import bigvgan as bigvgan_mod  # from the code repo

    logger.info(f"Loading BigVGAN from {ckpt_dir} (code: {code_dir})...")
    voc = bigvgan_mod.BigVGAN.from_pretrained(ckpt_dir, use_cuda_kernel=False)
    voc.remove_weight_norm()
    voc = voc.eval().to(device)
    for p in voc.parameters():
        p.requires_grad = False

    # Hard-assert the mel format matches what the model emits.
    h = voc.h
    got = dict(sampling_rate=h.sampling_rate, num_mels=h.num_mels,
               n_fft=h.n_fft, hop_size=h.hop_size, fmax=h.fmax)
    for k, v in EXPECTED_MEL.items():
        if int(got[k]) != int(v):
            raise RuntimeError(
                f"BigVGAN mel mismatch on '{k}': vocoder={got[k]} vs model-trained={v}. "
                f"Wrong checkpoint — would produce garbage audio."
            )
    logger.info(f"[bigvgan] mel format OK: {got}")
    return voc


def _to_video_tensor(frames, transform, device):
    """numpy (T,H,W) -> transformed (1,1,T,H,W) tensor + padding mask."""
    frames = transform(frames)
    frames = np.expand_dims(frames, axis=-1)                       # (T,H,W,1)
    t = torch.from_numpy(frames.astype(np.float32))
    t = t.permute(3, 0, 1, 2).unsqueeze(0).to(device)             # (1,1,T,H,W)
    mask = torch.zeros((1, t.size(2)), dtype=torch.bool, device=device)
    return t, mask


def run_inference(args):
    if not args.manifest:
        raise ValueError(
            "No test manifest set. Paste a path into DEFAULT_MANIFEST at the top of "
            "this script, or pass --manifest <test.tsv>."
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_dir = os.path.dirname(args.manifest)

    model = load_model(args.checkpoint, manifest_dir, device)
    vocoder = load_bigvgan(args.bigvgan_code_dir, args.bigvgan_ckpt_dir, device)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

    # Image transform — values from the MelVC / SynthVC config.
    video_transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((88, 88)),
        Normalize(0.421, 0.165),
    ])

    with open(args.manifest, "r") as f:
        root = f.readline().strip()
        lines = f.readlines()

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(root, path)

    if args.sorted is not None and args.random is not None:
        raise ValueError("Cannot use both --sorted and --random.")
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

    total_saved = 0
    for line in tqdm(selected_lines, desc="Entries"):
        parts = line.strip().split("\t")
        if len(parts) != 9:
            logger.warning(f"Skipping malformed line (expected 9 columns, got {len(parts)})")
            continue

        video_path   = resolve(parts[0])
        patho_path   = resolve(parts[1])
        healthy_path = resolve(parts[2])
        parent_dir = os.path.basename(os.path.dirname(patho_path))
        patho_basename = os.path.splitext(os.path.basename(patho_path))[0]
        sample_id = f"{parent_dir}__{patho_basename}"

        # Load raw video + patho audio once per entry.
        try:
            raw_frames = load_video(video_path)                   # (T,H,W)
        except Exception as e:
            logger.warning(f"Skipping {sample_id}: video load failed — {e}")
            continue
        try:
            sr_in, wav_data = wavfile.read(patho_path)
            assert sr_in == 16000, f"Expected 16kHz, got {sr_in} for {patho_path}"
            if wav_data.dtype == np.int16:
                wav_data = wav_data / 32768.0
            wav_data = wav_data.astype(np.float32)
        except Exception as e:
            logger.warning(f"Skipping {sample_id}: patho audio load failed — {e}")
            continue

        # Healthy xvector (the conversion target).
        try:
            xv = torch.load(_xvector_path(healthy_path), map_location="cpu", weights_only=True)
            if xv.dim() > 1:
                xv = xv.squeeze()
            spk_emb = xv.unsqueeze(0).to(device)                  # (1,512)
        except Exception as e:
            logger.warning(f"Skipping {sample_id}: xvector load failed — {e}")
            continue

        # Chunk plan (AV-HuBERT video @ 25 fps, audio @ 16 kHz).
        if args.chunk_seconds is not None and len(wav_data) > int(args.chunk_seconds * 16000):
            chunk_audio = int(args.chunk_seconds * 16000)
            chunk_video = int(args.chunk_seconds * 25)
            n_chunks = (len(wav_data) + chunk_audio - 1) // chunk_audio
        else:
            chunk_audio = len(wav_data)
            chunk_video = len(raw_frames)
            n_chunks = 1

        for chunk_idx in range(n_chunks):
            a0, a1 = chunk_idx * chunk_audio, min((chunk_idx + 1) * chunk_audio, len(wav_data))
            v0, v1 = chunk_idx * chunk_video, min((chunk_idx + 1) * chunk_video, len(raw_frames))
            chunk_wav = wav_data[a0:a1]
            chunk_frames = raw_frames[v0:v1]
            if len(chunk_wav) == 0 or len(chunk_frames) == 0:
                continue

            video_tensor, padding_mask = _to_video_tensor(chunk_frames, video_transform, device)
            whisper_feats = whisper_processor(
                chunk_wav, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)                            # (1,80,3000)

            # No mel_target_lengths at inference -> MelVC derives the mel-frame
            # count from audio_lengths (16k -> 22.05k / hop256).
            source = {
                "audio":          whisper_feats,
                "video":          video_tensor,
                "audio_lengths":  torch.tensor([len(chunk_wav)], dtype=torch.long, device=device),
                "spk_embeddings": spk_emb,
            }

            part_suffix = f"_part{chunk_idx+1}" if n_chunks > 1 else ""
            out_name = f"{sample_id}_target-healthy{part_suffix}.wav"
            out_path = os.path.join(args.output_dir, out_name)

            try:
                with torch.no_grad():
                    out = model(source=source, padding_mask=padding_mask)
                    mel = out["melspec"].transpose(1, 2).float()   # (1,80,T_mel), model runs bf16
                    wav_gen = vocoder(mel)                         # (1,1,T_audio) @ 22050
            except Exception as e:
                logger.warning(f"Skipping {out_name}: forward/vocoder failed — {e}")
                continue

            wav_np = wav_gen.squeeze(0).squeeze(0).cpu().float().numpy()
            wav_np = np.clip(wav_np, -1.0, 1.0)
            wav_int16 = (wav_np * 32767.0).astype(np.int16)
            wavfile.write(out_path, args.sample_rate, wav_int16)
            total_saved += 1
            logger.debug(f"Saved {out_name} | samples={len(wav_int16)}")

    logger.info(f"Done. Saved {total_saved} WAV files to {args.output_dir}")


if __name__ == "__main__":
    run_inference(get_parser().parse_args())
