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
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        "--use-ema",
        action="store_true",
        default=False,
        help="Load EMA weights from the checkpoint instead of raw weights. "
             "Falls back silently to raw weights if no EMA state is found.",
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


def apply_ema_weights(model, ema_state):
    """Overwrite model weights with EMA shadow weights from the criterion state.

    ema_state: the dict already extracted from state["criterion"]["ema_state"],
               or None.  Caller is responsible for loading (avoids double I/O).
    Returns True if EMA weights were applied, False if not found.
    """
    if ema_state is None:
        logger.warning("--use-ema requested but no 'ema_state' found in checkpoint criterion — "
                       "using raw model weights.")
        return False

    # During DDP training the model is wrapped multiple times (e.g. fairseq adds
    # LegacyDistributedDataParallel on top of DistributedFairseqModel), so
    # model.named_parameters() inside the criterion returns names like
    # "module.module.conformer.blocks.0.weight".  At inference the model is
    # unwrapped, so names are just "conformer.blocks.0.weight".
    # Strip ALL leading "module." segments before matching.
    def _strip_module(k):
        while k.startswith("module."):
            k = k[len("module."):]
        return k

    missing, unexpected = [], []
    model_params = dict(model.named_parameters())
    ema_state_stripped = {_strip_module(k): v for k, v in ema_state.items()}
    for name, ema_tensor in ema_state_stripped.items():
        if name in model_params:
            model_params[name].data.copy_(ema_tensor.to(model_params[name].device))
        else:
            unexpected.append(name)
    for name in model_params:
        if name not in ema_state_stripped:
            missing.append(name)

    logger.info(f"EMA weights applied ({len(ema_state)} tensors). "
                f"Missing from EMA: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        logger.debug(f"  Missing (will keep raw): {missing[:5]}{'...' if len(missing)>5 else ''}")
    return True


def load_model(checkpoint_path, manifest_dir, device, extract_ema=False):
    """Load the SynthVC model from checkpoint.

    Always returns (model, ema_state).  ema_state is None unless extract_ema=True
    and EMA state is present in the checkpoint.
    """
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

    # Ensure new config fields exist for older checkpoints that predate them.
    # This prevents OmegaConf merge failures on the primary loading path.
    _ckpt_cfg = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("cfg", {})
    _model_cfg = _ckpt_cfg.get("model", {}) if isinstance(_ckpt_cfg, dict) else getattr(_ckpt_cfg, "model", {})
    _model_dict = _model_cfg if isinstance(_model_cfg, dict) else OmegaConf.to_container(_model_cfg, resolve=True)
    if "use_cqt" not in _model_dict:
        model_overrides["model"]["use_cqt"] = False
    if "upsampling_method" not in _model_dict:
        model_overrides["model"]["upsampling_method"] = "interpolation"
    del _ckpt_cfg, _model_cfg, _model_dict

    # EMA extraction: only pay for torch.load when --use-ema is requested.
    ema_state = None
    if extract_ema:
        _raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ema_state = _raw.get("criterion", {}).get("ema_state", None)
        del _raw
        logger.info(f"EMA state {'found' if ema_state else 'NOT FOUND'} in checkpoint.")

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

        from src.task_synthvc import MMS_LLaMA_TrainingSynthVCTask
        task = MMS_LLaMA_TrainingSynthVCTask.setup_task(cfg.task)

        from src.modelSpeechNoLLM_E2E_SynthVC import MMS_Speech_NoLLM_E2E_SynthVC
        model = MMS_Speech_NoLLM_E2E_SynthVC.build_model(cfg.model, task)

        model.load_state_dict(_fb["model"], strict=False)
        del _fb

    model.eval()
    model.to(device)
    logger.info("Model loaded and set to eval mode.")
    return model, ema_state


# ---------------------------------------------------------------------------
# Multi-resolution mel spectrogram — exact same scales as MultiResolutionMelLoss
# in criterionSpeechE2E.py: (n_fft, hop_size, win_size) × num_mels=80
# ---------------------------------------------------------------------------
MEL_RESOLUTIONS = [
    ("fine",   512,  120,  512),
    ("medium", 1024, 160, 1024),
    ("coarse", 2048, 480, 2048),
]
MEL_NUM_MELS   = 80
MEL_SAMPLE_RATE = 16000


def _build_logmel(n_fft, hop_size, win_size):
    """Build a log-mel transform matching LogMelSpectrogram in criterionSpeechE2E.py."""
    mel = T.MelSpectrogram(
        sample_rate=MEL_SAMPLE_RATE,
        n_fft=n_fft,
        win_length=win_size,
        hop_length=hop_size,
        center=False,
        power=2.0,
        norm=None,
        f_min=0,
        f_max=8000,
        onesided=True,
        n_mels=MEL_NUM_MELS,
        mel_scale="slaney",
    )
    return mel


def wav_to_multires_mels(wav_np, device):
    """Compute fine/medium/coarse log-mel tensors from a float32 numpy waveform.

    Args:
        wav_np: 1-D float32 numpy array in [-1, 1]
        device:  torch device

    Returns:
        dict with keys "fine", "medium", "coarse", each a (num_mels, T) float32 tensor on CPU.
    """
    wav = torch.from_numpy(wav_np).float().unsqueeze(0)  # (1, T)
    mels = {}
    for name, n_fft, hop_size, win_size in MEL_RESOLUTIONS:
        logmel_fn = _build_logmel(n_fft, hop_size, win_size)
        pad = (n_fft - hop_size) // 2
        wav_padded = F.pad(wav, (pad, pad), "reflect")
        mel = logmel_fn(wav_padded)                      # (1, num_mels, T_mel)
        logmel = torch.log(torch.clamp(mel, min=1e-5))   # (1, num_mels, T_mel)
        mels[name] = logmel[0].cpu()                     # (num_mels, T_mel)
    return mels


def save_mel_pair_pngs(canon_mels, pred_mels, syn_idx, sample_id, output_dir):
    """Save one paired PNG per resolution scale (canonical on top, model prediction on bottom).

    Output file names:  mel_fine_{ID}.png / mel_medium_{ID}.png / mel_coarse_{ID}.png
    """
    for name in ("fine", "medium", "coarse"):
        canon = canon_mels[name].numpy()   # (num_mels, T_canon)
        pred  = pred_mels[name].numpy()    # (num_mels, T_pred)

        fig, axes = plt.subplots(2, 1, figsize=(12, 5))

        for ax, data, title in zip(
            axes,
            [canon, pred],
            [f"Canonical  —  {sample_id}",
             f"Prediction Syn{syn_idx}  —  {sample_id}"],
        ):
            im = ax.imshow(data, aspect="auto", origin="lower",
                           interpolation="nearest", cmap="magma")
            ax.set_title(title, fontsize=9)
            ax.set_ylabel("Mel band")
            ax.set_xlabel("Frame")
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

        fig.suptitle(f"Log-mel  [{name}]  —  {sample_id}", fontsize=10, y=1.01)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"mel_{name}_{sample_id}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved {os.path.basename(out_path)}")


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

    model, ema_state = load_model(args.checkpoint, manifest_dir, device, extract_ema=args.use_ema)
    if args.use_ema:
        ema_applied = apply_ema_weights(model, ema_state)
        if ema_applied:
            logger.info("Running inference with EMA weights.")
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

        # Pick one synthetic index (1-based) for mel generation — chosen once per entry
        mel_syn_idx = random.randint(1, 6)
        pred_wav_for_mel = None  # will hold the model prediction for mel_syn_idx

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

            # Capture the prediction for mel plotting
            if syn_idx == mel_syn_idx:
                pred_wav_for_mel = wav_np

        # -------------------------------------------------------------------
        # Multi-resolution mel generation (once per entry)
        # Prediction: model output for the randomly chosen variant (mel_syn_idx)
        # Canonical:  the original canonical audio waveform
        # -------------------------------------------------------------------
        if pred_wav_for_mel is not None:
            try:
                # --- Prediction mels ---
                pred_mels = wav_to_multires_mels(pred_wav_for_mel, device)

                # --- Canonical mels ---
                _, canon_wav = wavfile.read(canon_path)
                if canon_wav.dtype == np.int16:
                    canon_wav = canon_wav / 32768.0
                canon_wav = canon_wav.astype(np.float32)
                canon_mels = wav_to_multires_mels(canon_wav, device)

                # --- Save paired PNGs (canon top / prediction bottom, one per scale) ---
                save_mel_pair_pngs(canon_mels, pred_mels, mel_syn_idx, sample_id,
                                   args.output_dir)
                logger.info(f"{sample_id}: saved mel pair PNGs (Syn{mel_syn_idx} prediction vs canonical)")
            except Exception as e:
                logger.warning(f"{sample_id}: mel generation failed — {e}")
        else:
            logger.warning(f"{sample_id}: no prediction available for mel_syn_idx={mel_syn_idx}, skipping mel PNGs")

    logger.info(f"Done. Saved {total_saved} WAV files to {args.output_dir}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run_inference(args)
