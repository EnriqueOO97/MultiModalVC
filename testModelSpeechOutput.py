import os
import sys
import torch
import torchaudio
import numpy as np

# Mirror terminal export: PYTHONPATH="$(pwd)/fairseq:$(pwd):$PYTHONPATH"
repo_root = "/ceph/home/TUG/olivares-tug/MMS-LLaMA"
fairseq_dir = os.path.join(repo_root, "fairseq")
avhubert_dir = os.path.join(repo_root, "avhubert")
os.environ["PYTHONPATH"] = f"{fairseq_dir}:{repo_root}:{avhubert_dir}:" + os.environ.get("PYTHONPATH", "")

# Ensure repo + fairseq paths before importing fairseq
src_dir = os.path.join(repo_root, "src")
for p in [avhubert_dir, fairseq_dir, repo_root, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Avoid AVHubert debug import path that triggers duplicate model registration
if len(sys.argv) == 1:
    sys.argv.append("run")

from fairseq import checkpoint_utils, tasks
from fairseq.data.dictionary import Dictionary
from transformers import AutoTokenizer, WhisperProcessor
from omegaconf import OmegaConf

import src.task
from src.modelSpeech import MMS_LLaMA_Speech
from src.utils import Compose, Normalize, CenterCrop, load_video

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "/ceph/home/TUG/olivares-tug/MMS-LLaMA/pretrained_models/mms_llama/1759h/ckpt-1759h.pt"
llm_path = "meta-llama/Llama-3.2-3B"
w2v_path = os.path.join(repo_root, "pretrained_models/avhubert/muavic_multilingual_compatible.pt")

# Load config/task like demo.py (uses checkpoint defaults)
model_overrides = {
    "task": {
        "data": os.path.join(repo_root, "manifest/germanManifest"),
        "label_dir": os.path.join(repo_root, "manifest/germanManifest"),
        "llm_path": llm_path,
        "noise_prob": 0.75,
        "noise_wav": os.path.join(repo_root, "noise/babble_noise.wav"),
        "normalize": True,
    },
    "model": {
        "data": os.path.join(repo_root, "manifest/germanManifest"),
        "w2v_path": w2v_path,
        "llm_path": llm_path,
        "dropout_input": 0.0,
        "w2v_args": None,
        "normalize": True,
        "no_pretrained_weights": False,
        "window_level": False,
        "apply_mask": False,
        "mask_selection": "static",
        "mask_length": 10,
        "mask_other": 0,
        "mask_prob": 0.75,
        "no_mask_overlap": False,
        "mask_channel_selection": "static",
        "mask_channel_length": 64,
        "mask_channel_other": 0,
        "mask_channel_prob": 0.5,
        "no_mask_channel_overlap": False,
        "layerdrop": 0.1,
        "activation_dropout": 0.1,
        "attention_dropout": 0.0,
        "dropout": 0.0,
        "feature_grad_mult": 1.0,
        "freeze_finetune_updates": 0,
        "sr_predictor_layers": 2,
        "qformer_layers": 2,
        "qformer_dim": 1024,
        "queries_per_sec": 2,
        "use_qformer": True,
        "use_sr_predictor": True,
        "whisper_embed_dim": 1024,
        "avhubert_embed_dim": 1024,
        "llama_embed_dim": 3072,
        "modality_fuse": "concat",
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": "q_proj.k_proj.v_proj.o_proj",
    },
    "common": {
        "user_dir": src_dir,
    },
}
# 1. Load a valid base configuration structure from the checkpoint
state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path)
cfg = state["cfg"]

# 2. Disable strict mode so we can add keys that might be missing or 'protected'
OmegaConf.set_struct(cfg, False)

# 3. Apply overrides
# Task overrides
for k, v in model_overrides["task"].items():
    cfg.task[k] = v

# Model overrides (INCLUDING data, which is now allowed)
for k, v in model_overrides["model"].items():
    cfg.model[k] = v

# Common overrides
for k, v in model_overrides["common"].items():
    cfg.common[k] = v

# 4. Critical: Ensure the root-level fields needed by inner components are present
# src/model.py copies cfg.data to w2v_args.task.data
if not hasattr(cfg, "data") or cfg.data is None:
    cfg.data = model_overrides["task"]["data"]
    
# Explicitly set queries_per_sec for the test
cfg.model.queries_per_sec = 3

# 5. Build task and model (FROM SCRATCH)
task = tasks.setup_task(cfg.task)
task.load_dataset("test")

model = MMS_LLaMA_Speech.build_model(cfg.model, task)
model.eval().to(device)

# Prepare processors
tokenizer = AutoTokenizer.from_pretrained(llm_path)
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
video_transform = Compose([
    Normalize(0.0, 255.0),
    CenterCrop((88, 88)),
    Normalize(0.0, 1.0),
])

audio_path = "/ceph/home/TUG/olivares-tug/datasets/lrs3/lrs3_video_seg24s/test/0Fi83BHQsMA/00002.wav"
video_path = "/ceph/home/TUG/olivares-tug/datasets/lrs3/lrs3_video_seg24s/test/0Fi83BHQsMA/00002.mp4"

# Load raw audio and compute sample length (pre-Whisper padding)
wav, sr = torchaudio.load(audio_path)
if sr != 16000:
    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
wav = wav.squeeze(0)
audio_len_samples = torch.tensor([wav.numel()], dtype=torch.long, device=device)

# Whisper input features
audio_features = whisper_processor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)

# Video features
frames = load_video(video_path)
frames = video_transform(frames)
frames = np.expand_dims(frames, axis=-1)
video_tensor = torch.from_numpy(frames.astype(np.float32))
video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)

# Padding mask for video frames
T = video_tensor.shape[2]
padding_mask = torch.zeros((1, T), dtype=torch.bool, device=device)

# Instruction tokens (empty)
instruction_tokens = tokenizer("Focus on semantics, not voice characteristics", return_tensors="pt").input_ids[0].to(device)

# Minimal empty labels list for forward_speech
target_list = [torch.tensor([], dtype=torch.long, device=device)]

with torch.no_grad():
    out = model.forward_speech(
        source={
            "audio": audio_features,
            "video": video_tensor,
            "instruction": [instruction_tokens],
            "audio_lengths": audio_len_samples,
        },
        padding_mask=padding_mask,
        target_list=target_list,
    )

print(f"Instruction text: 'Focus on semantics, not voice characteristics'")
print(f"Instruction tokens length: {instruction_tokens.shape[0]}")

# Expected AV tokens = ceil(audio_duration * queries_per_sec)
duration_sec = audio_len_samples.item() / 16000
expected_av_tokens = int(duration_sec * model.cfg.queries_per_sec)
if model.cfg.use_sr_predictor:
     # sr_predictor doubles the queries if use_sr_predictor=True in some configs, 
     # but let's check what the model actually does.
     # model.py: max_queries = int(cfg.queries_per_sec * 20 * 2) if use_sr_predictor
     pass

print(f"Audio duration: {duration_sec:.2f}s")
print(f"Expected AV query tokens: ~{expected_av_tokens}")

print("-" * 20)
print("melspec shape:", out["melspec"].shape)
print("hidden_states shape:", out["hidden_states"].shape)

total_tokens = out["hidden_states"].shape[1]
av_part_len = total_tokens - instruction_tokens.shape[0]
print(f"inferred AV part length: {av_part_len}")