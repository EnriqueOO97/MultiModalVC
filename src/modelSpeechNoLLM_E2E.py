"""
End-to-End Speech Synthesis Model: Stage 1 (MMS_Speech_NoLLM) + Stage 2 (HiFi-GAN)

Inherits from MMS_Speech_NoLLM and adds HiFi-GAN vocoder.
Key change: conformer output (512-dim) is routed through an adapter conv_pre (512→512)
into HiFi-GAN's upsampling layers, bypassing mel_head (Stage 1).
"""

import sys
import os
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq.models import register_model
from typing import Optional

from .modelSpeechNoLLM import MMS_Speech_NoLLM, MMS_Speech_NoLLM_Config

# HiFi-GAN imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_hifigan"))
from hifigan.generator import HifiganGenerator
from hifigan.discriminator import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)

logger = logging.getLogger(__name__)


@dataclass
class MMS_Speech_NoLLM_E2E_Config(MMS_Speech_NoLLM_Config):
    # Path to pretrained HiFi-GAN checkpoint (model-best.pt from trainGermanVocoder.py)
    vocoder_checkpoint: str = field(
        default="???", metadata={"help": "Path to pretrained HiFi-GAN checkpoint"}
    )
    # Whether to freeze Stage 1 components initially
    freeze_stage1: bool = field(
        default=True, metadata={"help": "Whether to freeze Stage 1 components initially"}
    )


@register_model("MMS_Speech_NoLLM_E2E", dataclass=MMS_Speech_NoLLM_E2E_Config)
class MMS_Speech_NoLLM_E2E(MMS_Speech_NoLLM):
    """
    End-to-end speech synthesis model.
    
    Architecture:
        AV-HuBERT + Whisper → Q-Former → proj1 → interpolation → proj2 → conformer → ln3
        → [512-dim] → conv_pre adapter (512→512) → HiFi-GAN upsampling → waveform
    
    Removed bottleneck:
        - mel_head (512→128) from Stage 1: BYPASSED
        - conv_pre (128→512) from HiFi-GAN: REPLACED with conv_pre (512→512) adapter
    """

    def __init__(self, avhubert, whisper, cfg):
        # Initialize all Stage 1 components via parent
        super().__init__(avhubert, whisper, cfg)
        
        # mel_head is inherited but will NOT be used in forward_speech
        # We keep it in the state dict so Stage 1 checkpoint loading doesn't break
        
        # =====================================================================
        # HiFi-GAN Generator
        # =====================================================================
        # Create full generator first, then we'll load weights and extract components
        self._full_vocoder = HifiganGenerator()
        
        # Load pretrained HiFi-GAN weights
        if cfg.vocoder_checkpoint and cfg.vocoder_checkpoint != "???":
            self._load_vocoder_weights(cfg.vocoder_checkpoint)
        
        # Extract all vocoder components
        self.vocoder_ups = self._full_vocoder.ups
        self.vocoder_resblocks = self._full_vocoder.resblocks
        self.vocoder_conv_post = self._full_vocoder.conv_post
        self.vocoder_num_kernels = self._full_vocoder.num_kernels
        self.vocoder_num_upsamples = self._full_vocoder.num_upsamples
        
        # Adapter conv_pre: 512→512 (replaces original 128→512)
        # This bridges the conformer output distribution to what vocoder upsampling expects
        from torch.nn.utils import weight_norm
        self.vocoder_conv_pre = weight_norm(
            nn.Conv1d(512, 512, 7, 1, padding=3)
        )
        
        # We don't need the full vocoder anymore — components are extracted
        del self._full_vocoder
        
        # =====================================================================
        # HiFi-GAN Discriminator
        # =====================================================================
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        
        # Freeze discriminator from fairseq's optimizer — it has its own
        for param in self.mpd.parameters():
            param.requires_grad = False
        for param in self.msd.parameters():
            param.requires_grad = False
        
        # =====================================================================
        # Optionally freeze Stage 1 components
        # =====================================================================
        if cfg.freeze_stage1:
            self._freeze_stage1()
        
        # Update freeze_params list
        self.freeze_params = [n for n, p in self.named_parameters() if not p.requires_grad]
        
        logger.info(f"[E2E Model] Total params: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(f"[E2E Model] Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        logger.info(f"[E2E Model] Frozen params: {sum(p.numel() for p in self.parameters() if not p.requires_grad):,}")

    def _load_vocoder_weights(self, checkpoint_path):
        """Load pretrained HiFi-GAN generator weights."""
        logger.info(f"[E2E Model] Loading vocoder weights from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if "generator" in state and "model" in state["generator"]:
            # trainGermanVocoder.py format
            gen_state = state["generator"]["model"]
        elif "ema_generator" in state and state["ema_generator"] is not None:
            # EMA generator (preferred if available)
            gen_state = state["ema_generator"]
            logger.info("[E2E Model] Using EMA generator weights")
        elif "generator" in state:
            gen_state = state["generator"]
        else:
            gen_state = state
        
        # Load into full vocoder
        missing, unexpected = self._full_vocoder.load_state_dict(gen_state, strict=False)
        if missing:
            logger.warning(f"[E2E Model] Missing vocoder keys: {missing}")
        if unexpected:
            logger.warning(f"[E2E Model] Unexpected vocoder keys: {unexpected}")
        logger.info("[E2E Model] Vocoder weights loaded successfully")

    def _freeze_stage1(self):
        """Freeze Stage 1 components (proj1, proj2, conformer, ln layers, Q-Former, etc).
        Keeps vocoder upsampling trainable."""
        stage1_prefixes = (
            "proj1.", "proj2.", "conformer.", "ln1.", "ln2.", "ln3.",
            "mel_head.",  # not used but keep frozen
            "avhubert.", "whisper.",  # already frozen
            "Qformer.", "query_tokens", "avfeat_to_llm.",
            "afeat_1d_conv.", "vfeat_1d_conv.",
            "sr_predictor.",
            "audio_mask_emb", "video_mask_emb",
            "multimodal_attention_layer.",
        )
        for name, param in self.named_parameters():
            if any(name.startswith(prefix) for prefix in stage1_prefixes):
                param.requires_grad = False

    @classmethod
    def build_model(cls, cfg, task):
        """Build E2E model by calling parent's build_model."""
        import os
        from fairseq import checkpoint_utils, tasks
        from fairseq.dataclass.utils import convert_namespace_to_omegaconf
        from argparse import Namespace
        from avhubert.hubert_asr import HubertEncoderWrapper
        from transformers import WhisperForConditionalGeneration
        from .sub_model.modules import WhisperEncoderWrapper
        
        # === Build AV-HuBERT (identical to parent) ===
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if hasattr(cfg, 'w2v_path') and cfg.w2v_path is not None and cfg.w2v_path != '???':
            w2v_path = cfg.w2v_path
        else:
            w2v_path = f'{root_dir}/pretrained_models/avhubert/large_vox_iter5.pt'

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize

        w2v_args.task.data = cfg.data
        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)
        avhubert = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            del state['model']['mask_emb']
            avhubert.w2v_model.load_state_dict(state["model"], strict=False)
        avhubert.w2v_model.remove_pretraining_modules()

        whisper_ = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").model.encoder
        whisper = WhisperEncoderWrapper(whisper_)

        return cls(avhubert, whisper, cfg)

    def vocoder_forward(self, x):
        """
        Run through adapter conv_pre + HiFi-GAN upsampling chain.
        
        Args:
            x: (B, 512, T) — conformer output, already transposed
        
        Returns:
            waveform: (B, 1, T_audio)
        """
        # Adapter: bridges conformer output → vocoder upsampling input distribution
        o = self.vocoder_conv_pre(x)
        
        for i in range(self.vocoder_num_upsamples):
            o = F.leaky_relu(o, 0.1)
            o = self.vocoder_ups[i](o)
            z_sum = None
            for j in range(self.vocoder_num_kernels):
                if z_sum is None:
                    z_sum = self.vocoder_resblocks[i * self.vocoder_num_kernels + j](o)
                else:
                    z_sum += self.vocoder_resblocks[i * self.vocoder_num_kernels + j](o)
            o = z_sum / self.vocoder_num_kernels
        o = F.leaky_relu(o)
        o = self.vocoder_conv_post(o)
        o = torch.tanh(o)
        return o

    def forward_speech(self, **kwargs):
        """
        End-to-end forward: AV input → waveform.
        
        Overrides parent's forward_speech. Runs the same pipeline up to conformer+ln3,
        then routes through HiFi-GAN upsampling instead of mel_head.
        """
        # =====================================================================
        # PART A: IDENTICAL TO PARENT — Feature extraction, fusion, Q-Former,
        # projection, interpolation, conformer
        # Uses all inherited modules (self.whisper, self.avhubert, self.Qformer,
        # self.proj1, self.proj2, self.conformer, self.ln1, self.ln2, self.ln3)
        # =====================================================================
        
        # 1. Whisper & AVHubert feature extraction (no grad)
        with torch.no_grad():
            whisper_enc_out = self.whisper(kwargs['source'])
            avhubert_source = {'audio': None, 'video': kwargs['source']['video']}
            avhubert_output = self.avhubert(source=avhubert_source, padding_mask=kwargs['padding_mask'])
            avhubert_output['encoder_out'] = avhubert_output['encoder_out'].transpose(0, 1)

        video_lengths = torch.sum(~avhubert_output['padding_mask'], dim=1).tolist()
        max_vid_len = max(video_lengths)
        
        # 2. Speech rate predictor and query length calculation
        if self.cfg.use_sr_predictor:
            len_queries, resized_len_list = self.query_length_calculation(whisper_enc_out, video_lengths, max_vid_len)
        else:
            len_queries = [max(int(vid_len / 25 * self.cfg.queries_per_sec), self.cfg.queries_per_sec) for vid_len in video_lengths]

        # 3. Feature processing and modality fusion
        whisper_enc_out = self.afeat_1d_conv(whisper_enc_out.transpose(1, 2)).transpose(1, 2) 

        if self.cfg.use_qformer:
            padding_mask = (~avhubert_output['padding_mask']).long()
            len_feat = video_lengths 
        else:
            padding_mask = avhubert_output['padding_mask'][:, 1::2]
            padding_mask = (~padding_mask).long()
            len_feat = torch.sum(padding_mask, dim=1).tolist()
            avhubert_output['encoder_out'] = self.vfeat_1d_conv(
                avhubert_output['encoder_out'].transpose(1, 2)
            ).transpose(1, 2)

        B_dim, T_v, _ = avhubert_output['encoder_out'].size()
        whisper_enc_out = whisper_enc_out[:, :T_v, :]

        # Modality dropout (training only)
        mode = 'av'
        if self.training:
            mode = random.choices(
                ['av', 'video_only', 'audio_only'],
                weights=[self.cfg.p_modality_av, self.cfg.p_modality_video_only, self.cfg.p_modality_audio_only]
            )[0]
            if mode == 'video_only':
                whisper_enc_out = self.audio_mask_emb.unsqueeze(0).unsqueeze(0).expand_as(whisper_enc_out)
            elif mode == 'audio_only':
                avhubert_output['encoder_out'] = self.video_mask_emb.unsqueeze(0).unsqueeze(0).expand_as(avhubert_output['encoder_out'])

        # Fuse modalities
        if self.modality_fuse == 'concat':
            av_feat = torch.cat([whisper_enc_out, avhubert_output['encoder_out']], dim=2)
        elif self.modality_fuse == 'add':
            av_feat = whisper_enc_out + avhubert_output['encoder_out']
        elif self.modality_fuse == 'cross-att':
            av_feat = self.multimodal_attention_layer(
                audio_feature=whisper_enc_out,
                visual_feature=avhubert_output['encoder_out']
            )
        else:
            raise ValueError(f"Unknown modality fusion type: {self.modality_fuse}")

        # 4. Q-Former compression
        if self.cfg.use_qformer:
            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            query_output = self.avfeat_to_llm(query_output)
            queries = query_output
            query_lengths = len_queries
        else:
            queries = self.avfeat_to_llm(av_feat)
            query_lengths = len_feat

        # Stack with padding
        B = queries.size(0)
        av_lengths = query_lengths
        max_av_len = max(av_lengths)
        av_hidden_padded = queries.new_zeros((B, max_av_len, queries.size(2)))
        for i in range(B):
            av_hidden_padded[i, :av_lengths[i], :] = queries[i, :av_lengths[i], :]

        # Projection 1: 1024 -> 768
        x = self.proj1(av_hidden_padded.to(self.proj1.weight.dtype))
        x = self.ln1(x)

        # Interpolation to target audio length
        audio_lengths = None
        if isinstance(kwargs.get('source'), dict):
            audio_lengths = kwargs['source'].get('audio_lengths', None)
        if audio_lengths is None:
            audio_lengths = kwargs.get('audio_lengths', None)

        if audio_lengths is None:
            audio = kwargs['source'].get('audio', None) if isinstance(kwargs.get('source'), dict) else None
            if audio is not None:
                if audio.dim() == 2:
                    audio_lengths = torch.full((audio.size(0),), audio.size(1), device=audio.device, dtype=torch.long)
                elif audio.dim() == 3 and audio.size(1) == 1:
                    audio_lengths = torch.full((audio.size(0),), audio.size(2), device=audio.device, dtype=torch.long)

        if audio_lengths is None:
            raise ValueError("Audio lengths required for speech interpolation.")

        n_fft = 1024
        hop_length = 160
        pad = (n_fft - hop_length) // 2
        audio_lengths = audio_lengths.to(dtype=torch.long)
        target_lengths = torch.div(audio_lengths + 2 * pad - n_fft, hop_length, rounding_mode='floor') + 1
        target_lengths = torch.clamp(target_lengths, min=1)
        max_target_len = int(target_lengths.max().item())

        x = x.transpose(1, 2)  # (B, C, T)
        B, C, T_av = x.size()
        x_up = x.new_zeros((B, C, max_target_len))
        for i in range(B):
            actual_av_len = av_lengths[i]
            x_slice = x[i:i+1, :, :actual_av_len]
            tgt_len = int(target_lengths[i].item())
            x_i = F.interpolate(x_slice, size=tgt_len, mode='linear', align_corners=False)
            x_up[i, :, :tgt_len] = x_i[0]
        x = x_up.transpose(1, 2)  # (B, T, C)

        # Projection 2: 768 -> 512
        x = self.proj2(x)
        x = self.ln2(x)

        # Conformer
        x = self.conformer(x)
        x = self.ln3(x)  # (B, T, 512)

        # =====================================================================
        # PART B: E2E CHANGE — Route through HiFi-GAN upsampling instead of mel_head
        # =====================================================================
        
        # Transpose for HiFi-GAN: (B, T, 512) → (B, 512, T)
        x = x.transpose(1, 2)
        
        # Run through HiFi-GAN upsampling chain
        waveform = self.vocoder_forward(x)  # (B, 1, T_audio)
        
        return {
            "waveform": waveform,          # (B, 1, T_audio)
            "target_lengths": target_lengths,  # mel frame lengths per sample
        }

    def discriminate(self, waveform):
        """Run waveform through both discriminators.
        
        Args:
            waveform: (B, 1, T_audio) or (B, T_audio)
            
        Returns:
            mpd_scores, mpd_features, msd_scores, msd_features
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        mpd_scores, mpd_features = self.mpd(waveform)
        msd_scores, msd_features = self.msd(waveform)
        
        return mpd_scores, mpd_features, msd_scores, msd_features
