"""
SynthVC-Inspired End-to-End Speech Synthesis Model.

Inherits from MMS_Speech_NoLLM_E2E and adds:
1. Conformer with per-layer cross-attention to speaker embeddings
2. Two-step forward pass:
   - Step 1 (canonical): full pipeline → waveform → mel loss
   - Step 2 (synthetic): pipeline up to conformer → target features → conversion loss

The conversion loss is L1 between canonical_features and target_features.
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

from .modelSpeechNoLLM_E2E import MMS_Speech_NoLLM_E2E, MMS_Speech_NoLLM_E2E_Config
from .divise_conformer.encoder_xattn import ConformerEncoderWithCrossAttn

logger = logging.getLogger(__name__)


@dataclass
class MMS_Speech_NoLLM_E2E_SynthVC_Config(MMS_Speech_NoLLM_E2E_Config):
    pass  # No additional config needed; conformer cross-attn is always on


@register_model("MMS_Speech_NoLLM_E2E_SynthVC", dataclass=MMS_Speech_NoLLM_E2E_SynthVC_Config)
class MMS_Speech_NoLLM_E2E_SynthVC(MMS_Speech_NoLLM_E2E):
    """
    SynthVC-inspired E2E model with two-step forward and conversion loss.

    Architecture changes:
        - ConformerEncoder → ConformerEncoderWithCrossAttn (per-layer cross-attn)
        - Speaker embedding (B, 512) is passed as cross-attention context

    Forward pass:
        Step 1: canonical audio+video → conformer(cross-attn to spk_emb) → vocoder → waveform
        Step 2: synthetic audio + canonical video → conformer(cross-attn to spk_emb) → target features
    """

    def __init__(self, avhubert, whisper, cfg):
        super().__init__(avhubert, whisper, cfg)

        # Replace the conformer with cross-attention variant
        # The parent already created self.conformer = ConformerEncoder(size="L")
        # We replace it here with our cross-attention version
        self.conformer = ConformerEncoderWithCrossAttn(size="L")

        # Update freeze_params list after replacing conformer
        self.freeze_params = [n for n, p in self.named_parameters() if not p.requires_grad]

        logger.info(f"[SynthVC Model] Replaced conformer with cross-attention variant")
        logger.info(f"[SynthVC Model] Total params: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(f"[SynthVC Model] Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        self.num_updates = 0

    def set_num_updates(self, num_updates):
        """Store global update count for criterion to use."""
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    @classmethod
    def build_model(cls, cfg, task):
        """Build SynthVC model — identical to parent's build_model."""
        import os
        from fairseq import checkpoint_utils, tasks
        from fairseq.dataclass.utils import convert_namespace_to_omegaconf
        from argparse import Namespace
        from avhubert.hubert_asr import HubertEncoderWrapper
        from transformers import WhisperForConditionalGeneration
        from .sub_model.modules import WhisperEncoderWrapper

        # === Build AV-HuBERT ===
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

    def _run_frontend(self, source, padding_mask, spk_emb):
        """Run the full frontend pipeline: encoders → fusion → Q-Former → proj → conformer.

        Shared logic between canonical and synthetic passes to avoid code duplication.

        Args:
            source: dict with 'audio' (Whisper features) and 'video' (video tensor)
            padding_mask: padding mask for AV-HuBERT
            spk_emb: (B, 512) speaker embedding
            
        Returns:
            av_feat, whisper_enc_out, avhubert_output, video_lengths, max_vid_len
            (intermediate values needed for further processing)
        """
        # This method is NOT used directly — see forward_speech for the full pipeline.
        # Keeping it here as documentation of the shared logic.
        raise NotImplementedError("Use forward_speech directly")

    def _run_pipeline_to_conformer(self, whisper_enc_out, avhubert_output, video_lengths,
                                    max_vid_len, target_lengths, max_target_len, spk_emb,
                                    mode='av'):
        """Run pipeline from fused features through conformer with cross-attention.

        This is the shared core between canonical and synthetic passes, starting
        AFTER the Whisper/AV-HuBERT encoding (which differs between passes).

        Args:
            whisper_enc_out: (B, T, 1024) — Whisper encoder output
            avhubert_output: dict with 'encoder_out' (B, T, 1024) and 'padding_mask'
            video_lengths: list of per-sample video lengths
            max_vid_len: max video length in batch
            target_lengths: (B,) target mel frame lengths for interpolation
            max_target_len: scalar, max target length
            spk_emb: (B, 1, 512) speaker embedding for cross-attention
            mode: modality dropout mode ('av', 'video_only', 'audio_only') —
                  sampled ONCE in forward_speech and shared across both passes.

        Returns:
            conformer_features: (B, T, 512) — output of conformer + ln3
        """
        # 1. Feature processing
        whisper_proc = self.afeat_1d_conv(whisper_enc_out.transpose(1, 2)).transpose(1, 2)

        if self.cfg.use_qformer:
            padding_mask_q = (~avhubert_output['padding_mask']).long()
            len_feat = video_lengths
        else:
            padding_mask_q = avhubert_output['padding_mask'][:, 1::2]
            padding_mask_q = (~padding_mask_q).long()
            len_feat = torch.sum(padding_mask_q, dim=1).tolist()
            avhubert_output['encoder_out'] = self.vfeat_1d_conv(
                avhubert_output['encoder_out'].transpose(1, 2)
            ).transpose(1, 2)

        B_dim, T_v, _ = avhubert_output['encoder_out'].size()
        whisper_proc = whisper_proc[:, :T_v, :]

        # 2. Modality dropout — mode is pre-determined by the caller, not sampled here
        if mode == 'video_only':
            whisper_proc = self.audio_mask_emb.unsqueeze(0).unsqueeze(0).expand_as(whisper_proc)
        elif mode == 'audio_only':
            avhubert_output['encoder_out'] = self.video_mask_emb.unsqueeze(0).unsqueeze(0).expand_as(avhubert_output['encoder_out'])

        # 3. Fuse modalities
        if self.modality_fuse == 'concat':
            av_feat = torch.cat([whisper_proc, avhubert_output['encoder_out']], dim=2)
        elif self.modality_fuse == 'add':
            av_feat = whisper_proc + avhubert_output['encoder_out']
        elif self.modality_fuse == 'cross-att':
            av_feat = self.multimodal_attention_layer(
                audio_feature=whisper_proc,
                visual_feature=avhubert_output['encoder_out']
            )
        else:
            raise ValueError(f"Unknown modality fusion type: {self.modality_fuse}")

        # 4. Q-Former compression
        if self.cfg.use_sr_predictor:
            len_queries, resized_len_list = self.query_length_calculation(whisper_enc_out, video_lengths, max_vid_len)
        else:
            len_queries = [max(int(vid_len / 25 * self.cfg.queries_per_sec), self.cfg.queries_per_sec)
                           for vid_len in video_lengths]

        if self.cfg.use_qformer:
            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            query_output = self.avfeat_to_llm(query_output)
            queries = query_output
            query_lengths = len_queries
        else:
            queries = self.avfeat_to_llm(av_feat)
            query_lengths = len_feat

        # 5. Pad and project
        B = queries.size(0)
        av_lengths = query_lengths
        max_av_len = max(av_lengths)
        av_hidden_padded = queries.new_zeros((B, max_av_len, queries.size(2)))
        for i in range(B):
            av_hidden_padded[i, :av_lengths[i], :] = queries[i, :av_lengths[i], :]

        x = self.proj1(av_hidden_padded.to(self.proj1.weight.dtype))
        x = self.ln1(x)

        # 6. Interpolation to target audio length
        x = x.transpose(1, 2)  # (B, C, T)
        B, C, T_av = x.size()
        x_up = x.new_zeros((B, C, max_target_len))
        for i in range(B):
            actual_av_len = av_lengths[i]
            x_slice = x[i:i + 1, :, :actual_av_len]
            tgt_len = int(target_lengths[i].item())
            x_i = F.interpolate(x_slice, size=tgt_len, mode='linear', align_corners=False)
            x_up[i, :, :tgt_len] = x_i[0]
        x = x_up.transpose(1, 2)  # (B, T, C)

        # 7. Projection 2 + Conformer with cross-attention
        x = self.proj2(x)
        x = self.ln2(x)
        x = self.conformer(x, spk_emb=spk_emb)  # cross-attn to speaker embedding
        x = self.ln3(x)  # (B, T, 512)

        return x

    def forward_speech(self, **kwargs):
        """Forward pass with Phase 1 / Phase 2 / Inference branching.

        Phase 1 (disc_active=False, training):
            Two-step: canonical→waveform + synthetic→features (for conversion loss)

        Phase 2 (disc_active=True, training):
            Single synthetic pass: synth audio + canonical video → vocoder → waveform
            No canonical pass, no conversion loss. Modality forced to 'av'.

        Inference (not training):
            Single pass: input audio + video + spk_emb → vocoder → waveform

        Returns dict with:
            waveform: (B, 1, T_audio)
            target_lengths: (B,) — mel frame lengths
            canonical_features: (B, T, 512) or None
            target_features: (B, T, 512) or None
        """
        # =====================================================================
        # Determine phase: disc_active flag is set by criterion
        # =====================================================================
        disc_active = kwargs['source'].get('disc_active', False)

        # =====================================================================
        # SHARED: Extract speaker embeddings
        # =====================================================================
        spk_emb = kwargs['source'].get('spk_embeddings', None)
        if spk_emb is None:
            spk_emb = kwargs.get('spk_embeddings', None)
        if spk_emb is not None:
            spk_emb = spk_emb.unsqueeze(1)  # (B, 512) → (B, 1, 512) for cross-attention

        # =====================================================================
        # SHARED: AV-HuBERT (runs ONCE — same video for all passes)
        # =====================================================================
        with torch.no_grad():
            avhubert_source = {'audio': None, 'video': kwargs['source']['video']}
            avhubert_output = self.avhubert(source=avhubert_source, padding_mask=kwargs['padding_mask'])
            avhubert_output['encoder_out'] = avhubert_output['encoder_out'].transpose(0, 1)

        video_lengths = torch.sum(~avhubert_output['padding_mask'], dim=1).tolist()
        max_vid_len = max(video_lengths)

        # =====================================================================
        # SHARED: Compute target lengths from canonical audio lengths
        # =====================================================================
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
        hop_length = 160  # Fixed: must match vocoder upsample factor (10*4*2*2=160)
        pad = (n_fft - hop_length) // 2
        audio_lengths = audio_lengths.to(dtype=torch.long)
        target_lengths = torch.div(audio_lengths + 2 * pad - n_fft, hop_length, rounding_mode='floor') + 1
        target_lengths = torch.clamp(target_lengths, min=1)
        max_target_len = int(target_lengths.max().item())

        # =====================================================================
        # PHASE 2 / INFERENCE: Single synthetic-only pass through vocoder
        # =====================================================================
        if disc_active or not self.training:
            # Force modality to 'av' — no dropout during voice conversion
            mode = 'av'

            # Use synthetic audio if available, otherwise use whatever audio is in source
            synth_audio = kwargs['source'].get('synth_audio', None)
            if synth_audio is None:
                synth_audio = kwargs.get('synth_audio', None)

            if synth_audio is not None:
                # Build source with synthetic audio for Whisper
                synth_source = {
                    'audio': synth_audio,
                    'video': kwargs['source']['video'],
                }
                with torch.no_grad():
                    whisper_enc_out = self.whisper(synth_source)
            else:
                # Inference fallback: use whatever audio is in source
                with torch.no_grad():
                    whisper_enc_out = self.whisper(kwargs['source'])

            avhubert_output_pass = {
                'encoder_out': avhubert_output['encoder_out'].clone(),
                'padding_mask': avhubert_output['padding_mask'],
            }

            features = self._run_pipeline_to_conformer(
                whisper_enc_out, avhubert_output_pass, video_lengths,
                max_vid_len, target_lengths, max_target_len, spk_emb, mode=mode
            )

            # Route through vocoder → waveform
            x_voc = features.transpose(1, 2)  # (B, 512, T)
            waveform = self.vocoder_forward(x_voc)  # (B, 1, T_audio)

            return {
                "waveform": waveform,
                "target_lengths": target_lengths,
                "canonical_features": None,
                "target_features": None,
            }

        # =====================================================================
        # PHASE 1: Two-step forward (canonical + synthetic)
        # =====================================================================
        # Sample modality dropout mode ONCE for both passes
        mode = random.choices(
            ['av', 'video_only', 'audio_only'],
            weights=[self.cfg.p_modality_av, self.cfg.p_modality_video_only, self.cfg.p_modality_audio_only]
        )[0]

        # --- STEP 1: CANONICAL PASS ---
        with torch.no_grad():
            whisper_enc_out_canon = self.whisper(kwargs['source'])

        avhubert_output_canon = {
            'encoder_out': avhubert_output['encoder_out'].clone(),
            'padding_mask': avhubert_output['padding_mask'],
        }

        canonical_features = self._run_pipeline_to_conformer(
            whisper_enc_out_canon, avhubert_output_canon, video_lengths,
            max_vid_len, target_lengths, max_target_len, spk_emb, mode=mode
        )

        # Route canonical features through vocoder → waveform
        x_voc = canonical_features.transpose(1, 2)  # (B, 512, T)
        waveform = self.vocoder_forward(x_voc)  # (B, 1, T_audio)

        # --- STEP 2: SYNTHETIC PASS (features only, no vocoder) ---
        target_features = None
        synth_audio = kwargs['source'].get('synth_audio', None)
        if synth_audio is None:
            synth_audio = kwargs.get('synth_audio', None)

        if synth_audio is not None:
            with torch.no_grad():
                synth_source = {
                    'audio': synth_audio,
                    'video': kwargs['source']['video'],
                }
                whisper_enc_out_synth = self.whisper(synth_source)

            avhubert_output_synth = {
                'encoder_out': avhubert_output['encoder_out'].clone(),
                'padding_mask': avhubert_output['padding_mask'],
            }

            target_features = self._run_pipeline_to_conformer(
                whisper_enc_out_synth, avhubert_output_synth, video_lengths,
                max_vid_len, target_lengths, max_target_len, spk_emb, mode=mode
            )
            # STOP HERE — no vocoder for synthetic pass

        return {
            "waveform": waveform,                    # (B, 1, T_audio)
            "target_lengths": target_lengths,        # (B,)
            "canonical_features": canonical_features, # (B, T, 512)
            "target_features": target_features,      # (B, T, 512) or None
        }
