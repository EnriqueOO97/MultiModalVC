"""
MelVC: minimal variant of the SynthVC model that outputs MEL SPECTROGRAMS
(BigVGAN format) instead of waveforms.

Reuses, unchanged, two things from MMS_Speech_NoLLM_E2E_SynthVC:
  1. Speaker conditioning — the ConformerEncoderWithCrossAttn (xvector via
     per-layer cross-attention), inherited as-is.
  2. The upsampling block — Q-Former -> proj1 -> transposed-conv -> interpolation
     (`_run_pipeline_to_conformer`), inherited as-is.

Differences from SynthVC:
  - NO vocoder, NO discriminator, NO adversarial.
  - NO two-step (canonical/synthetic) pass — a single straightforward forward.
  - The conformer output (B, T, 512) goes to a mel head (512 -> num_mels) instead
    of the vocoder.
  - The interpolation target `tgt_len` is the BigVGAN MEL-FRAME count (passed in
    as `source['mel_target_lengths']` by the criterion, which computes the GT
    mel), not the vocoder-input length.

The mel head is format-agnostic: it learns whatever target the criterion gives
it. "BigVGAN-compatible" lives entirely in the criterion's target mel.
"""

import logging
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from fairseq.models import register_model

from .modelSpeechNoLLM_E2E_SynthVC import (
    MMS_Speech_NoLLM_E2E_SynthVC,
    MMS_Speech_NoLLM_E2E_SynthVC_Config,
)

logger = logging.getLogger(__name__)


@dataclass
class MMS_Speech_NoLLM_MelVC_Config(MMS_Speech_NoLLM_E2E_SynthVC_Config):
    mel_bands: int = field(default=80, metadata={"help": "Output mel bands (BigVGAN: 80)."})
    modality_mode: str = field(default="av", metadata={
        "help": "'av' (both encoders) | 'audio_only' (video masked) | 'video_only' "
                "(audio masked). Forces the fusion mode CONSISTENTLY across "
                "train / validation / inference. Default 'av' preserves prior behavior."})
    bigvgan_hop: int = field(default=256, metadata={"help": "BigVGAN hop (for inference-time tgt_len fallback)."})
    bigvgan_sr: int = field(default=22050, metadata={"help": "BigVGAN sample rate."})
    source_sr: int = field(default=16000, metadata={"help": "Dataset audio sample rate."})
    use_discriminator: bool = field(default=False, metadata={
        "help": "Attach the mel-domain U-Net discriminator (Guo 2022) for adversarial "
                "mel training. Disc weights live in the model (checkpointed); the "
                "criterion owns its optimizer and the warmup/ramp schedule."})
    # --- Task-2 warp: interleaved content cross-attn in the conformer (default OFF) ---
    use_content_xattn: bool = field(default=False, metadata={
        "help": "Interleave content cross-attn into the conformer (odd/even blocks "
                "swap self-attn for cross-attn to Q-Former content tokens)."})
    content_interleave_start: str = field(default="odd", metadata={
        "help": "'odd' → 1-based blocks 1,3,5,... get content cross-attn."})
    # --- Task-1 length: duration predictor (default OFF) ---
    use_duration_predictor: bool = field(default=False, metadata={
        "help": "Predict N (mel frames) at inference from content tokens + target spk."})
    dur_pred_layers: int = field(default=2, metadata={"help": "Duration predictor transformer layers."})
    dur_r_min: float = field(default=0.5, metadata={
        "help": "Min output/input ratio: N in [r_min*L_in, L_in]. Never slower than input."})


@register_model("MMS_Speech_NoLLM_MelVC", dataclass=MMS_Speech_NoLLM_MelVC_Config)
class MMS_Speech_NoLLM_MelVC(MMS_Speech_NoLLM_E2E_SynthVC):
    """Conformer(+speaker xattn) -> mel head -> BigVGAN-format mels."""

    def __init__(self, avhubert, whisper, cfg):
        super().__init__(avhubert, whisper, cfg)

        # The parent (_E2E) builds a HiFi-GAN vocoder in __init__. MelVC never
        # uses it (forward stops at the mel head), so drop it entirely to free
        # GPU memory and keep it out of the DDP graph. Pass vocoder_checkpoint=""
        # in the launch script so it isn't even weight-loaded before we delete it.
        for attr in ("_full_vocoder", "vocoder_ups", "vocoder_resblocks",
                     "vocoder_conv_post", "vocoder_conv_pre"):
            if hasattr(self, attr):
                delattr(self, attr)

        self.modality_mode = getattr(cfg, "modality_mode", "av")
        assert self.modality_mode in ("av", "audio_only", "video_only"), \
            f"[MelVC] invalid modality_mode={self.modality_mode!r}"
        # Training-only modality dropout is ENABLED only when modality_mode == 'av'.
        # In that case each TRAIN step samples a mode from p_modality_*; validation
        # and inference are always forced to full 'av'. When modality_mode != 'av'
        # (e.g. 'audio_only'), that mode is applied deterministically everywhere and
        # the dropout probabilities are ignored.
        # Capture the dropout probabilities HERE from the construction-time cfg.
        # (self.cfg is overwritten by fairseq post-init with a defaults-filled config,
        # so reading self.cfg.p_modality_* at forward time yields the defaults 1/0/0 —
        # which silently disables dropout. Storing them now mirrors modality_mode.)
        self._p_modality = [
            float(getattr(cfg, "p_modality_av", 1.0)),
            float(getattr(cfg, "p_modality_video_only", 0.0)),
            float(getattr(cfg, "p_modality_audio_only", 0.0)),
        ]
        self._dropout_active = (self.modality_mode == "av") and (
            self._p_modality[1] > 0.0 or self._p_modality[2] > 0.0
        )
        logger.warning(
            f"[MelVC] modality_mode={self.modality_mode!r}; "
            f"train-time dropout {'ON' if self._dropout_active else 'OFF'} "
            f"weights av/video_only/audio_only="
            f"{self._p_modality[0]}/{self._p_modality[1]}/{self._p_modality[2]}  "
            f"(dropout only in training; val+inference forced to {self.modality_mode!r})")

        self.mel_bands = getattr(cfg, "mel_bands", 80)
        # Conformer (size "L") outputs 512-dim features; project to mel bands.
        self.mel_head = nn.Linear(512, self.mel_bands)
        nn.init.xavier_uniform_(self.mel_head.weight)
        nn.init.constant_(self.mel_head.bias, 0.0)
        logger.info(f"[MelVC] mel_head 512 -> {self.mel_bands} bands; "
                    f"vocoder deleted (single-pass mel output).")

        # Optional mel-domain discriminator (adversarial training). Held here so its
        # weights are checkpointed with the model; the criterion manages its optimizer.
        if getattr(cfg, "use_discriminator", False):
            from .mel_discriminator import MelTFDiscriminator
            self.mel_disc = MelTFDiscriminator()
            logger.info("[MelVC] mel-domain U-Net discriminator attached (Guo 2022).")

        # --- Task-2 warp: rebuild the conformer with interleaved content cross-attn ---
        # (trains from scratch, so replacing the parent-built conformer is free.)
        qformer_dim = int(getattr(cfg, "qformer_dim", 1024))
        if getattr(cfg, "use_content_xattn", False):
            from .divise_conformer.encoder_xattn import ConformerEncoderWithCrossAttn
            self.content_proj = nn.Linear(qformer_dim, 512)
            self.conformer = ConformerEncoderWithCrossAttn(
                size="L", use_content_xattn=True,
                content_interleave_start=getattr(cfg, "content_interleave_start", "odd"))
            logger.info("[MelVC] content cross-attn ON (interleaved conformer blocks).")

        # --- Task-1 length: duration predictor ---
        self.dur_r_min = float(getattr(cfg, "dur_r_min", 0.5))
        self._dur_loss = None
        if getattr(cfg, "use_duration_predictor", False):
            from .sub_model.modules import DurationPredictor
            self.duration_predictor = DurationPredictor(
                content_dim=qformer_dim, spk_dim=512,
                num_layers=int(getattr(cfg, "dur_pred_layers", 2)), r_min=self.dur_r_min)
            logger.info("[MelVC] duration predictor ON (predicts N at inference).")

    def _maybe_predict_duration(self, content_tokens, query_lengths, spk_emb,
                                video_lengths, target_lengths, max_target_len):
        """Predict N = L_in * ratio. Train: teacher-force GT N, stash L_dur.
        Inference: override target_lengths with the prediction. Fully detached."""
        if getattr(self, "duration_predictor", None) is None:
            return target_lengths, max_target_len
        device = content_tokens.device
        frame_rate = float(self.cfg.bigvgan_sr) / float(self.cfg.bigvgan_hop)  # ~86.13 Hz
        L_in = (torch.tensor(video_lengths, dtype=torch.float32, device=device)
                / 25.0 * frame_rate).clamp(min=1.0)               # input mel frames
        spk = (spk_emb.detach() if spk_emb is not None
               else content_tokens.new_zeros(content_tokens.size(0), 512))
        ratio = self.duration_predictor(content_tokens.detach(), spk).float()  # (B,) in (r_min,1)
        if torch.is_grad_enabled():
            gt = target_lengths.to(device=device, dtype=torch.float32)
            ratio_tgt = torch.clamp(gt / L_in, self.dur_r_min, 1.0)
            self._dur_loss = torch.abs(ratio - ratio_tgt).mean()
            return target_lengths, max_target_len                 # teacher forcing
        self._dur_loss = None
        N = torch.clamp((L_in * ratio).round().long(), min=1)
        return N, int(N.max().item())

    def forward_speech(self, **kwargs):
        """Single pass: input audio + video + spk_emb -> conformer -> mel head.

        Returns dict with `melspec` (B, T_mel, bands).
        """
        src = kwargs["source"]

        # --- speaker embedding (B,512) -> (B,1,512) for cross-attention ---
        spk_emb = src.get("spk_embeddings", None)
        if spk_emb is None:
            spk_emb = kwargs.get("spk_embeddings", None)
        if spk_emb is not None:
            spk_emb = spk_emb.unsqueeze(1)

        # --- AV-HuBERT, once. NO no_grad wrapper: grad is gated by requires_grad,
        # so only the unfrozen top-N layers build graph / get gradient; the frozen
        # lower layers produce non-grad tensors and are pruned from the graph
        # automatically (no activation storage, no backward). Val/inference still
        # run under fairseq's global no_grad. ponytail: split layers only if OOM.
        avhubert_source = {"audio": None, "video": src["video"]}
        avhubert_output = self.avhubert(source=avhubert_source, padding_mask=kwargs["padding_mask"])
        avhubert_output["encoder_out"] = avhubert_output["encoder_out"].transpose(0, 1)
        video_lengths = torch.sum(~avhubert_output["padding_mask"], dim=1).tolist()
        max_vid_len = max(video_lengths)

        # --- interpolation target = BigVGAN mel-frame count ---
        # Preferred: provided by the criterion (== GT mel time dim) so predicted
        # and GT mels are guaranteed the same length. Fallback (inference): derive
        # from the 16 kHz audio length scaled to the BigVGAN SR/hop.
        device = next(self.parameters()).device
        mel_tl = src.get("mel_target_lengths", None)
        if mel_tl is not None:
            target_lengths = mel_tl.to(device=device, dtype=torch.long)
        else:
            audio_lengths = src.get("audio_lengths", None)
            if audio_lengths is None:
                raise ValueError("MelVC: need source['mel_target_lengths'] (train) "
                                 "or source['audio_lengths'] (inference).")
            audio_lengths = audio_lengths.to(device=device, dtype=torch.long)
            hop = int(getattr(self.cfg, "bigvgan_hop", 256))
            sr_t = int(getattr(self.cfg, "bigvgan_sr", 22050))
            sr_s = int(getattr(self.cfg, "source_sr", 16000))
            len_target_sr = (audio_lengths * sr_t) // sr_s
            target_lengths = torch.clamp((len_target_sr - hop) // hop + 1, min=1)
        max_target_len = int(target_lengths.max().item())

        # --- Whisper on the input audio. NO no_grad: same rationale as AV-HuBERT
        # above — only the unfrozen top-N whisper layers get graph/gradient.
        whisper_enc_out = self.whisper(src)

        # --- shared pipeline: fusion -> transconv -> interp(tgt=mel frames)
        #     -> proj2 -> conformer(+spk_emb) ---
        mode = self.modality_mode
        # Stochastic modality dropout: TRAINING ONLY, and only when modality_mode=='av'.
        # Gate on torch.is_grad_enabled() (True only in the training forward; fairseq
        # runs validation/inference under torch.no_grad()) — this is robust even if the
        # module's .training flag is not reliably set on the real training forwards.
        is_train_fwd = torch.is_grad_enabled()
        if self._dropout_active and is_train_fwd:
            mode = random.choices(
                ["av", "video_only", "audio_only"],
                weights=self._p_modality,
            )[0]
        # Diagnostics: first 25 forwards + every 500th, with a cumulative mode histogram,
        # so we can VERIFY dropout is actually firing video_only in training.
        self._fwd_count = getattr(self, "_fwd_count", 0) + 1
        hist = getattr(self, "_mode_hist", None) or {}
        hist[mode] = hist.get(mode, 0) + 1
        self._mode_hist = hist
        if self._fwd_count <= 25 or self._fwd_count % 500 == 0:
            # Also expose self.cfg-at-forward vs the values captured in __init__, to
            # detect whether self.cfg reverts to defaults at forward (the suspected bug).
            try:
                cfg_qps = self.cfg.queries_per_sec
                cfg_pvo = self.cfg.p_modality_video_only
            except Exception as e:
                cfg_qps = cfg_pvo = f"<err:{e}>"
            logger.warning(
                f"[MelVC dbg] fwd#{self._fwd_count} .training={self.training} "
                f"grad_enabled={is_train_fwd} mode={mode!r} cum_hist={hist} "
                f"|| self.cfg.queries_per_sec={cfg_qps} (init-built used construction cfg) "
                f"self.cfg.p_modality_video_only={cfg_pvo} vs captured _p_modality={self._p_modality}")

        features = self._run_pipeline_to_conformer(
            whisper_enc_out, avhubert_output, video_lengths,
            max_vid_len, target_lengths, max_target_len, spk_emb, mode=mode,
        )  # (B, T_mel, 512)

        melspec = self.mel_head(features)  # (B, T_mel, bands)

        return {
            "melspec": melspec,
            "target_lengths": target_lengths,
            "residual_ratio": getattr(self, "_last_residual_ratio", None),
            "dur_loss": getattr(self, "_dur_loss", None),   # duration predictor loss (or None)
        }
