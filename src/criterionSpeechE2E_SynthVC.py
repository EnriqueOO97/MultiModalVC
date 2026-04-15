"""
SynthVC-Inspired E2E Criterion with Conversion Loss.

Phase 1 (disc_active=False):
    Loss = mel_loss_weight * mel_loss + conv_loss_weight * L1(canonical_features, target_features)
    Model runs two-step forward (canonical → waveform + synthetic → features).

Phase 2 (disc_active=True):
    Loss = mel_loss_weight * mel_loss + GAN losses (discriminator + feature matching)
    Model runs single synthetic-only forward → waveform.
    No conversion loss. Standard HiFi-GAN adversarial training.

disc_active is determined by: use_discriminator flag OR num_updates >= disc_start_updates.
"""

import logging
import os
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.dataclass import FairseqDataclass

from .criterionSpeechE2E import E2EGanLoss, E2EGanLossConfig, LogMelSpectrogram, MultiResolutionMelLoss

# Add HiFi-GAN path once at module load time (avoid repeated sys.path.insert in forward())
_HIFIGAN_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_hifigan")
if _HIFIGAN_PATH not in sys.path:
    sys.path.insert(0, _HIFIGAN_PATH)
from hifigan.discriminator import feature_loss, discriminator_loss, generator_loss

logger = logging.getLogger("src.criterionSpeechE2E_SynthVC")

# Exclude frozen encoders (never change → EMA = weights, wasteful) and
# discriminators (training-only, not used for inference).
# These are checked against STRIPPED (no module.) parameter names.
_EMA_EXCLUDE_PREFIXES = ("avhubert.", "whisper.", "mpd.", "msstftd.", "cqtd.")
_EMA_DECAY = 0.999  # Standard for HiFi-GAN / voice conversion models


def _ema_strip_module(name: str) -> str:
    """Strip all leading 'module.' segments from a DDP-wrapped parameter name.

    During DDP training (legacy_ddp + ModuleProxyWrapper), fairseq wraps the model
    twice, so named_parameters() returns 'module.module.conformer.xxx'.
    Storing EMA with stripped keys makes the EMA independent of DDP wrapping depth
    and keeps keys consistent with the model state_dict (which fairseq already strips).
    """
    while name.startswith("module."):
        name = name[len("module."):]
    return name

@dataclass
class E2EGanLossSynthVCConfig(E2EGanLossConfig):
    conv_loss_weight: float = field(
        default=5.0, metadata={"help": "Weight for conversion loss (SynthVC uses 45:5 mel:conv ratio)"}
    )
    disc_start_updates: int = field(
        default=30000, metadata={"help": "Auto-activate discriminator after this many updates"}
    )
    disc_grad_clip: float = field(
        default=0.0, metadata={"help": "Max gradient norm for discriminator (0 = disabled)"}
    )
    adv_warmup_updates: int = field(
        default=0, metadata={"help": "Linearly ramp adversarial loss weight from 0→1 over this many updates after disc activates (0 = disabled)"}
    )
    use_multires_mel: bool = field(
        default=False, metadata={"help": "Use multi-resolution mel loss (3 scales) instead of single-resolution"}
    )
    disc_lr_t_max: int = field(
        default=550000, metadata={"help": "T_max for disc cosine LR decay in num_updates units "
                                        "(= max_update minus the num_updates at disc activation). "
                                        "Default 550000 = 600000 max_update - 50000 disc activation point."}
    )
    disc_pretrain: bool = field(
        default=True, metadata={"help": "Train discriminator during Phase 1 without adversarial gradient to generator. "
                                        "Warms up disc weights so it is ready when adversarial training starts."}
    )


@register_criterion("e2e_gan_loss_synthvc", dataclass=E2EGanLossSynthVCConfig)
class E2EGanLossSynthVC(E2EGanLoss):
    def __init__(self, task, mel_loss_weight=40.0, use_discriminator=True,
                 disc_lr=2e-4, disc_betas="0.8,0.99", conv_loss_weight=5.0,
                 disc_start_updates=30000, mel_num_mels=128, mel_hop_size=160,
                 disc_grad_clip=0.0, adv_warmup_updates=0, use_multires_mel=False,
                 disc_lr_t_max=550000, disc_pretrain=True):
        super().__init__(task, mel_loss_weight, use_discriminator, disc_lr, disc_betas,
                         mel_num_mels=mel_num_mels, mel_hop_size=mel_hop_size)
        self.conv_loss_weight = conv_loss_weight
        self.disc_start_updates = disc_start_updates
        self.disc_grad_clip = disc_grad_clip
        self.adv_warmup_updates = adv_warmup_updates
        self.use_multires_mel = use_multires_mel
        self.disc_lr_t_max = disc_lr_t_max
        self.disc_pretrain = disc_pretrain
        self.disc_lr_scheduler = None  # CosineAnnealingLR, created alongside disc_optimizer
        self._num_updates = 0
        # _disc_active_since is stored in *model* update units (fairseq optimizer steps),
        # which are preserved across resumes.  Previous code used criterion-call units
        # (which reset to 0 on each resume), causing negative adv_weight after resume.
        self._disc_active_since = None
        self._adv_warmup_complete = False   # once True, adv_weight stays 1.0 forever
        self.multires_mel = None  # lazy init
        self._pending_disc_optimizer_state = None   # deferred disc optimizer state from checkpoint
        self._pending_disc_scheduler_state = None   # deferred disc LR scheduler state from checkpoint
        # Dummy parameter so fairseq's has_parameters() returns True and saves/loads
        # criterion state (disc_optimizer, _disc_active_since, ema_state)
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        # EMA shadow copy of trained generator parameters (on-GPU, same dtype as model)
        self._ema_state = {}
        self._ema_initialized = False
        logger.info(f"[SynthVC Criterion] conv_loss_weight={self.conv_loss_weight}, "
                     f"disc_start_updates={self.disc_start_updates}, "
                     f"disc_grad_clip={self.disc_grad_clip}, adv_warmup_updates={self.adv_warmup_updates}, "
                     f"use_multires_mel={self.use_multires_mel}, disc_lr_t_max={self.disc_lr_t_max}, "
                     f"disc_pretrain={self.disc_pretrain}, "
                     f"mel_num_mels={mel_num_mels}, mel_hop_size={mel_hop_size}")

    def _is_disc_active(self, model=None):
        """Determine if the discriminator phase is active."""
        if self.use_discriminator:
            return True
        
        # Use global update count from model if available (syncs across resumes)
        update_num = getattr(model, "num_updates", 0) if model is not None else 0
        
        # We assume update_freq=4 for the threshold (matching user's DISK_START_UPDATES=40000)
        # Fairseq num_updates is optimizer steps. 10k steps * 4 = 40k.
        effective_updates = max(self._num_updates, update_num * 4)
        return effective_updates >= self.disc_start_updates

    def _lazy_init(self, model, device):
        """Initialize LogMelSpectrogram and discriminator optimizer.
        Unlike the base E2EGanLoss, we ALWAYS initialize the disc_optimizer
        because we might transition to disc_active=True mid-training.
        """
        if self.logmel is None:
            self.logmel = LogMelSpectrogram(num_mels=self.mel_num_mels, hop_size=self.mel_hop_size).to(device)

        if self.use_multires_mel and self.multires_mel is None:
            self.multires_mel = MultiResolutionMelLoss(num_mels=self.mel_num_mels).to(device)
            logger.info(f"[SynthVC Criterion] Initialized multi-resolution mel loss (3 scales, {self.mel_num_mels} bands)")

        # Move EMA state to GPU after checkpoint load (checkpoints store EMA on CPU)
        if self._ema_initialized and self._ema_state:
            first_val = next(iter(self._ema_state.values()))
            if first_val.device != device:
                self._ema_state = {k: v.to(device) for k, v in self._ema_state.items()}
                logger.info(f"[SynthVC EMA] Moved EMA state to {device}")

        if self.disc_optimizer is None:
            # Spectral disc (MS-STFT or CQT) uses cuFFT internally — needs float32.
            # MPD works fine in bf16.
            self._use_cqt = hasattr(model, 'cqtd')
            self._spec_disc = model.cqtd if self._use_cqt else model.msstftd
            self._spec_disc.float()
            # Collect discriminator params - enable grad for optimizer
            disc_params = []
            for param in model.mpd.parameters():
                param.requires_grad = True
                disc_params.append(param)
            for param in self._spec_disc.parameters():
                param.requires_grad = True
                disc_params.append(param)

            self.disc_optimizer = torch.optim.AdamW(
                disc_params,
                lr=self.disc_lr,
                betas=self.disc_betas,
            )
            # Cosine LR decay for the discriminator.
            # T_max is in num_updates units; we step the scheduler once per num_updates tick
            # (see the disc step in forward()). min_lr = disc_lr / 10.
            self.disc_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.disc_optimizer,
                T_max=self.disc_lr_t_max,
                eta_min=self.disc_lr / 10.0,
            )
            logger.info(f"[SynthVC Criterion] Initialized disc optimizer with lr={self.disc_lr}, "
                       f"betas={self.disc_betas}, params={sum(p.numel() for p in disc_params):,}, "
                       f"cosine T_max={self.disc_lr_t_max}, eta_min={self.disc_lr/10.0}")

            # Apply deferred disc_optimizer and scheduler states from checkpoint resume
            if self._pending_disc_optimizer_state is not None:
                try:
                    self.disc_optimizer.load_state_dict(self._pending_disc_optimizer_state)
                    logger.info("[SynthVC Criterion] Loaded deferred disc_optimizer state from checkpoint")
                except Exception as e:
                    logger.warning(f"[SynthVC Criterion] Failed to load disc_optimizer state: {e}")
                self._pending_disc_optimizer_state = None
            if self._pending_disc_scheduler_state is not None:
                try:
                    self.disc_lr_scheduler.load_state_dict(self._pending_disc_scheduler_state)
                    logger.info("[SynthVC Criterion] Loaded deferred disc_lr_scheduler state from checkpoint")
                except Exception as e:
                    logger.warning(f"[SynthVC Criterion] Failed to load disc_lr_scheduler state: {e}")
                self._pending_disc_scheduler_state = None

    def _init_ema(self, model):
        """Initialize EMA shadow copy using canonical (stripped) parameter names.

        Keys are stored WITHOUT any 'module.' prefix so the EMA is independent of
        DDP wrapping depth.  The exclude filter is applied after stripping so it
        correctly skips avhubert/whisper/mpd/msd regardless of how many 'module.'
        layers the DDP wrapper adds.
        """
        self._ema_state = {}
        for name, param in model.named_parameters():
            clean = _ema_strip_module(name)
            if any(clean.startswith(p) for p in _EMA_EXCLUDE_PREFIXES):
                continue
            self._ema_state[clean] = param.data.clone()
        self._ema_initialized = True
        n = len(self._ema_state)
        mb = sum(v.numel() * v.element_size() for v in self._ema_state.values()) / 1e6
        logger.info(f"[SynthVC EMA] Initialized {n} tensors ({mb:.0f} MB, decay={_EMA_DECAY})")

    def _update_ema(self, model):
        """Update EMA shadow copy using canonical (stripped) parameter names.

        Strips 'module.' prefixes before looking up in self._ema_state so the update
        works correctly regardless of DDP wrapping depth.  Frozen params (requires_grad
        =False) are still updated so the EMA converges to their frozen values over time.
        """
        if not self._ema_initialized:
            self._init_ema(model)
            return  # first call: capture current weights, don't blend yet
        with torch.no_grad():
            for name, param in model.named_parameters():
                clean = _ema_strip_module(name)
                if clean in self._ema_state:
                    self._ema_state[clean].mul_(_EMA_DECAY).add_(
                        param.data, alpha=1.0 - _EMA_DECAY
                    )

    def forward(self, model, sample, reduce=True):
        """
        Forward with phase-aware loss computation.

        Phase 1 (disc_active=False): mel loss + conversion loss
        Phase 2 (disc_active=True):  mel loss + GAN losses (no conversion loss)
        """
        self._lazy_init(model, next(model.parameters()).device)

        # Determine if disc is active
        disc_active = self._is_disc_active(model)

        # Freeze upstream modules the first time Phase 2 activates
        if disc_active:
            model._freeze_for_phase2()

        subset_name = sample.get("subset_name", "train")

        # =====================================================================
        # Pass spk_embeddings, synth_audio, and disc_active into the model
        # =====================================================================
        net_input = dict(sample["net_input"])
        source = dict(net_input.get("source", {}))

        # Inject speaker embeddings into source dict
        if "spk_embeddings" in sample:
            source["spk_embeddings"] = sample["spk_embeddings"].to(next(model.parameters()).device)

        # Inject synthetic audio and its lengths into source dict
        if "synth_audio" in sample:
            source["synth_audio"] = sample["synth_audio"].to(next(model.parameters()).device)
        if "synth_audio_lengths" in sample:
            source["synth_audio_lengths"] = sample["synth_audio_lengths"].to(next(model.parameters()).device)

        # Inject disc_active flag so the model knows which forward mode to use
        source["disc_active"] = disc_active

        net_input["source"] = source
        sample["net_input"] = net_input

        # =====================================================================
        # Model forward
        # =====================================================================
        net_output = model(**sample["net_input"])
        pred_wav = net_output["waveform"]  # (B, 1, T_pred)
        canonical_features = net_output.get("canonical_features")  # (B, T, 512) or None
        target_features = net_output.get("target_features")        # (B, T, 512) or None

        # =====================================================================
        # Mel loss (always computed: predicted waveform vs canonical ground truth)
        # Phase 2 uses clean waveform (no noise), Phase 1 uses potentially noisy
        # =====================================================================
        if disc_active and "target_waveform_clean" in sample:
            gt_wav = sample["target_waveform_clean"].to(pred_wav.device)
        else:
            gt_wav = sample["target_waveform"].to(pred_wav.device)
        if gt_wav.dim() == 2:
            gt_wav = gt_wav.unsqueeze(1)

        min_len = min(pred_wav.size(-1), gt_wav.size(-1))
        pred_wav = pred_wav[..., :min_len]
        gt_wav = gt_wav[..., :min_len]

        B = pred_wav.size(0)

        if self.use_multires_mel and self.multires_mel is not None:
            # Multi-resolution mel loss (3 scales averaged)
            loss_mel, mel_pred, mel_gt = self.multires_mel(pred_wav, gt_wav)
        else:
            # Single-resolution mel loss
            with torch.no_grad():
                mel_gt = self.logmel(gt_wav)
            mel_pred = self.logmel(pred_wav)

            mel_min_len = min(mel_pred.size(-1), mel_gt.size(-1))
            mel_pred = mel_pred[..., :mel_min_len]
            mel_gt = mel_gt[..., :mel_min_len]

            loss_mel = F.l1_loss(mel_pred, mel_gt)

        # =====================================================================
        # Training: Phase-aware loss computation
        # =====================================================================
        if model.training:
            # Increment update counter
            self._num_updates += 1

            # EMA update: on-GPU, same dtype as model — no PCIe transfer, no CUDA sync
            self._update_ema(model)

            if disc_active:
                # =============================================================
                # PHASE 2: Standard HiFi-GAN adversarial training
                # No conversion loss. Waveform comes from synthetic pass.
                # =============================================================

                # Track when disc phase started — use model.num_updates (fairseq optimizer
                # steps, preserved across resumes) so _disc_active_since stays valid.
                model_updates = getattr(model, 'num_updates', 0)
                if self._disc_active_since is None:
                    self._disc_active_since = model_updates
                    logger.info(f"[SynthVC Criterion] Disc phase started at model_updates={model_updates}")

                # cuFFT (used by spectral disc) does not support BFloat16.
                # Only spectral disc needs float32 input; MPD stays in bf16.
                gt_wav_f32 = gt_wav.float()
                pred_wav_f32 = pred_wav.float()

                # Discriminator step
                self.disc_optimizer.zero_grad()
                mpd_real_scores, _ = model.mpd(gt_wav)
                msstftd_real_scores, _ = self._spec_disc(gt_wav_f32)
                mpd_fake_scores, _ = model.mpd(pred_wav.detach())
                msstftd_fake_scores, _ = self._spec_disc(pred_wav_f32.detach())

                loss_disc_mpd, _, _ = discriminator_loss(mpd_real_scores, mpd_fake_scores)
                loss_disc_msstftd, _, _ = discriminator_loss(msstftd_real_scores, msstftd_fake_scores)
                loss_disc = loss_disc_mpd + loss_disc_msstftd
                loss_disc.backward()

                # Gradient clipping for discriminator
                if self.disc_grad_clip > 0:
                    disc_params = list(model.mpd.parameters()) + list(self._spec_disc.parameters())
                    torch.nn.utils.clip_grad_norm_(disc_params, self.disc_grad_clip)

                self.disc_optimizer.step()
                # Step cosine LR scheduler once per optimizer (num_updates) step, not per
                # criterion call — this keeps the schedule aligned with num_updates units.
                if self.disc_lr_scheduler is not None and model_updates > getattr(self, "_disc_sched_last_update", -1):
                    self.disc_lr_scheduler.step()
                    self._disc_sched_last_update = model_updates

                # Generator step
                mpd_real_scores, mpd_real_feats = model.mpd(gt_wav)
                msstftd_real_scores, msstftd_real_feats = self._spec_disc(gt_wav_f32)
                mpd_fake_scores, mpd_fake_feats = model.mpd(pred_wav)
                msstftd_fake_scores, msstftd_fake_feats = self._spec_disc(pred_wav_f32)

                loss_fm = feature_loss(mpd_real_feats, mpd_fake_feats) + feature_loss(msstftd_real_feats, msstftd_fake_feats)
                loss_gen_mpd, _ = generator_loss(mpd_fake_scores)
                loss_gen_msstftd, _ = generator_loss(msstftd_fake_scores)
                loss_gen_adv = loss_gen_mpd + loss_gen_msstftd

                # Adversarial warmup: linearly ramp adv weight from 0→1.
                # Uses model.num_updates (not local _num_updates) so warmup position
                # is correctly restored across resumes.
                # _adv_warmup_complete flag persists across resumes so warmup doesn't
                # restart every 72h session once it has completed.
                adv_weight = 1.0
                if self.adv_warmup_updates > 0 and not self._adv_warmup_complete:
                    steps_since_disc = model_updates - self._disc_active_since
                    adv_weight = min(1.0, max(0.0, steps_since_disc / self.adv_warmup_updates))
                    if adv_weight >= 1.0:
                        self._adv_warmup_complete = True
                        logger.info("[SynthVC Criterion] Adversarial warmup complete")

                loss_gen = (self.mel_loss_weight * loss_mel
                            + adv_weight * (loss_fm + loss_gen_adv))

                logging_output = {
                    "loss": loss_gen.item(),
                    "loss_mel": loss_mel.item(),
                    "loss_conv": 0.0,
                    "loss_mel_weighted": (self.mel_loss_weight * loss_mel).item(),
                    "loss_conv_weighted": 0.0,
                    "loss_fm": loss_fm.item(),
                    "loss_gen_adv": loss_gen_adv.item(),
                    "loss_disc": loss_disc.item(),
                    "adv_weight": adv_weight,
                    "disc_active": 1,
                    "num_updates": max(self._num_updates, getattr(model, "num_updates", 0) * 4),
                    "sample_size": B,
                    "nsentences": B,
                }
                return loss_gen, B, logging_output

            else:
                # =============================================================
                # PHASE 1: Mel + Conversion only (no discriminator for generator)
                # =============================================================
                loss_conv = torch.tensor(0.0, device=pred_wav.device)
                if canonical_features is not None and target_features is not None:
                    min_t = min(canonical_features.size(1), target_features.size(1))
                    loss_conv = F.l1_loss(
                        target_features[:, :min_t, :],
                        canonical_features[:, :min_t, :].detach()
                    )

                loss_gen = self.mel_loss_weight * loss_mel + self.conv_loss_weight * loss_conv

                # =============================================================
                # PHASE 1 DISC PRE-TRAINING: Train discriminator as a classifier
                # pred_wav is detached — no adversarial gradient reaches the generator.
                # The disc optimizer is stepped but the LR scheduler is NOT —
                # cosine decay budget is reserved for Phase 2 adversarial training.
                # =============================================================
                loss_disc_pretrain = 0.0
                if self.disc_pretrain:
                    # cuFFT (used by MS-STFT disc) does not support BFloat16.
                    # Only spectral disc needs float32 input; MPD stays in bf16.
                    gt_wav_f32 = gt_wav.float()
                    pred_wav_f32 = pred_wav.float()
                    self.disc_optimizer.zero_grad()
                    mpd_real_scores, _ = model.mpd(gt_wav)
                    msstftd_real_scores, _ = self._spec_disc(gt_wav_f32)
                    mpd_fake_scores, _ = model.mpd(pred_wav.detach())
                    msstftd_fake_scores, _ = self._spec_disc(pred_wav_f32.detach())

                    loss_disc_mpd, _, _ = discriminator_loss(mpd_real_scores, mpd_fake_scores)
                    loss_disc_msstftd, _, _ = discriminator_loss(msstftd_real_scores, msstftd_fake_scores)
                    loss_disc_pretrain_t = loss_disc_mpd + loss_disc_msstftd
                    loss_disc_pretrain_t.backward()

                    if self.disc_grad_clip > 0:
                        disc_params = list(model.mpd.parameters()) + list(self._spec_disc.parameters())
                        torch.nn.utils.clip_grad_norm_(disc_params, self.disc_grad_clip)

                    self.disc_optimizer.step()
                    loss_disc_pretrain = loss_disc_pretrain_t.item()

                logging_output = {
                    "loss": loss_gen.item(),
                    "loss_mel": loss_mel.item(),
                    "loss_conv": loss_conv.item(),
                    "loss_mel_weighted": (self.mel_loss_weight * loss_mel).item(),
                    "loss_conv_weighted": (self.conv_loss_weight * loss_conv).item(),
                    "loss_fm": 0.0,
                    "loss_gen_adv": 0.0,
                    "loss_disc": loss_disc_pretrain,
                    "disc_active": 0,
                    "num_updates": max(self._num_updates, getattr(model, "num_updates", 0) * 4),
                    "sample_size": B,
                    "nsentences": B,
                }
                return loss_gen, B, logging_output
        else:
            # =================================================================
            # Validation
            # =================================================================
            loss_conv = torch.tensor(0.0, device=pred_wav.device)
            if canonical_features is not None and target_features is not None:
                min_t = min(canonical_features.size(1), target_features.size(1))
                loss_conv = F.l1_loss(
                    target_features[:, :min_t, :],
                    canonical_features[:, :min_t, :].detach()
                )

            loss_fm = torch.tensor(0.0, device=pred_wav.device)
            loss_gen_adv = torch.tensor(0.0, device=pred_wav.device)
            loss_disc = torch.tensor(0.0, device=pred_wav.device)

            if disc_active and subset_name == 'valid_synth':
                with torch.no_grad():
                    gt_wav_f32 = gt_wav.float()
                    pred_wav_f32 = pred_wav.float()
                    mpd_real_scores, mpd_real_feats = model.mpd(gt_wav)
                    msstftd_real_scores, msstftd_real_feats = self._spec_disc(gt_wav_f32)
                    mpd_fake_scores, mpd_fake_feats = model.mpd(pred_wav)
                    msstftd_fake_scores, msstftd_fake_feats = self._spec_disc(pred_wav_f32)

                    loss_fm = feature_loss(mpd_real_feats, mpd_fake_feats) + feature_loss(msstftd_real_feats, msstftd_fake_feats)
                    loss_gen_mpd, _ = generator_loss(mpd_fake_scores)
                    loss_gen_msstftd, _ = generator_loss(msstftd_fake_scores)
                    loss_gen_adv = loss_gen_mpd + loss_gen_msstftd

                    loss_disc_mpd, _, _ = discriminator_loss(mpd_real_scores, mpd_fake_scores)
                    loss_disc_msstftd, _, _ = discriminator_loss(msstftd_real_scores, msstftd_fake_scores)
                    loss_disc = loss_disc_mpd + loss_disc_msstftd

            loss_gen = self.mel_loss_weight * loss_mel + self.conv_loss_weight * loss_conv

            logging_output = {
                "loss": loss_gen.item(),
                "loss_mel": loss_mel.item(),
                "loss_conv": loss_conv.item(),
                "loss_mel_weighted": (self.mel_loss_weight * loss_mel).item(),
                "loss_conv_weighted": (self.conv_loss_weight * loss_conv).item(),
                "loss_fm": loss_fm.item(),
                "loss_gen_adv": loss_gen_adv.item(),
                "loss_disc": loss_disc.item(),
                "disc_active": 1 if disc_active else 0,
                "num_updates": max(self._num_updates, getattr(model, "num_updates", 0) * 4),
                "sample_size": B,
                "nsentences": B,
            }

            from .criterionSpeech import compute_mcd, compute_ssim
            with torch.no_grad():
                if self.use_multires_mel and self.multires_mel is not None:
                    # Per-scale validation: compute SSIM/MCD at each of the 3 resolutions
                    scale_names = ["fine", "medium", "coarse"]
                    for logmel_fn, name in zip(self.multires_mel.logmels, scale_names):
                        mp = logmel_fn(pred_wav)
                        mg = logmel_fn(gt_wav)
                        min_t = min(mp.size(-1), mg.size(-1))
                        mp = mp[..., :min_t]
                        mg = mg[..., :min_t]
                        mp_bt = mp.transpose(1, 2).float()
                        mg_bt = mg.transpose(1, 2).float()
                        try:
                            logging_output[f"mcd_{name}"] = compute_mcd(mp_bt, mg_bt).item()
                        except Exception:
                            logging_output[f"mcd_{name}"] = 0.0
                        try:
                            logging_output[f"ssim_{name}"] = compute_ssim(mp_bt, mg_bt).item()
                        except Exception:
                            logging_output[f"ssim_{name}"] = 0.0
                else:
                    # Single-resolution validation
                    mel_pred_bt = mel_pred.transpose(1, 2).float()
                    mel_gt_bt = mel_gt.transpose(1, 2).float()
                    try:
                        logging_output["mcd"] = compute_mcd(mel_pred_bt, mel_gt_bt).item()
                    except Exception:
                        logging_output["mcd"] = 0.0
                    try:
                        logging_output["ssim"] = compute_ssim(mel_pred_bt, mel_gt_bt).item()
                    except Exception:
                        logging_output["ssim"] = 0.0
                logging_output["val_mel_loss"] = loss_mel.item()

            return loss_gen, B, logging_output

    def state_dict(self):
        """Save disc_active_since, adv_warmup_complete, EMA state, and disc LR scheduler."""
        state = super().state_dict()
        # disc_active_since is in model update units (preserved across resumes)
        if self._disc_active_since is not None:
            state["disc_active_since_v2"] = self._disc_active_since
        state["adv_warmup_complete"] = self._adv_warmup_complete
        # EMA: move to CPU for checkpoint (device-agnostic storage)
        if self._ema_initialized and self._ema_state:
            state["ema_state"] = {k: v.cpu() for k, v in self._ema_state.items()}
        # Disc LR scheduler
        if self.disc_lr_scheduler is not None:
            state["disc_lr_scheduler_state"] = self.disc_lr_scheduler.state_dict()
        state["disc_sched_last_update"] = getattr(self, "_disc_sched_last_update", -1)
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Restore criterion state from checkpoint.

        The disc_optimizer doesn't exist yet at checkpoint load time (lazy init),
        so we stash its state and apply it later in _lazy_init.
        Handles None state_dict gracefully (old checkpoints before dummy param fix).
        """
        if state_dict is None:
            logger.warning("[SynthVC Criterion] No criterion state in checkpoint "
                         "(first resume after checkpoint fix — disc_optimizer state not available)")
            return
        # disc_active_since_v2: in model update units (current format)
        disc_active_since_v2 = state_dict.pop("disc_active_since_v2", None)
        # disc_active_since (old key): was in criterion-call units, which reset to 0 on
        # each resume and caused negative adv_weight.  Discard it — warmup will restart.
        state_dict.pop("disc_active_since", None)
        adv_warmup_complete = state_dict.pop("adv_warmup_complete", False)
        # Intercept disc_optimizer and scheduler states for deferred loading in _lazy_init
        pending_disc = state_dict.pop("disc_optimizer_state", None)
        if pending_disc is not None:
            self._pending_disc_optimizer_state = pending_disc
            logger.info("[SynthVC Criterion] Stashed disc_optimizer state for deferred loading")
        pending_sched = state_dict.pop("disc_lr_scheduler_state", None)
        if pending_sched is not None:
            self._pending_disc_scheduler_state = pending_sched
            logger.info("[SynthVC Criterion] Stashed disc_lr_scheduler state for deferred loading")
        self._disc_sched_last_update = state_dict.pop("disc_sched_last_update", -1)
        ema_state = state_dict.pop("ema_state", None)
        # Use strict=False to tolerate keys from older checkpoints (e.g. logmel/multires_mel
        # buffers that were saved by a previous code version but are no longer registered).
        super().load_state_dict(state_dict, strict=False)
        if disc_active_since_v2 is not None:
            self._disc_active_since = disc_active_since_v2
            logger.info(f"[SynthVC Criterion] Restored _disc_active_since={self._disc_active_since} (model updates)")
        self._adv_warmup_complete = adv_warmup_complete
        if adv_warmup_complete:
            logger.info("[SynthVC Criterion] Adversarial warmup already complete (loaded from checkpoint)")
        if ema_state is not None:
            # Strip any leading 'module.' prefixes from old checkpoints (saved before the
            # canonical-key fix). New checkpoints already store clean keys; stripping is a no-op.
            cleaned_ema = {_ema_strip_module(k): v for k, v in ema_state.items()}
            self._ema_state = cleaned_ema
            self._ema_initialized = True
            logger.info(f"[SynthVC Criterion] Restored EMA state ({len(cleaned_ema)} tensors, on CPU until _lazy_init)")

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        n_batches = len(logging_outputs)
        if n_batches == 0:
            return

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / n_batches, priority=100, round=4)

        for key in ["loss_mel", "loss_conv", "loss_mel_weighted", "loss_conv_weighted", "loss_fm", "loss_gen_adv", "loss_disc"]:
            val_sum = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(key, val_sum / n_batches, priority=90, round=4)

        # Log disc_active and num_updates for monitoring
        disc_active_vals = [log.get("disc_active", 0) for log in logging_outputs]
        if disc_active_vals:
            metrics.log_scalar("disc_active", sum(disc_active_vals) / len(disc_active_vals), priority=85, round=0)
        num_updates_vals = [log.get("num_updates", 0) for log in logging_outputs]
        if num_updates_vals:
            metrics.log_scalar("criterion_updates", max(num_updates_vals), priority=84, round=0)

        adv_weight_vals = [log.get("adv_weight", -1) for log in logging_outputs if log.get("adv_weight", -1) >= 0]
        if adv_weight_vals:
            metrics.log_scalar("adv_weight", sum(adv_weight_vals) / len(adv_weight_vals), priority=83, round=4)

        # Single-res metrics
        for key, priority in [("mcd", 80), ("ssim", 70), ("val_mel_loss", 60)]:
            values = [log[key] for log in logging_outputs if key in log]
            if values:
                metrics.log_scalar(key, sum(values) / len(values), priority=priority, round=4)

        # Multi-res per-scale metrics
        for scale in ["fine", "medium", "coarse"]:
            for key_base, priority in [("mcd", 79), ("ssim", 69)]:
                key = f"{key_base}_{scale}"
                values = [log[key] for log in logging_outputs if key in log]
                if values:
                    metrics.log_scalar(key, sum(values) / len(values), priority=priority, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
