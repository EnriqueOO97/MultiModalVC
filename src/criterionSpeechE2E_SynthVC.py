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
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.dataclass import FairseqDataclass

from .criterionSpeechE2E import E2EGanLoss, E2EGanLossConfig, LogMelSpectrogram, MultiResolutionMelLoss

logger = logging.getLogger("src.criterionSpeechE2E_SynthVC")


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


@register_criterion("e2e_gan_loss_synthvc", dataclass=E2EGanLossSynthVCConfig)
class E2EGanLossSynthVC(E2EGanLoss):
    def __init__(self, task, mel_loss_weight=40.0, use_discriminator=True,
                 disc_lr=2e-4, disc_betas="0.8,0.99", conv_loss_weight=5.0,
                 disc_start_updates=30000, mel_num_mels=128, mel_hop_size=160,
                 disc_grad_clip=0.0, adv_warmup_updates=0, use_multires_mel=False):
        super().__init__(task, mel_loss_weight, use_discriminator, disc_lr, disc_betas,
                         mel_num_mels=mel_num_mels, mel_hop_size=mel_hop_size)
        self.conv_loss_weight = conv_loss_weight
        self.disc_start_updates = disc_start_updates
        self.disc_grad_clip = disc_grad_clip
        self.adv_warmup_updates = adv_warmup_updates
        self.use_multires_mel = use_multires_mel
        self._num_updates = 0
        self._disc_active_since = None  # track when disc phase started
        self.multires_mel = None  # lazy init
        logger.info(f"[SynthVC Criterion] conv_loss_weight={self.conv_loss_weight}, "
                     f"disc_start_updates={self.disc_start_updates}, "
                     f"disc_grad_clip={self.disc_grad_clip}, adv_warmup_updates={self.adv_warmup_updates}, "
                     f"use_multires_mel={self.use_multires_mel}, "
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

        if self.disc_optimizer is None:
            # Collect discriminator params - enable grad for optimizer
            disc_params = []
            for param in model.mpd.parameters():
                param.requires_grad = True
                disc_params.append(param)
            for param in model.msd.parameters():
                param.requires_grad = True
                disc_params.append(param)

            self.disc_optimizer = torch.optim.AdamW(
                disc_params,
                lr=self.disc_lr,
                betas=self.disc_betas,
            )
            logger.info(f"[SynthVC Criterion] Initialized disc optimizer with lr={self.disc_lr}, "
                       f"betas={self.disc_betas}, params={sum(p.numel() for p in disc_params):,}")

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

        # =====================================================================
        # Phase 2 Validation Optimization
        # Skip canonical 'valid' subset during Phase 2 to save resources
        # =====================================================================
        subset_name = sample.get("subset_name", "train")
        if not model.training and disc_active and subset_name == 'valid':
            B = len(sample.get("id", [0]))
            logging_output = {
                "loss": 0.0,
                "loss_mel": 0.0,
                "loss_conv": 0.0,
                "loss_mel_weighted": 0.0,
                "loss_conv_weighted": 0.0,
                "loss_fm": 0.0,
                "loss_gen_adv": 0.0,
                "loss_disc": 0.0,
                "disc_active": 1,
                "num_updates": max(self._num_updates, getattr(model, "num_updates", 0) * 4),
                "sample_size": B,
                "nsentences": B,
                "val_mel_loss": 0.0,
                "mcd": 0.0,
                "ssim": 0.0,
            }
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device), B, logging_output

        # =====================================================================
        # Pass spk_embeddings, synth_audio, and disc_active into the model
        # =====================================================================
        net_input = dict(sample["net_input"])
        source = dict(net_input.get("source", {}))

        # Inject speaker embeddings into source dict
        if "spk_embeddings" in sample:
            source["spk_embeddings"] = sample["spk_embeddings"].to(next(model.parameters()).device)

        # Inject synthetic audio into source dict
        if "synth_audio" in sample:
            source["synth_audio"] = sample["synth_audio"].to(next(model.parameters()).device)

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

            if disc_active:
                # =============================================================
                # PHASE 2: Standard HiFi-GAN adversarial training
                # No conversion loss. Waveform comes from synthetic pass.
                # =============================================================
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_hifigan"))
                from hifigan.discriminator import feature_loss, discriminator_loss, generator_loss

                # Track when disc phase started (for warmup)
                if self._disc_active_since is None:
                    self._disc_active_since = self._num_updates
                    logger.info(f"[SynthVC Criterion] Disc phase started at update {self._num_updates}")

                # Discriminator step
                self.disc_optimizer.zero_grad()
                mpd_real_scores, _ = model.mpd(gt_wav)
                msd_real_scores, _ = model.msd(gt_wav)
                mpd_fake_scores, _ = model.mpd(pred_wav.detach())
                msd_fake_scores, _ = model.msd(pred_wav.detach())

                loss_disc_mpd, _, _ = discriminator_loss(mpd_real_scores, mpd_fake_scores)
                loss_disc_msd, _, _ = discriminator_loss(msd_real_scores, msd_fake_scores)
                loss_disc = loss_disc_mpd + loss_disc_msd
                loss_disc.backward()

                # Gradient clipping for discriminator
                if self.disc_grad_clip > 0:
                    disc_params = list(model.mpd.parameters()) + list(model.msd.parameters())
                    torch.nn.utils.clip_grad_norm_(disc_params, self.disc_grad_clip)

                self.disc_optimizer.step()

                # Generator step
                mpd_real_scores, mpd_real_feats = model.mpd(gt_wav)
                msd_real_scores, msd_real_feats = model.msd(gt_wav)
                mpd_fake_scores, mpd_fake_feats = model.mpd(pred_wav)
                msd_fake_scores, msd_fake_feats = model.msd(pred_wav)

                loss_fm = feature_loss(mpd_real_feats, mpd_fake_feats) + feature_loss(msd_real_feats, msd_fake_feats)
                loss_gen_mpd, _ = generator_loss(mpd_fake_scores)
                loss_gen_msd, _ = generator_loss(msd_fake_scores)
                loss_gen_adv = loss_gen_mpd + loss_gen_msd

                # Adversarial warmup: linearly ramp adv weight from 0→1
                adv_weight = 1.0
                if self.adv_warmup_updates > 0:
                    steps_since_disc = self._num_updates - self._disc_active_since
                    adv_weight = min(1.0, steps_since_disc / self.adv_warmup_updates)

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
                # PHASE 1: Mel + Conversion only (no discriminator)
                # =============================================================
                loss_conv = torch.tensor(0.0, device=pred_wav.device)
                if canonical_features is not None and target_features is not None:
                    min_t = min(canonical_features.size(1), target_features.size(1))
                    loss_conv = F.l1_loss(
                        target_features[:, :min_t, :],
                        canonical_features[:, :min_t, :].detach()
                    )

                loss_gen = self.mel_loss_weight * loss_mel + self.conv_loss_weight * loss_conv

                logging_output = {
                    "loss": loss_gen.item(),
                    "loss_mel": loss_mel.item(),
                    "loss_conv": loss_conv.item(),
                    "loss_mel_weighted": (self.mel_loss_weight * loss_mel).item(),
                    "loss_conv_weighted": (self.conv_loss_weight * loss_conv).item(),
                    "loss_fm": 0.0,
                    "loss_gen_adv": 0.0,
                    "loss_disc": 0.0,
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
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_hifigan"))
                from hifigan.discriminator import feature_loss, discriminator_loss, generator_loss
                with torch.no_grad():
                    mpd_real_scores, mpd_real_feats = model.mpd(gt_wav)
                    msd_real_scores, msd_real_feats = model.msd(gt_wav)
                    mpd_fake_scores, mpd_fake_feats = model.mpd(pred_wav)
                    msd_fake_scores, msd_fake_feats = model.msd(pred_wav)

                    loss_fm = feature_loss(mpd_real_feats, mpd_fake_feats) + feature_loss(msd_real_feats, msd_fake_feats)
                    loss_gen_mpd, _ = generator_loss(mpd_fake_scores)
                    loss_gen_msd, _ = generator_loss(msd_fake_scores)
                    loss_gen_adv = loss_gen_mpd + loss_gen_msd

                    loss_disc_mpd, _, _ = discriminator_loss(mpd_real_scores, mpd_fake_scores)
                    loss_disc_msd, _, _ = discriminator_loss(msd_real_scores, msd_fake_scores)
                    loss_disc = loss_disc_mpd + loss_disc_msd

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
        """Save disc_active_since for warmup resume."""
        state = super().state_dict()
        if self._disc_active_since is not None:
            state["disc_active_since"] = self._disc_active_since
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Restore disc_active_since for warmup resume."""
        disc_active_since = state_dict.pop("disc_active_since", None)
        super().load_state_dict(state_dict, strict)
        if disc_active_since is not None:
            self._disc_active_since = disc_active_since
            logger.info(f"[SynthVC Criterion] Restored _disc_active_since={self._disc_active_since}")

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

        for key, priority in [("mcd", 80), ("ssim", 70), ("val_mel_loss", 60)]:
            values = [log[key] for log in logging_outputs if key in log]
            if values:
                metrics.log_scalar(key, sum(values) / len(values), priority=priority, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
