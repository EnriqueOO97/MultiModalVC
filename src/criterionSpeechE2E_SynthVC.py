"""
SynthVC-Inspired E2E Criterion with Conversion Loss.

Extends E2EGanLoss to add conversion loss:
    Loss = mel_loss_weight * mel_loss + conv_loss_weight * conversion_loss

When use_discriminator=false (Phase 1):
    Loss = mel_loss_weight * mel_loss + conv_loss_weight * L1(canonical_features, target_features)

When use_discriminator=true (Phase 2):
    Loss = mel_loss_weight * mel_loss + conv_loss_weight * conversion_loss + feat_matching + adversarial
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.dataclass import FairseqDataclass

from .criterionSpeechE2E import E2EGanLoss, E2EGanLossConfig, LogMelSpectrogram

logger = logging.getLogger(__name__)


@dataclass
class E2EGanLossSynthVCConfig(E2EGanLossConfig):
    conv_loss_weight: float = field(
        default=5.0, metadata={"help": "Weight for conversion loss (SynthVC uses 45:5 mel:conv ratio)"}
    )


@register_criterion("e2e_gan_loss_synthvc", dataclass=E2EGanLossSynthVCConfig)
class E2EGanLossSynthVC(E2EGanLoss):
    def __init__(self, task, mel_loss_weight=45.0, use_discriminator=True,
                 disc_lr=2e-4, disc_betas="0.8,0.99", conv_loss_weight=5.0):
        super().__init__(task, mel_loss_weight, use_discriminator, disc_lr, disc_betas)
        self.conv_loss_weight = conv_loss_weight
        logger.info(f"[SynthVC Criterion] conv_loss_weight={self.conv_loss_weight}")

    def forward(self, model, sample, reduce=True):
        """
        Forward with conversion loss.
        
        The model returns canonical_features and target_features alongside
        the waveform. We compute conversion_loss = L1(canonical, target)
        and add it to the total loss.
        """
        self._lazy_init(model, next(model.parameters()).device)

        # =====================================================================
        # Pass spk_embeddings and synth_audio into the model via net_input
        # =====================================================================
        net_input = dict(sample["net_input"])
        source = dict(net_input.get("source", {}))

        # Inject speaker embeddings into source dict
        if "spk_embeddings" in sample:
            source["spk_embeddings"] = sample["spk_embeddings"].to(next(model.parameters()).device)

        # Inject synthetic audio into source dict
        if "synth_audio" in sample:
            source["synth_audio"] = sample["synth_audio"].to(next(model.parameters()).device)

        net_input["source"] = source
        sample["net_input"] = net_input

        # =====================================================================
        # Model forward → waveform + canonical/target features
        # =====================================================================
        net_output = model(**sample["net_input"])
        pred_wav = net_output["waveform"]  # (B, 1, T_pred)
        canonical_features = net_output.get("canonical_features")  # (B, T, 512)
        target_features = net_output.get("target_features")        # (B, T, 512) or None

        # =====================================================================
        # Mel loss (from waveform, same as parent)
        # =====================================================================
        gt_wav = sample["target_waveform"].to(pred_wav.device)
        if gt_wav.dim() == 2:
            gt_wav = gt_wav.unsqueeze(1)

        min_len = min(pred_wav.size(-1), gt_wav.size(-1))
        pred_wav = pred_wav[..., :min_len]
        gt_wav = gt_wav[..., :min_len]

        B = pred_wav.size(0)

        with torch.no_grad():
            mel_gt = self.logmel(gt_wav)
        mel_pred = self.logmel(pred_wav)

        mel_min_len = min(mel_pred.size(-1), mel_gt.size(-1))
        mel_pred = mel_pred[..., :mel_min_len]
        mel_gt = mel_gt[..., :mel_min_len]

        loss_mel = F.l1_loss(mel_pred, mel_gt)

        # =====================================================================
        # Conversion loss
        # =====================================================================
        loss_conv = torch.tensor(0.0, device=pred_wav.device)
        if canonical_features is not None and target_features is not None:
            # Both should have the same shape since we use canonical audio lengths
            # for both passes' interpolation
            min_t = min(canonical_features.size(1), target_features.size(1))
            loss_conv = F.l1_loss(
                canonical_features[:, :min_t, :],
                target_features[:, :min_t, :].detach()  # detach target to not backprop through synth path twice
            )

        # =====================================================================
        # Total loss
        # =====================================================================
        if model.training:
            if self.use_discriminator:
                # Full GAN training with conversion loss
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_hifigan"))
                from hifigan.discriminator import feature_loss, discriminator_loss, generator_loss

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

                loss_gen = (self.mel_loss_weight * loss_mel
                            + self.conv_loss_weight * loss_conv
                            + loss_fm + loss_gen_adv)

                logging_output = {
                    "loss": loss_gen.item(),
                    "loss_mel": loss_mel.item(),
                    "loss_conv": loss_conv.item(),
                    "loss_fm": loss_fm.item(),
                    "loss_gen_adv": loss_gen_adv.item(),
                    "loss_disc": loss_disc.item(),
                    "sample_size": B,
                    "nsentences": B,
                }
                return loss_gen, B, logging_output

            else:
                # Phase 1: Mel + Conversion only (no discriminator)
                loss_gen = self.mel_loss_weight * loss_mel + self.conv_loss_weight * loss_conv

                logging_output = {
                    "loss": loss_gen.item(),
                    "loss_mel": loss_mel.item(),
                    "loss_conv": loss_conv.item(),
                    "loss_fm": 0.0,
                    "loss_gen_adv": 0.0,
                    "loss_disc": 0.0,
                    "sample_size": B,
                    "nsentences": B,
                }
                return loss_gen, B, logging_output
        else:
            # Validation
            loss_gen = self.mel_loss_weight * loss_mel + self.conv_loss_weight * loss_conv

            logging_output = {
                "loss": loss_gen.item(),
                "loss_mel": loss_mel.item(),
                "loss_conv": loss_conv.item(),
                "loss_fm": 0.0,
                "loss_gen_adv": 0.0,
                "loss_disc": 0.0,
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

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        n_batches = len(logging_outputs)
        if n_batches == 0:
            return

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / n_batches, priority=100, round=4)

        for key in ["loss_mel", "loss_conv", "loss_fm", "loss_gen_adv", "loss_disc"]:
            val_sum = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(key, val_sum / n_batches, priority=90, round=4)

        for key, priority in [("mcd", 80), ("ssim", 70), ("val_mel_loss", 60)]:
            values = [log[key] for log in logging_outputs if key in log]
            if values:
                metrics.log_scalar(key, sum(values) / len(values), priority=priority, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
