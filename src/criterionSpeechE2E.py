"""
End-to-End GAN Criterion for Stage 1 + HiFi-GAN.

Training loop:
    1. Model forward → waveform
    2. Discriminator step (internal optimizer): disc_loss.backward() + step()
    3. Generator step: mel_recon + feat_matching + adversarial → returned to fairseq

Validation:
    - Stage 1 metrics: MCD, SSIM (on LogMel spectrograms)
    - HiFi-GAN metrics: mel L1 loss
"""

import math
import os
import sys
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

from .criterionSpeech import compute_mcd, compute_ssim

# HiFi-GAN loss functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_hifigan"))
from hifigan.discriminator import feature_loss, discriminator_loss, generator_loss

logger = logging.getLogger(__name__)


class LogMelSpectrogram(nn.Module):
    """Compute log mel spectrogram from waveform.
    
    Uses the same parameters as trainGermanVocoder.py for consistency.
    """
    def __init__(self, n_fft=1024, num_mels=128, hop_size=160, win_size=1024, sample_rate=16000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            center=False,
            power=2.0,
            norm=None,
            f_min=0,
            f_max=8000,
            onesided=True,
            n_mels=num_mels,
            mel_scale="slaney",
        )

    def forward(self, wav):
        """
        Args:
            wav: (B, 1, T) or (B, T) waveform
        Returns:
            logmel: (B, num_mels, T_mel)
        """
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        pad = (self.n_fft - self.hop_size) // 2
        wav = F.pad(wav, (pad, pad), "reflect")
        mel = self.melspectrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


@dataclass
class E2EGanLossConfig(FairseqDataclass):
    mel_loss_weight: float = field(
        default=45.0, metadata={"help": "Weight for mel reconstruction loss"}
    )
    disc_lr: float = field(
        default=2e-4, metadata={"help": "Discriminator learning rate"}
    )
    disc_betas: str = field(
        default="0.8,0.99", metadata={"help": "Discriminator Adam betas (comma-separated)"}
    )


@register_criterion("e2e_gan_loss", dataclass=E2EGanLossConfig)
class E2EGanLoss(FairseqCriterion):
    def __init__(self, task, mel_loss_weight=45.0, disc_lr=2e-4, disc_betas="0.8,0.99"):
        super().__init__(task)
        self.mel_loss_weight = mel_loss_weight
        self.disc_lr = disc_lr
        self.disc_betas = tuple(float(x) for x in disc_betas.split(","))
        
        self.logmel = None  # Lazy init on first forward (needs device)
        self.disc_optimizer = None  # Lazy init (needs model reference)
    
    def _lazy_init(self, model, device):
        """Initialize LogMelSpectrogram and discriminator optimizer on first call."""
        if self.logmel is None:
            self.logmel = LogMelSpectrogram().to(device)
        
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
            logger.info(f"[E2E Criterion] Initialized disc optimizer with lr={self.disc_lr}, "
                       f"betas={self.disc_betas}, params={sum(p.numel() for p in disc_params):,}")

    def forward(self, model, sample, reduce=True):
        """
        GAN training step.
        
        1. Model forward → waveform
        2. Discriminator step (internal)
        3. Generator losses → returned to fairseq
        """
        self._lazy_init(model, next(model.parameters()).device)
        
        # =====================================================================
        # 1. Model forward → waveform
        # =====================================================================
        net_output = model(**sample["net_input"])
        pred_wav = net_output["waveform"]  # (B, 1, T_pred)
        
        # Get ground-truth waveform
        gt_wav = sample["target_waveform"].to(pred_wav.device)  # (B, T)
        wav_lengths = sample["waveform_lengths"].to(pred_wav.device)  # (B,)
        
        # Ensure shapes match: gt_wav (B, T) → (B, 1, T)
        if gt_wav.dim() == 2:
            gt_wav = gt_wav.unsqueeze(1)
        
        # Align lengths (pred_wav length is determined by upsampling factor)
        min_len = min(pred_wav.size(-1), gt_wav.size(-1))
        pred_wav = pred_wav[..., :min_len]
        gt_wav = gt_wav[..., :min_len]
        
        B = pred_wav.size(0)
        
        # Compute mel spectrograms for both (used in all modes)
        with torch.no_grad():
            mel_gt = self.logmel(gt_wav)
        mel_pred = self.logmel(pred_wav)
        
        # Align mel lengths
        mel_min_len = min(mel_pred.size(-1), mel_gt.size(-1))
        mel_pred = mel_pred[..., :mel_min_len]
        mel_gt = mel_gt[..., :mel_min_len]
        
        loss_mel = F.l1_loss(mel_pred, mel_gt)
        
        if model.training:
            # =================================================================
            # TRAINING: Full GAN loop
            # =================================================================
            
            # --- Discriminator step ---
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
            
            # --- Generator step ---
            mpd_real_scores, mpd_real_feats = model.mpd(gt_wav)
            msd_real_scores, msd_real_feats = model.msd(gt_wav)
            
            mpd_fake_scores, mpd_fake_feats = model.mpd(pred_wav)
            msd_fake_scores, msd_fake_feats = model.msd(pred_wav)
            
            loss_fm_mpd = feature_loss(mpd_real_feats, mpd_fake_feats)
            loss_fm_msd = feature_loss(msd_real_feats, msd_fake_feats)
            loss_fm = loss_fm_mpd + loss_fm_msd
            
            loss_gen_mpd, _ = generator_loss(mpd_fake_scores)
            loss_gen_msd, _ = generator_loss(msd_fake_scores)
            loss_gen_adv = loss_gen_mpd + loss_gen_msd
            
            loss_gen = self.mel_loss_weight * loss_mel + loss_fm + loss_gen_adv
            
            logging_output = {
                "loss": loss_gen.item(),
                "loss_mel": loss_mel.item(),
                "loss_fm": loss_fm.item(),
                "loss_gen_adv": loss_gen_adv.item(),
                "loss_disc": loss_disc.item(),
                "sample_size": B,
                "nsentences": B,
            }
            
            return loss_gen, B, logging_output
        
        else:
            # =================================================================
            # VALIDATION: Only mel loss + metrics (no GAN ops)
            # =================================================================
            loss_gen = self.mel_loss_weight * loss_mel
            
            logging_output = {
                "loss": loss_gen.item(),
                "loss_mel": loss_mel.item(),
                "loss_fm": 0.0,
                "loss_gen_adv": 0.0,
                "loss_disc": 0.0,
                "sample_size": B,
                "nsentences": B,
            }
            
            with torch.no_grad():
                mel_pred_bt = mel_pred.transpose(1, 2).float()
                mel_gt_bt = mel_gt.transpose(1, 2).float()
                
                try:
                    mcd = compute_mcd(mel_pred_bt, mel_gt_bt)
                    logging_output["mcd"] = mcd.item()
                except Exception as e:
                    logger.warning(f"MCD computation failed: {e}")
                    logging_output["mcd"] = 0.0
                
                try:
                    ssim = compute_ssim(mel_pred_bt, mel_gt_bt)
                    logging_output["ssim"] = ssim.item()
                except Exception as e:
                    logger.warning(f"SSIM computation failed: {e}")
                    logging_output["ssim"] = 0.0
                
                logging_output["val_mel_loss"] = loss_mel.item()
            
            return loss_gen, B, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        n_batches = len(logging_outputs)
        if n_batches == 0:
            return
        
        # Primary loss
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / n_batches, priority=100, round=4)
        
        # Component losses
        for key in ["loss_mel", "loss_fm", "loss_gen_adv", "loss_disc"]:
            val_sum = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(key, val_sum / n_batches, priority=90, round=4)
        
        # Validation metrics
        for key, priority in [("mcd", 80), ("ssim", 70), ("val_mel_loss", 60)]:
            values = [log[key] for log in logging_outputs if key in log]
            if values:
                metrics.log_scalar(key, sum(values) / len(values), priority=priority, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
