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


class MultiResolutionMelLoss(nn.Module):
    """Compute mel L1 loss at multiple (n_fft, hop_size) resolutions.

    Default resolutions follow BigVGAN / EnCodec conventions adapted for 16kHz.
    Returns the sum of L1 across all resolutions (standard practice in BigVGAN/EnCodec).
    """
    def __init__(self, resolutions=None, num_mels=80, sample_rate=16000):
        super().__init__()
        if resolutions is None:
            # (n_fft, hop_size, win_size) — covers fine, medium, and coarse scales
            resolutions = [
                (512, 120, 512),
                (1024, 160, 1024),
                (2048, 480, 2048),
            ]
        self.logmels = nn.ModuleList([
            LogMelSpectrogram(n_fft=n, num_mels=num_mels, hop_size=h, win_size=w, sample_rate=sample_rate)
            for n, h, w in resolutions
        ])

    def forward(self, pred_wav, gt_wav):
        """
        Args:
            pred_wav: (B, 1, T) predicted waveform (graph attached)
            gt_wav:   (B, 1, T) ground-truth waveform
        Returns:
            loss: scalar, mean of L1 across resolutions
            mel_pred_primary: mel from the second (primary) resolution, for logging/metrics
            mel_gt_primary: corresponding ground-truth mel
        """
        total_loss = 0.0
        mel_pred_primary = mel_gt_primary = None
        for i, logmel in enumerate(self.logmels):
            with torch.no_grad():
                mg = logmel(gt_wav)
            mp = logmel(pred_wav)
            min_t = min(mp.size(-1), mg.size(-1))
            mp = mp[..., :min_t]
            mg = mg[..., :min_t]
            total_loss = total_loss + F.l1_loss(mp, mg)
            # Keep the middle resolution for MCD/SSIM metrics
            if i == 1:
                mel_pred_primary = mp
                mel_gt_primary = mg
        return total_loss / len(self.logmels), mel_pred_primary, mel_gt_primary


@dataclass
class E2EGanLossConfig(FairseqDataclass):
    mel_loss_weight: float = field(
        default=45.0, metadata={"help": "Weight for mel reconstruction loss"}
    )
    use_discriminator: bool = field(
        default=True, metadata={"help": "Enable discriminator training (set False for mel-only phase)"}
    )
    disc_lr: float = field(
        default=2e-4, metadata={"help": "Discriminator learning rate"}
    )
    disc_betas: str = field(
        default="0.8,0.99", metadata={"help": "Discriminator Adam betas (comma-separated)"}
    )
    mel_num_mels: int = field(
        default=128, metadata={"help": "Number of mel filter banks"}
    )
    mel_hop_size: int = field(
        default=160, metadata={"help": "Mel spectrogram hop size in samples (160=100Hz, 320=50Hz at 16kHz)"}
    )


@register_criterion("e2e_gan_loss", dataclass=E2EGanLossConfig)
class E2EGanLoss(FairseqCriterion):
    def __init__(self, task, mel_loss_weight=45.0, use_discriminator=True, disc_lr=2e-4, disc_betas="0.8,0.99",
                 mel_num_mels=128, mel_hop_size=160):
        super().__init__(task)
        self.mel_loss_weight = mel_loss_weight
        self.use_discriminator = use_discriminator
        self.disc_lr = disc_lr
        self.disc_betas = tuple(float(x) for x in disc_betas.split(","))
        self.mel_num_mels = mel_num_mels
        self.mel_hop_size = mel_hop_size

        self.logmel = None  # Lazy init on first forward (needs device)
        self.disc_optimizer = None  # Lazy init (needs model reference)

        logger.info(f"[E2E Criterion] use_discriminator={self.use_discriminator}, "
                    f"mel_num_mels={self.mel_num_mels}, mel_hop_size={self.mel_hop_size}")
    
    def _lazy_init(self, model, device):
        """Initialize LogMelSpectrogram and discriminator optimizer on first call."""
        if self.logmel is None:
            self.logmel = LogMelSpectrogram(num_mels=self.mel_num_mels, hop_size=self.mel_hop_size).to(device)
        
        if self.disc_optimizer is None and self.use_discriminator:
            # Detect which spectral discriminator is available (CQT or MS-STFT)
            self._use_cqt = hasattr(model, 'cqtd')
            self._spec_disc = model.cqtd if self._use_cqt else model.msstftd
            self._spec_disc.float()  # cuFFT does not support BFloat16
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
            if self.use_discriminator:
                # =============================================================
                # TRAINING: Full GAN loop
                # =============================================================
                
                # --- Discriminator step ---
                self.disc_optimizer.zero_grad()
                
                mpd_real_scores, _ = model.mpd(gt_wav)
                msstftd_real_scores, _ = self._spec_disc(gt_wav)

                mpd_fake_scores, _ = model.mpd(pred_wav.detach())
                msstftd_fake_scores, _ = self._spec_disc(pred_wav.detach())

                loss_disc_mpd, _, _ = discriminator_loss(mpd_real_scores, mpd_fake_scores)
                loss_disc_msstftd, _, _ = discriminator_loss(msstftd_real_scores, msstftd_fake_scores)
                loss_disc = loss_disc_mpd + loss_disc_msstftd

                loss_disc.backward()
                self.disc_optimizer.step()

                # --- Generator step ---
                mpd_real_scores, mpd_real_feats = model.mpd(gt_wav)
                msstftd_real_scores, msstftd_real_feats = self._spec_disc(gt_wav)

                mpd_fake_scores, mpd_fake_feats = model.mpd(pred_wav)
                msstftd_fake_scores, msstftd_fake_feats = self._spec_disc(pred_wav)

                loss_fm_mpd = feature_loss(mpd_real_feats, mpd_fake_feats)
                loss_fm_msstftd = feature_loss(msstftd_real_feats, msstftd_fake_feats)
                loss_fm = loss_fm_mpd + loss_fm_msstftd

                loss_gen_mpd, _ = generator_loss(mpd_fake_scores)
                loss_gen_msstftd, _ = generator_loss(msstftd_fake_scores)
                loss_gen_adv = loss_gen_mpd + loss_gen_msstftd
                
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
                # =============================================================
                # TRAINING: Mel-only (Phase 1 — no discriminator)
                # =============================================================
                loss_gen = loss_mel
                
                logging_output = {
                    "loss": loss_gen.item(),
                    "loss_mel": loss_mel.item(),
                    "loss_fm": 0.0,
                    "loss_gen_adv": 0.0,
                    "loss_disc": 0.0,
                    "sample_size": B,
                    "nsentences": B,
                }
                
                return loss_gen, B, logging_output
        
        else:
            # =================================================================
            # VALIDATION: Only mel loss + metrics (no GAN ops)
            # =================================================================
            loss_gen = loss_mel
            
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

    def state_dict(self):
        """Override to save the discriminator optimizer and update states."""
        state = super().state_dict()
        if self.disc_optimizer is not None:
            state["disc_optimizer_state"] = self.disc_optimizer.state_dict()
        if hasattr(self, "_num_updates"):
            state["criterion_num_updates"] = self._num_updates
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Override to load the discriminator optimizer and update states."""
        disc_state = state_dict.pop("disc_optimizer_state", None)
        num_updates = state_dict.pop("criterion_num_updates", None)
        super().load_state_dict(state_dict, strict)
        
        if disc_state is not None and self.disc_optimizer is not None:
            self.disc_optimizer.load_state_dict(disc_state)
            logger.info("[E2E Criterion] Successfully loaded disc_optimizer state")
            
        if num_updates is not None and hasattr(self, "_num_updates"):
            self._num_updates = num_updates
            logger.info(f"[E2E Criterion] Successfully loaded _num_updates={self._num_updates}")

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
