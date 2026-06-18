"""
MelVC criterion: pure mel-reconstruction (L1) against BigVGAN-format target mels.

No discriminator, no MR-STFT, no adversarial — the model emits mels directly, so
the only loss is L1 between predicted and ground-truth mel. The GT mel is computed
ON THE FLY: target waveform (source SR) -> resample to BigVGAN SR -> exact BigVGAN
mel. The GT mel's time dimension is fed back to the model as the interpolation
target, guaranteeing predicted/GT alignment.

Validation metrics (computed directly on the mels, no waveform):
  - ssim_healthy: global SSIM between predicted and GT log-mel.
  - mcd_healthy : mel-cepstral distortion from DCT of the log-mels.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

from .bigvgan_mel import mel_spectrogram, resample


@dataclass
class MelVCCriterionConfig(FairseqDataclass):
    mel_loss_weight: float = field(default=1.0)
    source_sr: int = field(default=16000)
    target_sr: int = field(default=22050)
    mel_bands: int = field(default=80)
    n_fft: int = field(default=1024)
    hop_size: int = field(default=256)
    win_size: int = field(default=1024)
    fmin: int = field(default=0)
    fmax: int = field(default=8000)


@register_criterion("melvc_l1", dataclass=MelVCCriterionConfig)
class MelVCCriterion(FairseqCriterion):
    def __init__(self, task, mel_loss_weight, source_sr, target_sr, mel_bands,
                 n_fft, hop_size, win_size, fmin, fmax):
        super().__init__(task)
        self.w = mel_loss_weight
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.hop = hop_size
        self.mel_cfg = dict(sampling_rate=target_sr, num_mels=mel_bands, n_fft=n_fft,
                            hop_size=hop_size, win_size=win_size, fmin=fmin, fmax=fmax)
        self._dct = None  # lazy (B, bands, bands) DCT-II basis for MCD

    # ---------- helpers ----------
    def _gt_mel(self, gt_wav):
        wav_t = resample(gt_wav, self.source_sr, self.target_sr)
        return mel_spectrogram(wav_t, **self.mel_cfg)  # (B, bands, T_mel)

    @staticmethod
    def _global_ssim(x, y, mask):
        """Global SSIM per sample over valid frames. x,y: (B, bands, T); mask: (B,1,T)."""
        m = mask.expand_as(x)
        n = m.sum(dim=(1, 2)).clamp(min=1)
        mx = (x * m).sum(dim=(1, 2)) / n
        my = (y * m).sum(dim=(1, 2)) / n
        vx = ((x - mx[:, None, None]) ** 2 * m).sum(dim=(1, 2)) / n
        vy = ((y - my[:, None, None]) ** 2 * m).sum(dim=(1, 2)) / n
        cov = ((x - mx[:, None, None]) * (y - my[:, None, None]) * m).sum(dim=(1, 2)) / n
        L = 14.0  # approx dynamic range of log-mel
        c1 = (0.01 * L) ** 2
        c2 = (0.03 * L) ** 2
        ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / \
               ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2) + 1e-8)
        return ssim.mean().item()

    def _mcd(self, x, y, mask, device):
        """Mel-cepstral distortion (dB) via DCT-II of the log-mels. x,y: (B,bands,T)."""
        bands = x.size(1)
        if self._dct is None or self._dct.size(0) != bands:
            n = torch.arange(bands, dtype=torch.float32)
            k = n.unsqueeze(1)
            basis = torch.cos(math.pi / bands * (n.unsqueeze(0) + 0.5) * k)  # (bands, bands)
            self._dct = basis.to(device)
        # DCT over the BANDS dim: basis (K, M) x mel (B, M, T) -> cepstra (B, K, T).
        cx = torch.einsum("km,bmt->bkt", self._dct, x)
        cy = torch.einsum("km,bmt->bkt", self._dct, y)
        # drop c0 (energy), standard MCD
        diff2 = (cx[:, 1:, :] - cy[:, 1:, :]) ** 2  # (B, bands-1, T)
        diff2 = (diff2 * mask).sum(dim=1)           # (B, T) sum over coeffs
        valid = mask.squeeze(1)                     # (B, T)
        per_frame = torch.sqrt(diff2.clamp(min=0))
        mcd = (10.0 * math.sqrt(2) / math.log(10)) * \
              (per_frame * valid).sum() / valid.sum().clamp(min=1)
        return mcd.item()

    # ---------- forward ----------
    def forward(self, model, sample, reduce=True):
        device = next(model.parameters()).device
        net_input = dict(sample["net_input"])
        source = dict(net_input.get("source", {}))
        if "spk_embeddings" in sample:
            source["spk_embeddings"] = sample["spk_embeddings"].to(device)

        gt_wav = sample["target_waveform"].to(device).float()      # (B, T) @ source_sr
        wav_lens = sample["waveform_lengths"].to(device)
        with torch.no_grad():
            gt_mel = self._gt_mel(gt_wav)                          # (B, bands, T_mel)
        T_mel = gt_mel.size(-1)

        mel_lengths = torch.clamp(
            ((wav_lens * self.target_sr) // self.source_sr - self.hop) // self.hop + 1,
            min=1, max=T_mel)
        source["mel_target_lengths"] = mel_lengths
        net_input["source"] = source

        net_output = model(**net_input)
        # Model runs in bf16, but the GT mel is float32 (torch.stft has no bf16
        # support), so compare in float32: stable loss + metric ops (einsum/SSIM)
        # that don't auto-promote bf16. Gradients still flow to the bf16 params.
        pred_mel = net_output["melspec"].transpose(1, 2).float()   # (B, bands, T_pred)

        m = min(pred_mel.size(-1), gt_mel.size(-1))
        pred_mel = pred_mel[..., :m]
        gt_mel = gt_mel[..., :m]
        idx = torch.arange(m, device=device).unsqueeze(0)
        mask = (idx < mel_lengths.clamp(max=m).unsqueeze(1)).float().unsqueeze(1)  # (B,1,m)

        denom = mask.sum() * pred_mel.size(1) + 1e-8
        l1 = (torch.abs(pred_mel - gt_mel) * mask).sum() / denom
        loss = self.w * l1

        with torch.no_grad():
            ssim = self._global_ssim(pred_mel, gt_mel, mask)
            mcd = self._mcd(pred_mel, gt_mel, mask, device)

        B = gt_wav.size(0)
        rr = net_output.get("residual_ratio")
        logging_output = {
            "loss": loss.detach().item(),
            "loss_mel": l1.detach().item(),
            "ssim_healthy": ssim,
            "mcd_healthy": mcd,
            "nsentences": B,
            "sample_size": B,
        }
        if rr is not None:
            logging_output["residual_ratio"] = float(rr)
        return loss, B, logging_output

    # ---------- metrics ----------
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        n = max(len(logging_outputs), 1)
        for key, prio in [("loss", 100), ("loss_mel", 95), ("ssim_healthy", 90),
                          ("mcd_healthy", 90), ("residual_ratio", 80)]:
            vals = [lo[key] for lo in logging_outputs if key in lo]
            if vals:
                metrics.log_scalar(key, sum(vals) / len(vals), priority=prio, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
