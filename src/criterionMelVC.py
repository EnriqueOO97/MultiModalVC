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
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

from .bigvgan_mel import mel_spectrogram, resample
from .mel_discriminator import disc_loss, gen_adv_loss, feature_matching_loss

logger = logging.getLogger(__name__)


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
    # Anti-oversmoothing terms (default 0.0 -> pure L1, unchanged behavior).
    ssim_weight: float = field(default=0.0)
    mse_weight: float = field(default=0.0)
    gv_weight: float = field(default=0.0)
    # Alignment-tolerant mel loss (soft-DTW) + duration predictor loss (default 0 -> off).
    softdtw_weight: float = field(default=0.0)
    softdtw_gamma: float = field(default=0.1)
    dur_loss_weight: float = field(default=0.0)
    # Adversarial mel discriminator (Guo 2022). Default off -> pure reconstruction.
    use_discriminator: bool = field(default=False)
    disc_pretrain: bool = field(default=True)      # train disc alone before adversarial
    disc_start_updates: int = field(default=0)     # criterion-updates (= num_updates*world_size)
    adv_warmup_updates: int = field(default=0)     # ramp adv weight 0->1 over this many model-updates
    adv_weight: float = field(default=0.2)         # lambda_a (paper)
    fm_weight: float = field(default=2.0)          # lambda_f (paper)
    disc_lr: float = field(default=2e-4)
    disc_beta1: float = field(default=0.8)         # split (not "b1,b2") to avoid Hydra list ambiguity
    disc_beta2: float = field(default=0.99)
    disc_grad_clip: float = field(default=20.0)
    freeze_disc: bool = field(default=False)


@register_criterion("melvc_l1", dataclass=MelVCCriterionConfig)
class MelVCCriterion(FairseqCriterion):
    def __init__(self, task, mel_loss_weight, source_sr, target_sr, mel_bands,
                 n_fft, hop_size, win_size, fmin, fmax,
                 ssim_weight=0.0, mse_weight=0.0, gv_weight=0.0,
                 softdtw_weight=0.0, softdtw_gamma=0.1, dur_loss_weight=0.0,
                 use_discriminator=False, disc_pretrain=True, disc_start_updates=0,
                 adv_warmup_updates=0, adv_weight=0.2, fm_weight=2.0,
                 disc_lr=2e-4, disc_beta1=0.8, disc_beta2=0.99, disc_grad_clip=20.0,
                 freeze_disc=False):
        super().__init__(task)
        self.w = mel_loss_weight
        self.ssim_w = ssim_weight
        self.mse_w = mse_weight
        self.gv_w = gv_weight
        self.softdtw_w = softdtw_weight
        self.softdtw_gamma = softdtw_gamma
        self.dur_w = dur_loss_weight
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.hop = hop_size
        self.mel_cfg = dict(sampling_rate=target_sr, num_mels=mel_bands, n_fft=n_fft,
                            hop_size=hop_size, win_size=win_size, fmin=fmin, fmax=fmax)
        self._dct = None  # lazy (B, bands, bands) DCT-II basis for MCD

        # ---- adversarial state (mirrors criterionSpeechE2E_SynthVC) ----
        self.use_discriminator = use_discriminator
        self.disc_pretrain = disc_pretrain
        self.disc_start_updates = disc_start_updates
        self.adv_warmup_updates = adv_warmup_updates
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight
        self.disc_lr = disc_lr
        self.disc_betas = (disc_beta1, disc_beta2)
        self.disc_grad_clip = disc_grad_clip
        self.freeze_disc = freeze_disc
        self._disc = None                 # ref to model.mel_disc (lazy)
        self.disc_optimizer = None
        self._num_updates = 0
        self._disc_active_since = None
        self._adv_warmup_complete = False
        self._pending_disc_opt_state = None
        # A param so fairseq saves/loads this criterion's state_dict (disc optimizer).
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _lazy_init_disc(self, model):
        if self._disc is not None or getattr(model, "mel_disc", None) is None:
            return
        self._disc = model.mel_disc
        self._disc.float()   # model runs bf16, but mels are float32 (cf. E2E _spec_disc.float())
        trainable = not self.freeze_disc
        for p in self._disc.parameters():
            p.requires_grad = trainable
        if trainable:
            self.disc_optimizer = torch.optim.AdamW(
                self._disc.parameters(), lr=self.disc_lr, betas=self.disc_betas)
            if self._pending_disc_opt_state is not None:
                try:
                    self.disc_optimizer.load_state_dict(self._pending_disc_opt_state)
                    logger.info("[MelVC adv] restored disc_optimizer state from checkpoint")
                except Exception as e:
                    logger.warning(f"[MelVC adv] disc_optimizer state load failed: {e}")
                self._pending_disc_opt_state = None
        logger.info(f"[MelVC adv] disc ready: lr={self.disc_lr} betas={self.disc_betas} "
                    f"start_updates={self.disc_start_updates} pretrain={self.disc_pretrain} "
                    f"adv_warmup={self.adv_warmup_updates} freeze={self.freeze_disc}")

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
        return ssim.mean()   # tensor (grad kept); call .item() at metric sites

    @staticmethod
    def _gv(x, y, mask):
        """Global-variance loss: L1 between per-band (over time) variances. x,y:(B,bands,T)."""
        n = mask.sum(dim=2).clamp(min=1)                      # (B,1)
        mx = (x * mask).sum(dim=2) / n                        # (B,bands)
        my = (y * mask).sum(dim=2) / n
        vx = ((x - mx.unsqueeze(2)) ** 2 * mask).sum(dim=2) / n
        vy = ((y - my.unsqueeze(2)) ** 2 * mask).sum(dim=2) / n
        return (vx - vy).abs().mean()

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
        # Each term computed once (cheap) so it can be logged individually.
        l1 = (torch.abs(pred_mel - gt_mel) * mask).sum() / denom
        mse = ((pred_mel - gt_mel) ** 2 * mask).sum() / denom
        ssim_t = self._global_ssim(pred_mel, gt_mel, mask)   # tensor (grad kept)
        gv = self._gv(pred_mel, gt_mel, mask)
        ssim_pen = 1.0 - ssim_t
        loss = (self.w * l1 + self.mse_w * mse
                + self.ssim_w * ssim_pen + self.gv_w * gv)

        # Soft-DTW: alignment-tolerant mel loss (added, not replacing L1). Per-sample
        # over valid frames, length-normalised so short/long clips are comparable.
        # ponytail: python loop over batch (B small); swap to a batched CUDA kernel if slow.
        sdtw_val = 0.0
        if self.softdtw_w > 0:
            from .soft_dtw_cuda import soft_dtw_loss
            lengths = mel_lengths.clamp(max=m)               # (B,) valid frames
            P = pred_mel.transpose(1, 2)                     # (B, m, bands)
            G = gt_mel.transpose(1, 2)
            sdtw = soft_dtw_loss(P, G, lengths, self.softdtw_gamma)
            loss = loss + self.softdtw_w * sdtw
            sdtw_val = sdtw.detach().item()

        # Duration predictor loss (already computed inside the model, detached inputs).
        dur_loss = net_output.get("dur_loss")
        dur_val = 0.0
        if self.dur_w > 0 and dur_loss is not None:
            loss = loss + self.dur_w * dur_loss
            dur_val = float(dur_loss.detach().item())

        with torch.no_grad():
            mcd = self._mcd(pred_mel, gt_mel, mask, device)

        # ---------- adversarial mel discriminator (optional) ----------
        # Mirrors criterionSpeechE2E_SynthVC: a separate disc_optimizer (manual
        # backward/step), disc_pretrain (disc alone before adversarial), and the
        # disc_start_updates + adv_warmup_updates schedule. Disc lives in the model
        # (model.mel_disc) so it is checkpointed; the task freezes it out of the main
        # optimizer ("mel_disc." in always_frozen_prefixes).
        adv_logs = {}
        if getattr(model, "mel_disc", None) is not None and model.training:
            self._num_updates += 1
            self._lazy_init_disc(model)
            model_updates = getattr(model, "num_updates", 0)
            # disc_active: force-on via use_discriminator (E2E semantics = adversarial
            # from step 0), ELSE when criterion-updates (= num_updates * world_size=4)
            # reach disc_start_updates. Keep use_discriminator=False to use the schedule.
            disc_active = self.use_discriminator or (
                max(self._num_updates, model_updates * 4) >= self.disc_start_updates)
            real = gt_mel.unsqueeze(1)          # (B,1,bands,m) mel image
            fake = pred_mel.unsqueeze(1)

            loss_disc_val = 0.0
            if not self.freeze_disc and (disc_active or self.disc_pretrain):
                self.disc_optimizer.zero_grad(set_to_none=True)
                Cr, Fr, _ = self._disc(real.detach())
                Cf, Ff, _ = self._disc(fake.detach())
                loss_d = disc_loss([Cr, Fr], [Cf, Ff])
                loss_d.backward()
                if self.disc_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._disc.parameters(), self.disc_grad_clip)
                self.disc_optimizer.step()
                loss_disc_val = loss_d.item()

            loss_fm_val = loss_adv_val = 0.0
            adv_w = 0.0
            if disc_active:
                if self._disc_active_since is None:
                    self._disc_active_since = model_updates
                    logger.info(f"[MelVC adv] adversarial phase started at model_updates={model_updates}")
                with torch.no_grad():                       # real feats = fixed FM target
                    _, _, real_feats = self._disc(real)
                Cf, Ff, fake_feats = self._disc(fake)       # grad -> generator (+disc, ignored)
                loss_adv = gen_adv_loss([Cf, Ff])
                loss_fm = feature_matching_loss(real_feats, fake_feats)
                adv_w = 1.0
                if self.adv_warmup_updates > 0 and not self._adv_warmup_complete:
                    adv_w = min(1.0, max(0.0,
                        (model_updates - self._disc_active_since) / self.adv_warmup_updates))
                    if adv_w >= 1.0:
                        self._adv_warmup_complete = True
                        logger.info("[MelVC adv] adversarial warmup complete")
                loss = loss + adv_w * (self.adv_weight * loss_adv + self.fm_weight * loss_fm)
                loss_fm_val = loss_fm.item()
                loss_adv_val = loss_adv.item()
            adv_logs = {
                "loss_disc": loss_disc_val, "loss_fm": loss_fm_val,
                "loss_gen_adv": loss_adv_val, "adv_weight": adv_w,
                "disc_active": float(disc_active),
            }

        B = gt_wav.size(0)
        rr = net_output.get("residual_ratio")
        logging_output = {
            "loss": loss.detach().item(),
            "loss_mel": l1.detach().item(),
            "loss_mse": mse.detach().item(),       # raw MSE term (pre-weight)
            "loss_ssim": ssim_pen.detach().item(),  # raw (1-SSIM) penalty (pre-weight)
            "loss_gv": gv.detach().item(),          # raw GV term (pre-weight)
            "ssim_healthy": ssim_t.detach().item(),
            "mcd_healthy": mcd,
            "loss_softdtw": sdtw_val,
            "loss_dur": dur_val,
            "nsentences": B,
            "sample_size": B,
            **adv_logs,
        }
        if rr is not None:
            logging_output["residual_ratio"] = float(rr)
        return loss, B, logging_output

    # ---------- metrics ----------
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        n = max(len(logging_outputs), 1)
        for key, prio in [("loss", 100), ("loss_mel", 95),
                          ("loss_mse", 94), ("loss_ssim", 93), ("loss_gv", 92),
                          ("ssim_healthy", 90),
                          ("mcd_healthy", 90), ("residual_ratio", 80),
                          ("loss_softdtw", 91), ("loss_dur", 89),
                          ("loss_disc", 88), ("loss_fm", 87), ("loss_gen_adv", 86),
                          ("adv_weight", 85), ("disc_active", 84)]:
            vals = [lo[key] for lo in logging_outputs if key in lo]
            if vals:
                metrics.log_scalar(key, sum(vals) / len(vals), priority=prio, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False

    # ---------- adversarial state persistence (disc weights live in the model) ----------
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.disc_optimizer is not None:
            state["disc_optimizer"] = self.disc_optimizer.state_dict()
        state["disc_active_since"] = self._disc_active_since
        state["adv_warmup_complete"] = self._adv_warmup_complete
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        if state_dict is None:
            return
        # Stash disc_optimizer state — the optimizer is built lazily on first forward.
        self._pending_disc_opt_state = state_dict.pop("disc_optimizer", None)
        self._disc_active_since = state_dict.pop("disc_active_since", None)
        self._adv_warmup_complete = state_dict.pop("adv_warmup_complete", False)
        # Drop the disc submodule weights when loading FROM an adversarial checkpoint.
        # self._disc is a lazily-built reference to model.mel_disc (saved in / loaded
        # from the MODEL state), so these keys are redundant. At load time _disc does
        # not exist yet (lazy-init runs on the first forward, AFTER checkpoint load),
        # so a strict load would reject them -> crash on any finetune-from-adv-ckpt.
        # The disc still arrives via the model; disc_optimizer via _pending above.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_disc.")}
        return super().load_state_dict(state_dict, *args, **kwargs)


if __name__ == "__main__":
    # ponytail: smallest check that fails if the new loss terms break.
    torch.manual_seed(0)
    x = torch.randn(2, 80, 50, requires_grad=True)
    y = torch.randn(2, 80, 50)
    mask = torch.ones(2, 1, 50)
    assert MelVCCriterion._gv(x, x.detach(), mask).item() < 1e-6, "GV must be ~0 for identical inputs"
    assert MelVCCriterion._gv(x, y, mask).item() > 0, "GV must be >0 for different inputs"
    assert abs(MelVCCriterion._global_ssim(x, x.detach(), mask).item() - 1.0) < 1e-3, "SSIM~1 identical"
    (1.0 - MelVCCriterion._global_ssim(x, y, mask) + MelVCCriterion._gv(x, y, mask)).backward()
    assert x.grad is not None, "loss terms must be differentiable"
    print("criterionMelVC self-check OK")
