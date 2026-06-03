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


class MultiResolutionSTFTLoss(nn.Module):
    """L1 on linear-magnitude STFT at multiple resolutions (spectral convergence
    + log-magnitude L1, the standard Parallel-WaveGAN formulation).

    Computes everything in float32 (cuFFT has no bf16 support). GT side is detached.
    Windows are kept moderately large (no tiny hops) for robustness to the
    imperfectly-aligned subset of the training data.
    """
    def __init__(self, resolutions=None):
        super().__init__()
        if resolutions is None:
            # (n_fft, hop, win)
            resolutions = [
                (512, 128, 512),
                (1024, 256, 1024),
                (2048, 512, 2048),
            ]
        self.resolutions = resolutions
        for i, (n_fft, hop, win) in enumerate(resolutions):
            self.register_buffer(f"window_{i}", torch.hann_window(win), persistent=False)

    def _stft_mag(self, wav, n_fft, hop, win, window):
        pad = (n_fft - hop) // 2
        wav = F.pad(wav, (pad, pad), mode="reflect")
        spec = torch.stft(
            wav, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, center=False, return_complex=True,
        )
        mag = torch.abs(spec)
        return torch.clamp(mag, min=1e-7)

    def forward(self, pred_wav, gt_wav):
        if pred_wav.dim() == 3:
            pred_wav = pred_wav.squeeze(1)
        if gt_wav.dim() == 3:
            gt_wav = gt_wav.squeeze(1)
        pred_wav = pred_wav.float()
        gt_wav = gt_wav.float()

        total = 0.0
        for i, (n_fft, hop, win) in enumerate(self.resolutions):
            window = getattr(self, f"window_{i}")
            mp = self._stft_mag(pred_wav, n_fft, hop, win, window)
            with torch.no_grad():
                mg = self._stft_mag(gt_wav, n_fft, hop, win, window)
            min_t = min(mp.size(-1), mg.size(-1))
            mp = mp[..., :min_t]
            mg = mg[..., :min_t]
            sc = torch.norm(mg - mp, p="fro") / (torch.norm(mg, p="fro") + 1.0)
            logmag = F.l1_loss(torch.log1p(mp), torch.log1p(mg))
            total = total + sc + logmag
        return total / len(self.resolutions)


class DNSMOSProLoss(nn.Module):
    """Perceptual loss using a frozen DNSMOSPro TorchScript model.

    The generator's predicted waveform is scored by the frozen MOS predictor.
    Training loss = mean(5 - score), minimised when score → 5 (perfect quality).
    Validation metric = mean(score) directly (logged as dnsmos_score, range 1–5).

    STFT is reimplemented in pure PyTorch (no librosa/numpy) so gradients flow
    back through the spectrogram into the generator during training.
    Model weights are permanently frozen and receive no gradients.

    Input: 16 kHz mono waveform. Internally float32 (cuFFT has no bf16 support).
    Parameters match the BVCC gin config: n_fft=320, hop=160, win=320, log-mag.
    """

    N_FFT      = 320
    HOP_LENGTH = 160
    WIN_LENGTH = 320

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.dnsmos = None  # lazy: loaded on first forward (needs device)
        self.register_buffer("_window", torch.hann_window(self.WIN_LENGTH), persistent=False)

    def _load_model(self, device):
        if self.dnsmos is None:
            self.dnsmos = torch.jit.load(self.checkpoint_path, map_location=device)
            self.dnsmos.eval()
            for p in self.dnsmos.parameters():
                p.requires_grad_(False)
            logger_inner = logging.getLogger("src.criterionSpeechE2E_SynthVC")
            logger_inner.info(f"[DNSMOSPro] loaded frozen model from {self.checkpoint_path}")

    def _wav_to_spec(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) float32 → spec: (B, 1, T_frames, 161) float32 log10-magnitude."""
        window = self._window.to(wav.device)
        spec = torch.stft(
            wav,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            win_length=self.WIN_LENGTH,
            window=window,
            center=True,
            return_complex=True,
        )                                              # (B, 161, T_frames)
        mag = torch.abs(spec)                          # (B, 161, T_frames)
        mag = torch.clamp(mag, min=1e-7, max=1e7)
        logmag = torch.log10(mag)                      # (B, 161, T_frames)
        logmag = logmag.transpose(1, 2)                # (B, T_frames, 161)
        return logmag.unsqueeze(1)                     # (B, 1, T_frames, 161)

    def score(self, pred_wav: torch.Tensor) -> torch.Tensor:
        """Return predicted MOS mean for each sample. Shape: (B,), range ~[1, 5]."""
        if pred_wav.dim() == 3:
            pred_wav = pred_wav.squeeze(1)
        pred_wav = pred_wav.float()
        self._load_model(pred_wav.device)
        spec = self._wav_to_spec(pred_wav)
        out = self.dnsmos(spec)                        # (B, 2)
        return out[:, 0]                               # MOS mean

    def forward(self, pred_wav: torch.Tensor) -> torch.Tensor:
        """Scalar training loss = mean(5 - MOS). Lower → better generator output."""
        return (5.0 - self.score(pred_wav)).mean()


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
    freeze_disc: bool = field(
        default=False, metadata={"help": "Freeze discriminator weights — skip disc optimizer/scheduler, keep loss_fm "
                                        "and loss_gen_adv flowing to generator. For fine-tuning on small datasets "
                                        "where the disc is already well-trained."}
    )
    reset_disc_schedule: bool = field(
        default=False, metadata={"help": "On a fresh finetune_from_model, do NOT inherit the source checkpoint's "
                                        "adversarial schedule state (_num_updates, _disc_active_since, "
                                        "_adv_warmup_complete, disc optimizer/scheduler). Forces the disc "
                                        "pretrain->adversarial schedule to start from zero so disc_start_updates and "
                                        "adv_warmup_updates are honored relative to this run."}
    )
    use_mrstft_loss: bool = field(
        default=False, metadata={"help": "Enable multi-resolution STFT reconstruction loss (generator-side)"}
    )
    mrstft_loss_weight: float = field(
        default=2.0, metadata={"help": "Weight for MR-STFT loss (mel weight is ~30 for scale reference)"}
    )
    use_dnsmos_loss: bool = field(
        default=False, metadata={"help": "Enable DNSMOSPro perceptual loss (frozen MOS predictor as generator loss)"}
    )
    dnsmos_loss_weight: float = field(
        default=1.0, metadata={"help": "Weight for DNSMOSPro loss. Tune: if gnorm spikes on enable, halve it."}
    )
    dnsmos_checkpoint: str = field(
        default="", metadata={"help": "Path to DNSMOSPro TorchScript checkpoint (.pt). "
                                      "Also used as validation metric when model is initialised."}
    )



@register_criterion("e2e_gan_loss_synthvc", dataclass=E2EGanLossSynthVCConfig)
class E2EGanLossSynthVC(E2EGanLoss):
    def __init__(self, task, mel_loss_weight=40.0, use_discriminator=True,
                 disc_lr=2e-4, disc_betas="0.8,0.99", conv_loss_weight=5.0,
                 disc_start_updates=30000, mel_num_mels=128, mel_hop_size=160,
                 disc_grad_clip=0.0, adv_warmup_updates=0, use_multires_mel=False,
                 disc_lr_t_max=550000, disc_pretrain=True, freeze_disc=False,
                 reset_disc_schedule=False, use_mrstft_loss=False, mrstft_loss_weight=2.0,
                 use_dnsmos_loss=False, dnsmos_loss_weight=1.0, dnsmos_checkpoint=""):
        super().__init__(task, mel_loss_weight, use_discriminator, disc_lr, disc_betas,
                         mel_num_mels=mel_num_mels, mel_hop_size=mel_hop_size)
        self.conv_loss_weight = conv_loss_weight
        self.disc_start_updates = disc_start_updates
        self.disc_grad_clip = disc_grad_clip
        self.adv_warmup_updates = adv_warmup_updates
        self.use_multires_mel = use_multires_mel
        self.disc_lr_t_max = disc_lr_t_max
        self.disc_pretrain = disc_pretrain
        self.freeze_disc = freeze_disc
        self.reset_disc_schedule = reset_disc_schedule
        self._spec_disc = None  # lazy init in _lazy_init
        self.disc_lr_scheduler = None  # CosineAnnealingLR, created alongside disc_optimizer
        self._num_updates = 0
        # _disc_active_since is stored in *model* update units (fairseq optimizer steps),
        # which are preserved across resumes.  Previous code used criterion-call units
        # (which reset to 0 on each resume), causing negative adv_weight after resume.
        self._disc_active_since = None
        self._adv_warmup_complete = False   # once True, adv_weight stays 1.0 forever
        self.multires_mel = None  # lazy init
        self.use_mrstft_loss = use_mrstft_loss
        self.mrstft_loss_weight = mrstft_loss_weight
        self.mrstft = None     # lazy init
        self.use_dnsmos_loss = use_dnsmos_loss
        self.dnsmos_loss_weight = dnsmos_loss_weight
        self.dnsmos_checkpoint = dnsmos_checkpoint
        self.dnsmos_model = None  # lazy init
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
                     f"disc_pretrain={self.disc_pretrain}, freeze_disc={self.freeze_disc}, "
                     f"use_mrstft_loss={self.use_mrstft_loss}, mrstft_loss_weight={self.mrstft_loss_weight}, "
                     f"use_dnsmos_loss={self.use_dnsmos_loss}, dnsmos_loss_weight={self.dnsmos_loss_weight}, "
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

        if self.use_mrstft_loss and self.mrstft is None:
            self.mrstft = MultiResolutionSTFTLoss().to(device)
            logger.info("[SynthVC Criterion] Initialized MR-STFT loss")

        # DNSMOSPro: initialised whenever a checkpoint path is provided.
        # The model is always frozen. When use_dnsmos_loss=True it also contributes
        # to the training loss; otherwise it is a validation-only metric.
        if self.dnsmos_checkpoint and self.dnsmos_model is None:
            self.dnsmos_model = DNSMOSProLoss(self.dnsmos_checkpoint).to(device)

        # Move EMA state to GPU after checkpoint load (checkpoints store EMA on CPU)
        if self._ema_initialized and self._ema_state:
            first_val = next(iter(self._ema_state.values()))
            if first_val.device != device:
                self._ema_state = {k: v.to(device) for k, v in self._ema_state.items()}
                logger.info(f"[SynthVC EMA] Moved EMA state to {device}")

        if self._spec_disc is None:
            # Spectral disc (MS-STFT or CQT) uses cuFFT internally — needs float32.
            # MPD works fine in bf16.
            self._use_cqt = hasattr(model, 'cqtd')
            self._spec_disc = model.cqtd if self._use_cqt else model.msstftd
            self._spec_disc.float()

            if self.freeze_disc:
                # Disc is frozen: no optimizer, no scheduler, params not trainable.
                # The disc still runs in forward() so loss_fm and loss_gen_adv flow
                # to the generator, but its weights stay fixed.
                for param in model.mpd.parameters():
                    param.requires_grad = False
                for param in self._spec_disc.parameters():
                    param.requires_grad = False
                # Discard any pending optimizer/scheduler state from checkpoint —
                # we're not building those optimizers.
                self._pending_disc_optimizer_state = None
                self._pending_disc_scheduler_state = None
                n_disc = sum(p.numel() for p in model.mpd.parameters()) + sum(p.numel() for p in self._spec_disc.parameters())
                logger.info(f"[SynthVC Criterion] Disc FROZEN: weights not trainable, "
                            f"no disc optimizer/scheduler created ({n_disc:,} params held fixed)")
            else:
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

        # Freeze upstream modules the first time Phase 2 activates.
        # The pathological finetune task owns its own freeze plan and signals
        # this via model._finetune_owns_freeze — skip the legacy auto-freeze there.
        if disc_active and not getattr(model, "_finetune_owns_freeze", False):
            model._freeze_disc = self.freeze_disc
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
        # Auxiliary generator-side losses (MR-STFT + DNSMOSPro perceptual).
        # Computed here so they feed into all three branches (disc-inactive,
        # disc-active, validation). Default OFF — zero when disabled.
        # =====================================================================
        loss_mrstft = torch.tensor(0.0, device=pred_wav.device)
        if self.use_mrstft_loss and self.mrstft is not None:
            loss_mrstft = self.mrstft(pred_wav, gt_wav)

        # DNSMOSPro perceptual loss: gradient only flows once the disc is active
        # (generator is "ready"), mirroring the adversarial schedule. Before that,
        # and during validation, the score is computed under no_grad as a metric only.
        # The DNSMOS model weights are always frozen.
        loss_dnsmos = torch.tensor(0.0, device=pred_wav.device)
        dnsmos_score_val = torch.tensor(0.0, device=pred_wav.device)
        if self.dnsmos_model is not None:
            if model.training and self.use_dnsmos_loss and disc_active:
                mos = self.dnsmos_model.score(pred_wav)
                loss_dnsmos = (5.0 - mos).mean()
                dnsmos_score_val = mos.detach().mean()
            else:
                with torch.no_grad():
                    mos = self.dnsmos_model.score(pred_wav)
                    dnsmos_score_val = mos.mean()

        # aux_recon = reference-anchored reconstruction losses, applied in all branches.
        # DNSMOS is added separately in the disc-active branch only, ramped by adv_weight.
        aux_recon = self.mrstft_loss_weight * loss_mrstft

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

                # Discriminator step (skipped when disc is frozen)
                if not self.freeze_disc:
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

                # When disc is frozen we never ran the disc-update block above, so
                # `loss_disc` was never set.  Compute it here from the gen-step scores
                # for logging only — no backward, no optimizer step.
                if self.freeze_disc:
                    with torch.no_grad():
                        loss_disc_mpd, _, _ = discriminator_loss(mpd_real_scores, mpd_fake_scores)
                        loss_disc_msstftd, _, _ = discriminator_loss(msstftd_real_scores, msstftd_fake_scores)
                        loss_disc = loss_disc_mpd + loss_disc_msstftd

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
                            + aux_recon
                            + adv_weight * (loss_fm + loss_gen_adv)
                            + adv_weight * self.dnsmos_loss_weight * loss_dnsmos)

                logging_output = {
                    "loss": loss_gen.item(),
                    "loss_mel": loss_mel.item(),
                    "loss_conv": 0.0,
                    "loss_mel_weighted": (self.mel_loss_weight * loss_mel).item(),
                    "loss_conv_weighted": 0.0,
                    "loss_fm": loss_fm.item(),
                    "loss_gen_adv": loss_gen_adv.item(),
                    "loss_disc": loss_disc.item(),
                    "loss_mrstft": loss_mrstft.item(),
                    "loss_dnsmos": loss_dnsmos.item(),
                    "dnsmos_score": dnsmos_score_val.item(),
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

                loss_gen = self.mel_loss_weight * loss_mel + self.conv_loss_weight * loss_conv + aux_recon

                # =============================================================
                # PHASE 1 DISC PRE-TRAINING: Train discriminator as a classifier
                # pred_wav is detached — no adversarial gradient reaches the generator.
                # The disc optimizer is stepped but the LR scheduler is NOT —
                # cosine decay budget is reserved for Phase 2 adversarial training.
                # =============================================================
                loss_disc_pretrain = 0.0
                if self.disc_pretrain and not self.freeze_disc:
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
                    "loss_mrstft": loss_mrstft.item(),
                    "loss_dnsmos": loss_dnsmos.item(),
                    "dnsmos_score": dnsmos_score_val.item(),
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

            loss_gen = self.mel_loss_weight * loss_mel + self.conv_loss_weight * loss_conv + aux_recon

            logging_output = {
                "loss": loss_gen.item(),
                "loss_mel": loss_mel.item(),
                "loss_conv": loss_conv.item(),
                "loss_mel_weighted": (self.mel_loss_weight * loss_mel).item(),
                "loss_conv_weighted": (self.conv_loss_weight * loss_conv).item(),
                "loss_fm": loss_fm.item(),
                "loss_gen_adv": loss_gen_adv.item(),
                "loss_disc": loss_disc.item(),
                "loss_mrstft": loss_mrstft.item(),
                "loss_dnsmos": loss_dnsmos.item(),
                "dnsmos_score": dnsmos_score_val.item(),
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

                # Split MCD/SSIM by target type: healthy (fid ends with "__t0")
                # vs synth (any other target column). Uses the primary mel rep so
                # the two groups are computed identically regardless of multires.
                utt_ids = sample.get("utt_id", []) or []
                mp_primary = mel_pred.transpose(1, 2).float()
                mg_primary = mel_gt.transpose(1, 2).float()
                if len(utt_ids) == mp_primary.size(0):
                    h_idx = [i for i, uid in enumerate(utt_ids) if str(uid).endswith("__t0")]
                    s_idx = [i for i, uid in enumerate(utt_ids) if not str(uid).endswith("__t0")]
                    for grp, idxs in (("healthy", h_idx), ("synth", s_idx)):
                        if not idxs:
                            continue  # group absent in this batch — omit (averaged only over present batches)
                        sel = torch.tensor(idxs, device=mp_primary.device, dtype=torch.long)
                        mp_g = mp_primary.index_select(0, sel)
                        mg_g = mg_primary.index_select(0, sel)
                        try:
                            logging_output[f"mcd_{grp}"] = compute_mcd(mp_g, mg_g).item()
                        except Exception:
                            pass
                        try:
                            logging_output[f"ssim_{grp}"] = compute_ssim(mp_g, mg_g).item()
                        except Exception:
                            pass

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

        # Fresh finetune: discard the source checkpoint's adversarial schedule so
        # disc_start_updates / adv_warmup_updates are honored relative to THIS run.
        # We still let the disc *weights* load (those live in the model state_dict),
        # but drop the schedule counters and the disc optimizer/scheduler momentum.
        if self.reset_disc_schedule:
            for k in ("disc_active_since_v2", "disc_active_since", "adv_warmup_complete",
                      "disc_optimizer_state", "disc_lr_scheduler_state", "disc_sched_last_update",
                      "criterion_num_updates"):
                state_dict.pop(k, None)
            self._num_updates = 0
            self._disc_active_since = None
            self._adv_warmup_complete = False
            self._pending_disc_optimizer_state = None
            self._pending_disc_scheduler_state = None
            self._disc_sched_last_update = -1
            # EMA is still restored below (it's just a shadow of generator weights).
            ema_state = state_dict.pop("ema_state", None)
            super().load_state_dict(state_dict, strict=False)
            if ema_state is not None:
                cleaned_ema = {_ema_strip_module(k): v for k, v in ema_state.items()}
                self._ema_state = cleaned_ema
                self._ema_initialized = True
                logger.info(f"[SynthVC Criterion] Restored EMA state ({len(cleaned_ema)} tensors)")
            logger.info("[SynthVC Criterion] reset_disc_schedule=True — adversarial schedule starts fresh "
                        "(_num_updates=0, _disc_active_since=None, warmup not complete, disc optimizer reset)")
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

        for key in ["loss_mel", "loss_conv", "loss_mel_weighted", "loss_conv_weighted", "loss_fm", "loss_gen_adv", "loss_disc", "loss_mrstft", "loss_dnsmos", "dnsmos_score"]:
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

        # Per-target-type metrics: healthy vs synth (averaged over batches that
        # contained that group). Lets you see if synth targets drag the score.
        # mcd_healthy is also used as the best-checkpoint metric, and fairseq
        # reads stats[best_checkpoint_metric] with a direct dict access at
        # validation time — so it MUST exist whenever we validate.
        is_validation = any("val_mel_loss" in log for log in logging_outputs)
        for key, priority in [
            ("mcd_healthy", 78), ("ssim_healthy", 68),
            ("mcd_synth", 77), ("ssim_synth", 67),
        ]:
            values = [log[key] for log in logging_outputs if key in log]
            if values:
                metrics.log_scalar(key, sum(values) / len(values), priority=priority, round=4)
            elif is_validation and key == "mcd_healthy":
                # Safety net: guarantee the best-checkpoint metric is present so
                # fairseq never KeyErrors. Large value => this validation is never
                # selected as "best".
                metrics.log_scalar(key, 1e9, priority=priority, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
