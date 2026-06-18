"""
Exact BigVGAN-v2 mel-spectrogram computation.

Copied to match NVIDIA/BigVGAN `meldataset.py:mel_spectrogram()` so the mels we
SUPERVISE against are byte-for-byte the format the pretrained BigVGAN vocoder
expects. ANY deviation (filterbank, magnitude vs power, log compression,
padding/centering, normalization) makes BigVGAN produce garbage — so this file
is deliberately a faithful copy, not a re-derivation.

Default config = bigvgan_v2_22khz_80band_fmax8k_256x (best match for 16 kHz
source audio, whose content is band-limited to 8 kHz):
    sampling_rate=22050, num_mels=80, n_fft=1024, hop=256, win=1024,
    fmin=0, fmax=8000, center=False (reflect-pad (n_fft-hop)//2 each side).

Mel frame rate = 22050/256 ≈ 86.13 fps  ->  n_frames ≈ audio_samples // 256.
"""

import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn

# BigVGAN v2 22kHz/80-band/fmax8k mel config.
BIGVGAN_22K_80B_FMAX8K = dict(
    sampling_rate=22050,
    num_mels=80,
    n_fft=1024,
    hop_size=256,
    win_size=1024,
    fmin=0,
    fmax=8000,
)

_mel_basis_cache = {}
_hann_cache = {}
_resampler_cache = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    # BigVGAN's exact log compression (no normalization afterwards).
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectrogram(y, sampling_rate, num_mels, n_fft, hop_size, win_size,
                    fmin, fmax, center=False):
    """y: (B, T) waveform in [-1, 1] at `sampling_rate`. Returns (B, num_mels, T_frames).

    Matches BigVGAN meldataset.py exactly: librosa (slaney) mel basis, magnitude
    (not power) spectrogram, reflect-pad (n_fft-hop)//2 with center=False, then
    log(clamp(., 1e-5)). No mean/var normalization.
    """
    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"
    if key not in _mel_basis_cache:
        mb = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                            fmin=fmin, fmax=fmax)
        _mel_basis_cache[key] = torch.from_numpy(mb).float().to(device)
        _hann_cache[key] = torch.hann_window(win_size).to(device)
    mel_basis = _mel_basis_cache[key]
    hann_window = _hann_cache[key]

    pad = int((n_fft - hop_size) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
        center=center, pad_mode="reflect", normalized=False, onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)  # magnitude
    spec = torch.matmul(mel_basis, spec)                            # (B, num_mels, T)
    spec = dynamic_range_compression_torch(spec)
    return spec


def mel_from_config(y, cfg=BIGVGAN_22K_80B_FMAX8K):
    return mel_spectrogram(
        y, sampling_rate=cfg["sampling_rate"], num_mels=cfg["num_mels"],
        n_fft=cfg["n_fft"], hop_size=cfg["hop_size"], win_size=cfg["win_size"],
        fmin=cfg["fmin"], fmax=cfg["fmax"],
    )


def resample(wav, orig_sr, target_sr):
    """Cached on-the-fly resample. wav: (B, T) or (T,). No-op if sr matches."""
    if orig_sr == target_sr:
        return wav
    key = (orig_sr, target_sr, str(wav.device))
    if key not in _resampler_cache:
        _resampler_cache[key] = torchaudio.transforms.Resample(
            orig_sr, target_sr).to(wav.device)
    return _resampler_cache[key](wav)


def mel_frames_for_samples(n_samples, hop_size=256):
    """Number of BigVGAN mel frames for `n_samples` at the target SR.
    Matches center=False + (n_fft-hop)//2 padding -> floor((n - hop)/hop) + 1."""
    return max(1, (int(n_samples) - hop_size) // hop_size + 1)
