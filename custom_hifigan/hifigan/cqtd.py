# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/discriminators.py
# Originally from https://github.com/open-mmlab/Amphion/blob/main/models/vocoders/gan/discriminator/mssbcqtd.py
# MIT License

import typing
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class DiscriminatorCQT(nn.Module):
    """Single-scale CQT sub-discriminator.

    Args:
        sample_rate: Audio sample rate (e.g. 16000).
        hop_length: CQT hop length in samples. Should be a multiple of the generator hop.
        n_octaves: Number of octaves to cover. Use 8 with fmin=31.0 for near-full 16kHz coverage
                   (31Hz * 2^8 = 7936Hz, safely below Nyquist=8000Hz). Requires hop_length divisible by 2^(n_octaves-1).
        fmin: Lowest frequency in Hz. Default 31.0 gives ceiling of 7936Hz — close to Nyquist without exceeding it.
        bins_per_octave: Frequency bins per octave (12 = 1 per semitone).
        filters: Base number of conv filters.
        max_filters: Maximum number of conv filters.
        filters_scale: Filter growth factor per layer.
        dilations: List of dilation values for conv layers.
        in_channels: Number of input channels (1 for mono audio).
        out_channels: Number of output channels.
        normalize_volume: Apply DC removal and peak normalization before CQT.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 480,
        n_octaves: int = 7,
        bins_per_octave: int = 24,
        fmin: float = 31.0,
        filters: int = 32,
        max_filters: int = 1024,
        filters_scale: int = 1,
        dilations: List[int] = [1, 2, 4],
        in_channels: int = 1,
        out_channels: int = 1,
        normalize_volume: bool = False,
    ):
        super().__init__()

        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.normalize_volume = normalize_volume

        kernel_size = (3, 9)
        stride = (1, 2)

        # Lazy-load nnAudio to avoid hard dependency at import time
        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=bins_per_octave * n_octaves,
            bins_per_octave=bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        # Per-octave pre-processing convolutions
        self.conv_pres = nn.ModuleList()
        for _ in range(n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    in_channels * 2,
                    in_channels * 2,
                    kernel_size=kernel_size,
                    padding=self._get_2d_padding(kernel_size),
                )
            )

        # Main conv stack
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Conv2d(
                in_channels * 2,
                filters,
                kernel_size=kernel_size,
                padding=self._get_2d_padding(kernel_size),
            )
        )

        in_chs = min(filters_scale * filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * filters, max_filters)
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=(dilation, 1),
                        padding=self._get_2d_padding(kernel_size, (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs

        out_chs = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(kernel_size[0], kernel_size[0]),
                    padding=self._get_2d_padding((kernel_size[0], kernel_size[0])),
                )
            )
        )

        self.conv_post = weight_norm(
            nn.Conv2d(
                out_chs,
                out_channels,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=self._get_2d_padding((kernel_size[0], kernel_size[0])),
            )
        )

        self.activation = nn.LeakyReLU(negative_slope=0.1)

    @staticmethod
    def _get_2d_padding(
        kernel_size: typing.Tuple[int, int],
        dilation: typing.Tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        if self.normalize_volume:
            x = x - x.mean(dim=-1, keepdims=True)
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        # nnAudio CQT2010v2 downsamples by 2 per octave; pad to multiple of
        # 2^(n_octaves-1) to ensure consistent frame counts across all octaves
        pad_multiple = 2 ** (self.n_octaves - 1)
        remainder = x.shape[-1] % pad_multiple
        if remainder != 0:
            x = torch.nn.functional.pad(x, (0, pad_multiple - remainder))

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for l in self.convs:
            latent_z = l(latent_z)
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    """Multi-scale CQT discriminator.

    Defaults are tuned for 16kHz speech with generator hop_length=160:
      - hop_lengths are all multiples of 128 (= 2^(8-1)), required for n_octaves=8
      - n_octaves=8 with fmin=31.0Hz covers 31Hz * 2^8 = 7936Hz, near-full coverage safely below Nyquist=8000Hz
      - bins_per_octave decreases for coarser scales (lower time resolution scales)

    Divisibility check (hop must be divisible by 2^(n_octaves-1) = 128):
      1024/128=8 ✓, 640/128=5 ✓, 512/128=4 ✓, 256/128=2 ✓, 128/128=1 ✓

    Args:
        sample_rate: Audio sample rate.
        hop_lengths: CQT hop per scale.
        n_octaves: Octaves per scale.
        bins_per_octaves: Frequency resolution per scale.
        filters: Base conv filter count.
        max_filters: Max conv filter count.
        filters_scale: Filter growth factor.
        dilations: Dilation list for conv layers.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_lengths: List[int] = [1024, 640, 512, 256, 128],
        n_octaves: List[int] = [8, 8, 8, 8, 8],
        bins_per_octaves: List[int] = [12, 18, 24, 36, 48],
        fmin: float = 31.0,
        filters: int = 32,
        max_filters: int = 1024,
        filters_scale: int = 1,
        dilations: List[int] = [1, 2, 4],
    ):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    sample_rate=sample_rate,
                    hop_length=hop_lengths[i],
                    n_octaves=n_octaves[i],
                    bins_per_octave=bins_per_octaves[i],
                    fmin=fmin,
                    filters=filters,
                    max_filters=max_filters,
                    filters_scale=filters_scale,
                    dilations=dilations,
                )
                for i in range(len(hop_lengths))
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        scores = []
        fmaps = []
        for disc in self.discriminators:
            score, fmap = disc(x)
            scores.append(score)
            fmaps.append(fmap)
        return scores, fmaps
