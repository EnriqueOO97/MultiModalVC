"""
Multi-scale time-frequency mel-spectrogram discriminator (Guo et al., Interspeech
2022, arXiv:2203.01080), used to sharpen the MelVC generator's mels adversarially.

A symmetric U-Net over the mel "image" (1, N, T): a conv encoder downsamples to a
256-channel bottleneck and emits a COARSE score map; a transpose-conv decoder with
U-Net skip-concats upsamples back and emits a FINE score map at input resolution.
WeightNorm on every conv; LeakyReLU(0.2) on all layers except the input conv.

Losses (paper Eqs. 4-7, LS-GAN):
  Ld = MSE(Cr,1)+MSE(Fr,1)+MSE(Cf,0)+MSE(Ff,0)
  La = MSE(Cf,1)+MSE(Ff,1)
  Lf = MAE over hidden feature pairs
  (generator total adds  lambda_a*La + lambda_f*Lf  to the reconstruction loss;
   paper uses lambda_a=0.2, lambda_f=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def _wn_conv(i, o, k, s):
    return weight_norm(nn.Conv2d(i, o, k, s, padding=k // 2))


def _wn_tconv(i, o, k=4, s=2):
    return weight_norm(nn.ConvTranspose2d(i, o, k, s, padding=1))


def _crop_to(x, ref):
    """Crop x (B,C,H,W) to ref's H,W (transpose-conv can overshoot by 1 on odd dims)."""
    return x[..., : ref.size(-2), : ref.size(-1)]


class MelTFDiscriminator(nn.Module):
    def __init__(self, lrelu: float = 0.2):
        super().__init__()
        self.lrelu = lrelu
        # Encoder: (in, out, k, stride) — strides 1/2/1/2/1/2 -> /8 downsample.
        self.e1 = _wn_conv(1, 8, 3, 1)
        self.e2 = _wn_conv(8, 16, 3, 2)
        self.e3 = _wn_conv(16, 32, 3, 1)
        self.e4 = _wn_conv(32, 64, 3, 2)
        self.e5 = _wn_conv(64, 128, 3, 1)
        self.e6 = _wn_conv(128, 256, 3, 2)            # bottleneck
        self.coarse = _wn_conv(256, 1, 3, 1)          # coarse-grained score map
        # Decoder: transpose-conv upsamples, skip-concat doubles channels, conv halves.
        self.t1 = _wn_tconv(256, 128)
        self.d1 = _wn_conv(128 + 128, 64, 3, 1)       # concat e5 (128ch @ N/4)
        self.t2 = _wn_tconv(64, 32)
        self.d2 = _wn_conv(32 + 32, 16, 3, 1)         # concat e3 (32ch @ N/2)
        self.t3 = _wn_tconv(16, 8)
        self.fine = _wn_conv(8 + 8, 1, 3, 1)          # concat e1 (8ch @ N); fine score

    def forward(self, x):
        """x: (B, 1, N, T) mel image -> (coarse, fine, feats). feats for feat-matching."""
        a = self.lrelu
        f1 = self.e1(x)                                # input layer: NO activation
        f2 = F.leaky_relu(self.e2(f1), a)
        f3 = F.leaky_relu(self.e3(f2), a)
        f4 = F.leaky_relu(self.e4(f3), a)
        f5 = F.leaky_relu(self.e5(f4), a)
        f6 = F.leaky_relu(self.e6(f5), a)              # bottleneck
        coarse = self.coarse(f6)

        u1 = _crop_to(F.leaky_relu(self.t1(f6), a), f5)
        g1 = F.leaky_relu(self.d1(torch.cat([u1, f5], 1)), a)
        u2 = _crop_to(F.leaky_relu(self.t2(g1), a), f3)
        g2 = F.leaky_relu(self.d2(torch.cat([u2, f3], 1)), a)
        u3 = _crop_to(F.leaky_relu(self.t3(g2), a), f1)
        fine = self.fine(torch.cat([u3, f1], 1))

        feats = [f1, f2, f3, f4, f5, f6, g1, g2]
        return coarse, fine, feats


# ---- LS-GAN + feature-matching losses (paper Eqs. 4-6) -----------------------
def disc_loss(real_maps, fake_maps):
    """Sum over [coarse, fine] of MSE(real,1)+MSE(fake,0)."""
    return sum(((r - 1.0) ** 2).mean() + (f ** 2).mean()
               for r, f in zip(real_maps, fake_maps))


def gen_adv_loss(fake_maps):
    """Sum over [coarse, fine] of MSE(fake,1)."""
    return sum(((f - 1.0) ** 2).mean() for f in fake_maps)


def feature_matching_loss(real_feats, fake_feats):
    """Mean MAE over hidden feature pairs."""
    return sum((rf - ff).abs().mean() for rf, ff in zip(real_feats, fake_feats)) \
        / max(len(real_feats), 1)


if __name__ == "__main__":
    # ponytail: smallest check that fails if the U-Net wiring / losses break.
    torch.manual_seed(0)
    d = MelTFDiscriminator()
    mel = torch.randn(2, 1, 80, 57, requires_grad=True)  # odd T -> exercises _crop_to
    C, Fm, feats = d(mel)
    assert C.dim() == 4 and Fm.shape[-2:] == mel.shape[-2:], (C.shape, Fm.shape)
    gt = torch.randn(2, 1, 80, 57)
    Cr, Fr, fr = d(gt)
    assert disc_loss([Cr, Fr], [Cr.detach(), Fr.detach()]).item() >= 0
    # identical real/fake -> adv pushes fake->1; FM(x,x)=0
    assert feature_matching_loss(fr, fr).item() < 1e-6
    loss = gen_adv_loss([C, Fm]) + feature_matching_loss(fr, feats)
    loss.backward()
    assert mel.grad is not None
    print("mel_discriminator self-check OK; coarse", tuple(C.shape), "fine", tuple(Fm.shape))
