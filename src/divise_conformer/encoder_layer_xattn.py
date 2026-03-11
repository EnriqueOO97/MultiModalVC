#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Encoder layer with cross-attention to speaker embeddings.

Extends EncoderLayer with a cross-attention sub-layer inserted after
self-attention:

    FFN (macaron) → Self-Attn → Cross-Attn(q=x, k=spk, v=spk) → Conv → FFN

If spk_emb is None, the cross-attention is skipped (backward-compatible).
"""

import copy
import torch

from torch import nn

from .layer_norm import LayerNorm
from .encoder_layer import EncoderLayer


class EncoderLayerWithCrossAttn(EncoderLayer):
    """Conformer encoder layer with an added cross-attention sub-layer.

    The cross-attention uses plain MultiHeadedAttention (no positional
    encoding) because the speaker embedding has no temporal structure.

    Args:
        size (int): Input/output dimension (512 for conformer-L).
        self_attn: Self-attention module (may be relative-position MHA).
        cross_attn: Cross-attention module (plain MHA, query=x, key/value=spk_emb).
        feed_forward: Feed-forward module.
        conv_module: Convolution module (or None).
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Pre-norm vs post-norm.
        concat_after (bool): Concat-style residual for self-attention.
        macaron_style (bool): Macaron-style FFN.
    """

    def __init__(
        self,
        size,
        self_attn,
        cross_attn,
        feed_forward,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        macaron_style=False,
    ):
        # Parent sets up: self_attn, feed_forward, conv_module, norms, macaron
        super().__init__(
            size, self_attn, feed_forward, conv_module,
            dropout_rate, normalize_before, concat_after, macaron_style,
        )
        # Cross-attention sub-layer (new)
        self.cross_attn = cross_attn
        self.norm_cross = LayerNorm(size)

    def forward(self, x_input, mask, cache=None, spk_emb=None):
        """Forward with optional cross-attention to speaker embedding.

        Args:
            x_input: (B, T, size) or tuple((B, T, size), pos_emb)
            mask: (B, T) or None
            cache: optional cache tensor
            spk_emb: (B, 1, size) speaker embedding, or None to skip

        Returns:
            Tuple of (output, mask) — same interface as parent
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # === Macaron-style first FFN ===
        if self.macaron_style:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # === Self-attention (identical to parent) ===
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # === Cross-attention to speaker embedding (NEW) ===
        if spk_emb is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_cross(x)
            # query = x (B, T, size), key = value = spk_emb (B, 1, size)
            # No mask needed — spk_emb is always valid (single vector)
            x_cross = self.cross_attn(x, spk_emb, spk_emb, mask=None)
            x = residual + self.dropout(x_cross)
            if not self.normalize_before:
                x = self.norm_cross(x)

        # === Convolution module ===
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # === Feed-forward ===
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask
        else:
            return x, mask
