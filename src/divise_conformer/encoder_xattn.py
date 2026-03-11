#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conformer encoder with per-layer cross-attention to speaker embeddings.

Mirrors the structure of encoder.py but uses EncoderLayerWithCrossAttn and
passes spk_emb through the layer stack.
"""

import torch

from .attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    LegacyRelPositionMultiHeadedAttention,
)
from .convolution import ConvolutionModule
from .embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
    LegacyRelPositionalEncoding,
)
from .encoder_layer_xattn import EncoderLayerWithCrossAttn
from .layer_norm import LayerNorm
from .multi_layer_conv import Conv1dLinear, MultiLayeredConv1d
from .positionwise_feed_forward import PositionwiseFeedForward
from .subsampling import Conv2dSubsampling


class MultiSequentialWithCrossAttn(torch.nn.Sequential):
    """Sequential that forwards (x, mask) positional args AND spk_emb kwarg."""

    def forward(self, *args, spk_emb=None):
        for m in self:
            args = m(*args, spk_emb=spk_emb)
        return args


class EncoderWithCrossAttn(torch.nn.Module):
    """Conformer encoder with per-layer cross-attention.

    Identical to Encoder in encoder.py except:
    - Uses EncoderLayerWithCrossAttn (adds cross-attn sub-layer)
    - forward() accepts spk_emb keyword argument
    - Uses MultiSequentialWithCrossAttn to propagate spk_emb
    """

    def __init__(
        self,
        idim=None,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer=None,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        encoder_attn_layer_type="mha",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
    ):
        super().__init__()

        if encoder_attn_layer_type == "rel_mha":
            pos_enc_class = RelPositionalEncoding
        elif encoder_attn_layer_type == "legacy_rel_mha":
            pos_enc_class = LegacyRelPositionalEncoding

        # Input embedding (same logic as encoder.py, simplified for our use case)
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError(f"Unsupported input_layer: {input_layer}")

        self.normalize_before = normalize_before

        # Positionwise feed-forward
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # Self-attention layer
        if encoder_attn_layer_type == "mha":
            encoder_attn_layer = MultiHeadedAttention
            encoder_attn_layer_args = (attention_heads, attention_dim, attention_dropout_rate)
        elif encoder_attn_layer_type == "legacy_rel_mha":
            encoder_attn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_attn_layer_args = (attention_heads, attention_dim, attention_dropout_rate)
        elif encoder_attn_layer_type == "rel_mha":
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (attention_heads, attention_dim, attention_dropout_rate, zero_triu)
        else:
            raise ValueError("unknown encoder_attn_layer: " + encoder_attn_layer_type)

        # Cross-attention uses plain MHA (no positional encoding — spk_emb has no sequence structure)
        cross_attn_layer = MultiHeadedAttention
        cross_attn_layer_args = (attention_heads, attention_dim, attention_dropout_rate)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        # Build the layer stack
        self.encoders = MultiSequentialWithCrossAttn(
            *[
                EncoderLayerWithCrossAttn(
                    attention_dim,
                    encoder_attn_layer(*encoder_attn_layer_args),
                    cross_attn_layer(*cross_attn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    macaron_style,
                )
                for _ in range(num_blocks)
            ]
        )

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks=None, spk_emb=None):
        """Encode input sequence with optional speaker cross-attention.

        Args:
            xs (torch.Tensor): Input features (B, T, D)
            masks (torch.Tensor): Mask (B, T) or None
            spk_emb (torch.Tensor): Speaker embedding (B, 1, D) or None

        Returns:
            torch.Tensor: Encoded output (B, T, D)
            torch.Tensor: Output mask
        """
        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks, spk_emb=spk_emb)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks


class ConformerEncoderWithCrossAttn(torch.nn.Module):
    """Wrapper matching ConformerEncoder API but with cross-attention support.

    Drop-in replacement for ConformerEncoder(size="L") in the model.
    """

    @staticmethod
    def lookup(size):
        lookup_table = {
            "S": {"num_blocks": 3, "attention_dim": 144, "attention_heads": 4},
            "M": {"num_blocks": 4, "attention_dim": 256, "attention_heads": 4},
            "L": {"num_blocks": 12, "attention_dim": 512, "attention_heads": 8},
        }
        return lookup_table[size]

    def __init__(self, size) -> None:
        super().__init__()
        kwargs = self.lookup(size)
        print(f"conformer encoder with cross-attention, details={kwargs}")
        self.encoder = EncoderWithCrossAttn(
            macaron_style=True,
            use_cnn_module=True,
            input_layer=None,
            **kwargs,
        )

    def forward(self, xs, masks=None, spk_emb=None):
        x, mask = self.encoder(xs, masks, spk_emb=spk_emb)
        return x
