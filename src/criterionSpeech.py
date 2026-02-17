# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_mcd(predicted, target, n_mfcc=13):
    """
    Compute Mel Cepstral Distortion (MCD) between predicted and target mel spectrograms.
    
    MCD = (10 * sqrt(2) / ln(10)) * ||MFCC_pred - MFCC_target||_2
    
    Args:
        predicted: (B, T, 80) predicted log-mel spectrogram
        target: (B, T, 80) target log-mel spectrogram
        n_mfcc: number of MFCC coefficients to use (typically 13, skip 0th)
    
    Returns:
        mcd: scalar MCD value averaged over batch
    """
    # Convert log-mel to MFCC using DCT (Discrete Cosine Transform)
    # MFCCs = DCT(log_mel_spec)
    B, T, D = predicted.shape
    
    # Create DCT matrix for converting mel to MFCC
    # We use type-II DCT
    n_mels = D
    dct_matrix = torch.zeros(n_mfcc, n_mels, device=predicted.device, dtype=predicted.dtype)
    for k in range(n_mfcc):
        for n in range(n_mels):
            dct_matrix[k, n] = math.cos(math.pi * k * (2 * n + 1) / (2 * n_mels))
    dct_matrix *= math.sqrt(2.0 / n_mels)
    
    # Apply DCT: (B, T, 80) @ (80, 13).T -> (B, T, 13)
    mfcc_pred = torch.matmul(predicted, dct_matrix.T)
    mfcc_target = torch.matmul(target, dct_matrix.T)
    
    # Skip 0th coefficient (energy), use 1:n_mfcc
    mfcc_pred = mfcc_pred[:, :, 1:n_mfcc]
    mfcc_target = mfcc_target[:, :, 1:n_mfcc]
    
    # Compute MCD
    # MCD = (10 * sqrt(2) / ln(10)) * mean(||mfcc_diff||_2)
    diff = mfcc_pred - mfcc_target
    frame_mcd = torch.sqrt((diff ** 2).sum(dim=-1))  # (B, T)
    
    # Average over time and batch
    mcd = frame_mcd.mean()
    
    # Scale factor for MCD in dB
    mcd_db = (10.0 * math.sqrt(2) / math.log(10)) * mcd
    
    return mcd_db


def compute_ssim(predicted, target, window_size=11, data_range=None):
    """
    Compute Structural Similarity Index (SSIM) between spectrograms.
    Treats spectrograms as 2D images (B, T, 80) -> (B, 1, T, 80) for 2D conv.
    
    Args:
        predicted: (B, T, 80) predicted mel spectrogram
        target: (B, T, 80) target mel spectrogram
        window_size: size of Gaussian window for local statistics
        data_range: dynamic range of the data (max - min), auto-computed if None
    
    Returns:
        ssim: scalar SSIM value (0 to 1, higher is better)
    """
    B, T, D = predicted.shape
    
    # Reshape to (B, 1, T, D) for 2D convolution
    pred_2d = predicted.unsqueeze(1)  # (B, 1, T, 80)
    tgt_2d = target.unsqueeze(1)
    
    # Auto-compute data range
    if data_range is None:
        data_range = max(target.max() - target.min(), predicted.max() - predicted.min())
        data_range = max(data_range.item(), 1e-6)  # Avoid division by zero
    
    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, device=predicted.device, dtype=predicted.dtype) - window_size // 2
    gauss_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # 2D Gaussian kernel
    gauss_2d = gauss_1d.outer(gauss_1d)
    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)
    
    # Pad to handle edges
    pad = window_size // 2
    
    # Compute local means using Gaussian filter
    mu_pred = F.conv2d(F.pad(pred_2d, (pad, pad, pad, pad), mode='reflect'), gauss_2d)
    mu_tgt = F.conv2d(F.pad(tgt_2d, (pad, pad, pad, pad), mode='reflect'), gauss_2d)
    
    mu_pred_sq = mu_pred ** 2
    mu_tgt_sq = mu_tgt ** 2
    mu_pred_tgt = mu_pred * mu_tgt
    
    # Compute local variances and covariance
    sigma_pred_sq = F.conv2d(F.pad(pred_2d ** 2, (pad, pad, pad, pad), mode='reflect'), gauss_2d) - mu_pred_sq
    sigma_tgt_sq = F.conv2d(F.pad(tgt_2d ** 2, (pad, pad, pad, pad), mode='reflect'), gauss_2d) - mu_tgt_sq
    sigma_pred_tgt = F.conv2d(F.pad(pred_2d * tgt_2d, (pad, pad, pad, pad), mode='reflect'), gauss_2d) - mu_pred_tgt
    
    # SSIM constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # SSIM formula
    ssim_map = ((2 * mu_pred_tgt + C1) * (2 * sigma_pred_tgt + C2)) / \
               ((mu_pred_sq + mu_tgt_sq + C1) * (sigma_pred_sq + sigma_tgt_sq + C2))
    
    # Average SSIM over all pixels and batch
    ssim = ssim_map.mean()
    
    return ssim


@dataclass
class MelSpectrogramL1LossConfig(FairseqDataclass):
    pass


@register_criterion("mel_spectrogram_l1_loss", dataclass=MelSpectrogramL1LossConfig)
class MelSpectrogramL1Loss(FairseqCriterion):
    def __init__(self, task, sentence_avg=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample."""
        
        # 1. Run model forward pass
        net_output = model(**sample["net_input"])
        
        # 2. Retrieve prediction and target
        predicted_mels = net_output["melspec"]
        target_mels = sample["target_mel"] 
        
        if target_mels is None:
             raise ValueError("Sample must contain 'target_mel' for MelSpectrogramL1Loss.")

        # Move target_mels to same device if needed
        if target_mels.device != predicted_mels.device:
            target_mels = target_mels.to(predicted_mels.device)
        
        B, T_pred, C = predicted_mels.shape
        B_tgt, T_tgt, C_tgt = target_mels.shape
        
        # Handle length mismatch
        if T_pred != T_tgt:
             min_len = min(T_pred, T_tgt)
             predicted_mels = predicted_mels[:, :min_len, :]
             target_mels = target_mels[:, :min_len, :]
        
        # Cast to fp32 for loss computation to prevent overflow
        predicted_mels_fp32 = predicted_mels.float()
        target_mels_fp32 = target_mels.float()
        
        # Compute L1 loss
        loss = F.l1_loss(predicted_mels_fp32, target_mels_fp32, reduction='none')
        
        # Apply masking using actual mel lengths
        mel_lengths = sample["mel_lengths"].to(loss.device)
        T_current = predicted_mels.size(1)
        mel_lengths = torch.clamp(mel_lengths, max=T_current)
        
        mask = torch.arange(T_current, device=loss.device).expand(B, T_current) < mel_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        
        masked_loss = loss * mask
        total_elements = mel_lengths.sum() * C
        
        loss_sum = masked_loss.sum()
        
        if reduce:
            if total_elements > 0:
                final_loss = loss_sum / total_elements
            else:
                final_loss = loss_sum
        else:
            final_loss = loss_sum
        
        sample_size = B

        logging_output = {
            "loss": final_loss.item(),
            "sample_size": sample_size,
            "ntokens": total_elements.item(),
            "nsentences": B,
        }
        
        # ===== VALIDATION METRICS (only computed when not training) =====
        if not model.training:
            with torch.no_grad():
                # Compute MCD
                try:
                    mcd = compute_mcd(predicted_mels_fp32, target_mels_fp32)
                    logging_output["mcd"] = mcd.item()
                except Exception as e:
                    logger.warning(f"MCD computation failed: {e}")
                    logging_output["mcd"] = 0.0
                
                # Compute SSIM
                try:
                    ssim = compute_ssim(predicted_mels_fp32, target_mels_fp32)
                    logging_output["ssim"] = ssim.item()
                except Exception as e:
                    logger.warning(f"SSIM computation failed: {e}")
                    logging_output["ssim"] = 0.0

        return final_loss, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        n_batches = len(logging_outputs)
        
        if n_batches > 0:
            metrics.log_scalar("loss", loss_sum / n_batches, priority=100, round=3)
        
        # Aggregate MCD (validation)
        mcd_sum = sum(log.get("mcd", 0) for log in logging_outputs)
        mcd_count = sum(1 for log in logging_outputs if "mcd" in log)
        if mcd_count > 0:
            metrics.log_scalar("mcd", mcd_sum / mcd_count, priority=90, round=3)
        
        # Aggregate SSIM (validation)
        ssim_sum = sum(log.get("ssim", 0) for log in logging_outputs)
        ssim_count = sum(1 for log in logging_outputs if "ssim" in log)
        if ssim_count > 0:
            metrics.log_scalar("ssim", ssim_sum / ssim_count, priority=80, round=4)
