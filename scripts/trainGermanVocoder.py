"""
German Pre-training Script for HiFi-GAN Vocoder
================================================
Pre-trains HiFi-GAN on German mel spectrograms, optionally initializing
from an English pre-trained checkpoint (warm start).

This is PRE-TRAINING (real mels from real audio), not fine-tuning.

Usage:
    python trainGermanVocoder.py

Configure the paths in the CONFIGURATION section before running.
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as transforms
import random

# Add custom_hifigan to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "custom_hifigan"))

from hifigan.generator import HifiganGenerator
from hifigan.discriminator import (
    HifiganDiscriminator,
    feature_loss,
    discriminator_loss,
    generator_loss,
)
from hifigan.utils import plot_spectrogram
from hifigan.constants import mel_bins
import copy

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, ema_model, beta=0.999):
        self.beta = beta
        self.ema_model = ema_model
        
        # Initialize EMA weights to match source model exactly
        self.ema_model.load_state_dict(model.state_dict())
        
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.beta).add_(param.data, alpha=1 - self.beta)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


def save_checkpoint_minimal(
    checkpoint_dir,
    generator,
    ema_generator,  # Added EMA generator
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    step,
    loss,
    save_type,  # 'best' or 'last'
    logger,
):
    """Save checkpoint as either best or last only (no intermediate checkpoints)."""
    is_ddp = isinstance(generator, torch.nn.parallel.DistributedDataParallel)
    state = {
        "generator": {
            "model": generator.module.state_dict() if is_ddp else generator.state_dict(),
            "optimizer": optimizer_generator.state_dict(),
            "scheduler": scheduler_generator.state_dict(),
        },
        "ema_generator": ema_generator.state_dict() if ema_generator else None, # Save EMA
        "discriminator": {
            "model": discriminator.state_dict(),
            "optimizer": optimizer_discriminator.state_dict(),
            "scheduler": scheduler_discriminator.state_dict(),
        },
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    if save_type == 'best':
        checkpoint_path = checkpoint_dir / "model-best.pt"
    else:  # 'last'
        checkpoint_path = checkpoint_dir / "model-last.pt"
    
    torch.save(state, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")

# ==============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ==============================================================================

# Path to training manifest (.tsv file)
TRAIN_MANIFEST = "/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/train.tsv"

# Path to validation manifest (.tsv file)
VALID_MANIFEST = "/ceph/home/TUG/olivares-tug/MMS-LLaMA/manifest/germanManifest/valid.tsv"

# Mel spectrogram file suffix (appended to wav basename)
MEL_SUFFIX = "_pred.pt"

# Path to English pre-trained checkpoint for warm start (set to None to train from scratch)
PRETRAINED_CHECKPOINT = "/ceph/home/TUG/olivares-tug/DiVISe/exp-vocoder/exp2-New-checkpointing-more-patience/model-best.pt"

# Base output directory for checkpoints (experiments will be created inside)
CHECKPOINT_BASE_DIR = "/ceph/home/TUG/olivares-tug/DiVISe/exp-vocoder"

# Optional comment for this experiment (leave empty for no comment)
# The folder will be named: exp1, exp2, ... or exp1-your_comment, exp2-your_comment, ...
EXPERIMENT_COMMENT = "finetuning-on-german-EMA-bz-64"

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

BATCH_SIZE = 64
SEGMENT_LENGTH = 8320  # Audio samples per segment
HOP_LENGTH = 160       # Mel hop length (for 100Hz frame rate at 16kHz: 160)
SAMPLE_RATE = 16000
LEARNING_RATE = 1e-4   # Pre-training learning rate
BETAS = (0.8, 0.99)
LEARNING_RATE_DECAY = 0.999
WEIGHT_DECAY = 1e-5
MAX_UPDATES = 400_000
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 5000
NUM_GENERATED_EXAMPLES = 10
PATIENCE = 10  # Early stopping patience (based on validation mel loss)
EMA_DECAY = 0.999   # EMA decay rate

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LogMelSpectrogram(torch.nn.Module):
    """Compute log mel spectrogram for target comparison."""
    def __init__(self, n_fft=1024, num_mels=mel_bins, hop_size=HOP_LENGTH, win_size=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=self.n_fft,
            win_length=win_size,
            hop_length=self.hop_size,
            center=False,
            power=2.0,          # Match librosa default (power spectrogram)
            norm=None,          # Match librosa default (no slaney normalization)
            f_min=0,            # Explicit
            f_max=8000,         # Explicit (matches preprocessing)
            onesided=True,
            n_mels=num_mels,
            mel_scale="slaney",
        )

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


class GermanMelDataset(Dataset):
    """
    Dataset for German pre-training using offline-generated mel spectrograms.
    
    Reads a TSV manifest and loads corresponding .pt mel files.
    """
    def __init__(
        self,
        manifest_path: str,
        mel_suffix: str,
        segment_length: int,
        sample_rate: int,
        hop_length: int,
        train: bool = True,
    ):
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.train = train
        self.mel_suffix = mel_suffix
        
        # Parse manifest
        self.wav_paths = []
        self.mel_paths = []
        
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header line (starts with /)
        for line in lines:
            line = line.strip()
            if not line or line.startswith('/'):
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                wav_path = parts[2]  # Third column is wav path
                self.wav_paths.append(wav_path)
                
                # Mel path: same directory, same basename + mel_suffix
                mel_path = wav_path.replace('.wav', mel_suffix)
                self.mel_paths.append(mel_path)
        
        # Filter out samples where mel file doesn't exist
        valid_indices = []
        for i, mel_path in enumerate(self.mel_paths):
            if os.path.exists(mel_path):
                valid_indices.append(i)
            else:
                if i < 5:  # Only warn for first few
                    logger.warning(f"Mel file not found: {mel_path}")
        
        self.wav_paths = [self.wav_paths[i] for i in valid_indices]
        self.mel_paths = [self.mel_paths[i] for i in valid_indices]
        
        logger.info(f"Loaded {len(self.wav_paths)} samples from {manifest_path}")
        
        self.logmel = LogMelSpectrogram()

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        mel_path = self.mel_paths[index]
        
        # Load mel spectrogram
        src_logmel = torch.load(mel_path, map_location='cpu')
        
        # Mels are saved as (time, mel_bins) -> transpose to (mel_bins, time)
        if src_logmel.dim() == 3:
            src_logmel = src_logmel.squeeze(0)
        
        # Always transpose: (time, 128) -> (128, time)
        src_logmel = src_logmel.transpose(0, 1)
        
        src_logmel = src_logmel.unsqueeze(0)  # Add batch dim: (1, mel_bins, time)
        
        # Calculate mel frames per audio segment
        mel_frames_per_segment = math.ceil(self.segment_length / self.hop_length)
        
        # Random offset for training
        if self.train:
            mel_diff = src_logmel.size(-1) - mel_frames_per_segment
            mel_offset = random.randint(0, max(mel_diff, 0))
        else:
            mel_offset = 0
        
        frame_offset = self.hop_length * mel_offset
        
        # Load corresponding audio segment
        try:
            wav, sr = torchaudio.load(
                wav_path,
                frame_offset=frame_offset if self.train else 0,
                num_frames=self.segment_length if self.train else -1,
            )
        except Exception as e:
            logger.error(f"Error loading {wav_path}: {e}")
            # Return a dummy sample
            wav = torch.zeros(1, self.segment_length)
            src_logmel = torch.zeros(1, mel_bins, mel_frames_per_segment)
            tgt_logmel = torch.zeros(mel_bins, mel_frames_per_segment)
            return wav.squeeze(0), src_logmel, tgt_logmel
        
        if sr != self.sample_rate:
            raise ValueError(f"Sample rate {sr} doesn't match {self.sample_rate}")
        
        # Pad if too short
        if wav.size(-1) < self.segment_length:
            wav = F.pad(wav, (0, self.segment_length - wav.size(-1)))
        
        # Compute target mel from audio (ground truth)
        tgt_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)
        
        # Slice source mel to match segment
        if self.train:
            src_logmel = src_logmel[:, :, mel_offset:mel_offset + mel_frames_per_segment]
        
        # Pad source mel if needed
        if src_logmel.size(-1) < mel_frames_per_segment:
            src_logmel = F.pad(
                src_logmel,
                (0, mel_frames_per_segment - src_logmel.size(-1)),
                "constant",
                src_logmel.min(),
            )
        
        return wav, src_logmel, tgt_logmel


def load_pretrained_weights(checkpoint_path: str, generator, discriminator, rank, logger):
    """Load only model weights from checkpoint (no optimizer states)."""
    logger.info(f"Loading pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location={"cuda:0": f"cuda:{rank}"})
    
    # Handle different checkpoint formats
    if "generator" in checkpoint:
        gen_state = checkpoint["generator"]["model"]
        disc_state = checkpoint["discriminator"]["model"]
    else:
        # Direct state dict format
        gen_state = checkpoint
        disc_state = None
    
    # Remove "module." prefix if present
    gen_state = {k.replace("module.", ""): v for k, v in gen_state.items()}
    
    # Load generator
    incompatible = generator.load_state_dict(gen_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(f"Incompatible keys: {incompatible}")
    
    # Load discriminator if available
    if disc_state is not None:
        disc_state = {k.replace("module.", ""): v for k, v in disc_state.items()}
        discriminator.load_state_dict(disc_state)
    
    logger.info("Pretrained weights loaded successfully (warm start)")


def train_model(rank, world_size, args):
    """Main training function."""
    if world_size > 1:
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            init_method="tcp://localhost:54321",
        )

    log_dir = Path(args.checkpoint_dir) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if rank == 0:
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_dir / "training.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    # Create models
    generator = HifiganGenerator().to(rank)
    discriminator = HifiganDiscriminator().to(rank)

    # Load pretrained weights (warm start)
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        load_pretrained_weights(
            args.pretrained_checkpoint, generator, discriminator, rank, logger
        )
    
    # Initialize EMA Generator
    # Safely create a fresh model instead of using deepcopy (which breaks weight_norm)
    if rank == 0:
        ema_generator_model = HifiganGenerator().to(rank)
        ema_generator = EMA(generator, ema_model=ema_generator_model, beta=EMA_DECAY)
    else:
        ema_generator = None
    
    # Create optimizers (fresh, not loaded from checkpoint)
    optimizer_generator = optim.AdamW(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_discriminator = optim.AdamW(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_generator = optim.lr_scheduler.ExponentialLR(
        optimizer_generator, gamma=LEARNING_RATE_DECAY
    )
    scheduler_discriminator = optim.lr_scheduler.ExponentialLR(
        optimizer_discriminator, gamma=LEARNING_RATE_DECAY
    )

    # Start from scratch (pre-training)
    global_step = 0
    best_loss = float("inf")
    patience_counter = 0

    if world_size > 1:
        generator = DDP(generator, device_ids=[rank])
        discriminator = DDP(discriminator, device_ids=[rank])

    # Create datasets
    train_dataset = GermanMelDataset(
        manifest_path=args.train_manifest,
        mel_suffix=args.mel_suffix,
        segment_length=SEGMENT_LENGTH,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        train=True,
    )
    train_sampler = DistributedSampler(train_dataset, drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8 if world_size > 1 else 0,
        pin_memory=True,
        shuffle=(train_sampler is None),
        drop_last=True,
    )

    validation_dataset = GermanMelDataset(
        manifest_path=args.valid_manifest,
        mel_suffix=args.mel_suffix,
        segment_length=SEGMENT_LENGTH,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        train=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8 if world_size > 1 else 0,
        pin_memory=True,
    )

    melspectrogram = LogMelSpectrogram().to(rank)

    n_epochs = math.ceil(MAX_UPDATES / (len(train_loader) * world_size))
    start_epoch = 1

    logger.info("=" * 80)
    logger.info("GERMAN VOCODER PRE-TRAINING (Warm Start from English)")
    logger.info("=" * 80)
    logger.info(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
    logger.info(f"Learning rate: {LEARNING_RATE} (pre-training rate)")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(validation_dataset)}")
    logger.info(f"Iterations per epoch: {len(train_loader)}")
    logger.info(f"Total epochs: {n_epochs}")
    logger.info("=" * 80)

    should_stop = False
    
    for epoch in range(start_epoch, n_epochs + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()
        average_loss_mel = average_loss_discriminator = average_loss_generator = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (wavs, inputs, tgts) in enumerate(pbar, 1):
            wavs, inputs, tgts = wavs.to(rank), inputs.to(rank), tgts.to(rank)

            # Discriminator step
            optimizer_discriminator.zero_grad()

            wavs_ = generator(inputs.squeeze(1))
            mels_ = melspectrogram(wavs_)

            scores, _ = discriminator(wavs)
            scores_, _ = discriminator(wavs_.detach())

            loss_discriminator, _, _ = discriminator_loss(scores, scores_)

            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Generator step
            optimizer_generator.zero_grad()

            scores, features = discriminator(wavs)
            scores_, features_ = discriminator(wavs_)

            loss_mel = F.l1_loss(mels_, tgts)
            loss_features = feature_loss(features, features_)
            loss_generator_adversarial, _ = generator_loss(scores_)
            loss_generator = 45 * loss_mel + loss_features + loss_generator_adversarial

            loss_generator.backward()
            optimizer_generator.step()

            # Update EMA Generator
            if rank == 0:
                ema_generator.update(generator)

            global_step += 1

            average_loss_mel += (loss_mel.item() - average_loss_mel) / i
            average_loss_discriminator += (loss_discriminator.item() - average_loss_discriminator) / i
            average_loss_generator += (loss_generator.item() - average_loss_generator) / i

            pbar.set_postfix({
                'mel': f'{average_loss_mel:.4f}',
                'gen': f'{average_loss_generator:.4f}',
                'disc': f'{average_loss_discriminator:.4f}'
            })

            if rank == 0:
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("train/loss_mel", loss_mel.item(), global_step)
                    writer.add_scalar("train/loss_generator", loss_generator.item(), global_step)
                    writer.add_scalar("train/loss_discriminator", loss_discriminator.item(), global_step)

            # Validation
            if global_step % VALIDATION_INTERVAL == 0:
                generator.eval()

                average_validation_loss = 0
                for j, (wavs, inputs, tgts) in enumerate(validation_loader, 1):
                    wavs, inputs, tgts = wavs.to(rank), inputs.to(rank), tgts.to(rank)

                    with torch.no_grad():
                        wavs_ = generator(inputs.squeeze(1))
                        mels_ = melspectrogram(wavs_)

                        length = min(mels_.size(-1), tgts.size(-1))
                        loss_mel = F.l1_loss(mels_[..., :length], tgts[..., :length])

                    average_validation_loss += (loss_mel.item() - average_validation_loss) / j

                    if rank == 0 and j <= NUM_GENERATED_EXAMPLES:
                        writer.add_audio(
                            f"generated/wav_{j}",
                            wavs_.squeeze(0),
                            global_step,
                            sample_rate=SAMPLE_RATE,
                        )
                        writer.add_figure(
                            f"generated/mel_{j}",
                            plot_spectrogram(mels_.squeeze().cpu().numpy()),
                            global_step,
                        )

                generator.train()
                discriminator.train()

                if rank == 0:
                    writer.add_scalar("validation/mel_loss", average_validation_loss, global_step)
                    logger.info(f"Validation -- step: {global_step}, mel loss: {average_validation_loss:.4f}")

                new_best = best_loss > average_validation_loss
                if new_best:
                    if rank == 0:
                        save_checkpoint_minimal(
                            checkpoint_dir=Path(args.checkpoint_dir),
                            generator=generator,
                            ema_generator=ema_generator,
                            discriminator=discriminator,
                            optimizer_generator=optimizer_generator,
                            optimizer_discriminator=optimizer_discriminator,
                            scheduler_generator=scheduler_generator,
                            scheduler_discriminator=scheduler_discriminator,
                            step=global_step,
                            loss=average_validation_loss,
                            save_type='best',
                            logger=logger,
                        )
                
                if new_best:
                    best_loss = average_validation_loss
                    logger.info(f"New best model! Loss: {best_loss:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    logger.info(f"Early stopping triggered after {patience_counter} validations without improvement")
                    should_stop = True
                    break

        if should_stop:
            break

        scheduler_discriminator.step()
        scheduler_generator.step()

        logger.info(
            f"Epoch {epoch} -- mel: {average_loss_mel:.4f}, gen: {average_loss_generator:.4f}, disc: {average_loss_discriminator:.4f}"
        )

    if world_size > 1:
        dist.destroy_process_group()
    
    # Save final (last) checkpoint
    if rank == 0:
        save_checkpoint_minimal(
            checkpoint_dir=Path(args.checkpoint_dir),
            generator=generator,
            ema_generator=ema_generator,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            scheduler_generator=scheduler_generator,
            scheduler_discriminator=scheduler_discriminator,
            step=global_step,
            loss=best_loss,
            save_type='last',
            logger=logger,
        )
        logger.info(f"Saved final checkpoint at step {global_step}")
    
    logger.info("Training complete!")


def get_next_experiment_dir(base_dir: str, comment: str = "") -> str:
    """Find next available experiment number and create directory name."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Find existing exp folders
    existing = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith('exp')]
    
    # Extract numbers
    numbers = []
    for name in existing:
        try:
            # Handle both 'exp1' and 'exp1-comment' formats
            num_part = name.split('-')[0].replace('exp', '')
            numbers.append(int(num_part))
        except ValueError:
            continue
    
    next_num = max(numbers, default=0) + 1
    
    # Create folder name
    if comment:
        folder_name = f"exp{next_num}-{comment}"
    else:
        folder_name = f"exp{next_num}"
    
    return str(base_path / folder_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="German Pre-training for HiFi-GAN Vocoder")
    parser.add_argument("--train_manifest", default=TRAIN_MANIFEST, help="Path to training manifest")
    parser.add_argument("--valid_manifest", default=VALID_MANIFEST, help="Path to validation manifest")
    parser.add_argument("--mel_suffix", default=MEL_SUFFIX, help="Mel file suffix")
    parser.add_argument("--pretrained_checkpoint", default=PRETRAINED_CHECKPOINT, help="English pretrained checkpoint")
    parser.add_argument("--comment", default=EXPERIMENT_COMMENT, help="Comment for this experiment")
    args = parser.parse_args()
    
    # Create experiment directory
    args.checkpoint_dir = get_next_experiment_dir(CHECKPOINT_BASE_DIR, args.comment)

    # Display info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"# of GPUs: {torch.cuda.device_count()}")

    logger.handlers.clear()

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            train_model,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )
    else:
        train_model(0, world_size, args)
