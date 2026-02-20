from pathlib import Path
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

import torchaudio
import torchaudio.transforms as transforms
from hifigan.constants import mel_bins


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, n_fft=1024, num_mels=mel_bins, hop_size=160, win_size=1024):
        super().__init__()
        self.n_fft=n_fft
        self.hop_size=hop_size
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=self.n_fft,
            win_length=win_size,
            hop_length=self.hop_size,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=num_mels,
            mel_scale="slaney",
        )

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft-self.hop_size) // 2, (self.n_fft-self.hop_size) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


class BaseDataset(Dataset):
    def __init__(
        self,
        root: Path,
        segment_length: int,
        sample_rate: int,
        hop_length: int,
        train: bool = True,
        finetune: bool = False,
    ):
        self.root = root
        self.wavs_dir = root / "wavs"
        self.mels_dir = root / "mels"
        self.data_dir = self.wavs_dir if not finetune else self.mels_dir

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.train = train
        self.finetune = finetune

        suffix = ".wav" if not finetune else ".npy"
        file_path = "training.txt" if train else "validation.txt"
        file_path = root / file_path
        with open(file_path, "r") as f:
            self.metadata = f.readlines()
        self.metadata = [x.split("|")[0].strip()+suffix for x in self.metadata]

        self.logmel = LogMelSpectrogram()

    def __len__(self):
        return len(self.metadata)

class MelDataset(BaseDataset):
    def __init__(
        self,
        root: Path,
        segment_length: int,
        sample_rate: int,
        hop_length: int,
        train: bool = True,
        finetune: bool = False,
    ):
        super().__init__(root, segment_length, sample_rate, hop_length, train, finetune)

    def __getitem__(self, index):
        path = self.metadata[index]
        wav_path = self.wavs_dir / path

        info = torchaudio.info(wav_path.with_suffix(".wav"))
        if info.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
            )

        if self.finetune:
            mel_path = self.mels_dir / path
            src_logmel = torch.from_numpy(np.load(mel_path.with_suffix(".npy")))
            src_logmel = src_logmel.unsqueeze(0)

            mel_frames_per_segment = math.ceil(self.segment_length / self.hop_length)
            mel_diff = src_logmel.size(-1) - mel_frames_per_segment if self.train else 0
            mel_offset = random.randint(0, max(mel_diff, 0))

            frame_offset = self.hop_length * mel_offset
        else:
            frame_diff = info.num_frames - self.segment_length
            frame_offset = random.randint(0, max(frame_diff, 0))

        wav, _ = torchaudio.load(
            wav_path.with_suffix(".wav"),
            frame_offset=frame_offset if self.train else 0,
            num_frames=self.segment_length if self.train else -1,
        )

        if wav.size(-1) < self.segment_length:
            wav = F.pad(wav, (0, self.segment_length - wav.size(-1)))

        if not self.finetune and self.train:
            gain = random.random() * (0.99 - 0.4) + 0.4
            flip = -1 if random.random() > 0.5 else 1
            wav = flip * gain * wav / max(wav.abs().max(), 1e-5)

        tgt_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)

        if self.finetune:
            if self.train:
                src_logmel = src_logmel[
                    :, :, mel_offset : mel_offset + mel_frames_per_segment
                ]

            if src_logmel.size(-1) < mel_frames_per_segment:
                src_logmel = F.pad(
                    src_logmel,
                    (0, mel_frames_per_segment - src_logmel.size(-1)),
                    "constant",
                    src_logmel.min(),
                )
        else:
            src_logmel = tgt_logmel.clone()

        return wav, src_logmel, tgt_logmel

class HuBERTLabelDataset(BaseDataset):
    def __init__(self, root: Path, 
                 segment_length: int, 
                 sample_rate: int, 
                 hop_length: int, 
                 train: bool = True, 
                 sub_dirname: str = "30h_km",
                 num_classes=2000,
                 ):
        super().__init__(root, segment_length, sample_rate, hop_length, train, False)
        self.pad_idx = num_classes
        file_path = "training.km" if train else "validation.km"
        file_path = self.root / sub_dirname / file_path
        with open(file_path, "r") as f:
            self.km_labels = f.readlines()
        assert len(self.km_labels) == len(self.metadata), f"{len(self.km_labels)=} does not match lines in tsv files." \
                "Please check if they are on the same split."
        
    def __getitem__(self, index):
        path = self.metadata[index]
        wav_path = self.wavs_dir / path

        info = torchaudio.info(wav_path.with_suffix(".wav"))
        if info.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
            )
        src_label = [int(x) for x in self.km_labels[index].strip().split(' ')]
        label_frames_per_segment = math.ceil(self.segment_length / self.hop_length)
        label_diff = len(src_label) - label_frames_per_segment if self.train else 0
        label_offset = random.randint(0, max(label_diff, 0))

        frame_offset = self.hop_length * label_offset

        wav, _ = torchaudio.load(
            wav_path.with_suffix(".wav"),
            frame_offset=frame_offset if self.train else 0,
            num_frames=self.segment_length if self.train else -1,
        )

        if wav.size(-1) < self.segment_length:
            wav = F.pad(wav, (0, self.segment_length - wav.size(-1)))

        if self.train:
            gain = random.random() * (0.99 - 0.4) + 0.4
            flip = -1 if random.random() > 0.5 else 1
            wav = flip * gain * wav / max(wav.abs().max(), 1e-5)

        tgt_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)
        
        if self.train:
            src_label = src_label[
                label_offset : label_offset + label_frames_per_segment
            ]

        src_label = torch.LongTensor(src_label)
        if len(src_label) < label_frames_per_segment:
            src_label = F.pad(
                src_label,
                (0, label_frames_per_segment - len(src_label)),
                "constant",
                self.pad_idx,
            )

        return wav, src_label.unsqueeze(0), tgt_logmel
    
class LRS3MelDataset(Dataset):
    def __init__(
        self,
        root: Path,
        segment_length: int,
        sample_rate: int,
        hop_length: int,
        train: bool = True,
        npy_postfix: str = None,
    ):
        self.root = root

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.train = train
        self.finetune = npy_postfix is not None

        file_path = "train.tsv" if train else "valid.tsv"
        file_path = root / file_path
        self.metadata, inds, tot, sizes = self.load_audio_visual_simple(file_path, 500, None)
        self.metadata = [str(root.parent / "audio" / x.split("\t")[0].strip()) for x in self.metadata[1:]]
        self.wav_paths = [x+'.wav' for x in self.metadata]
        if self.finetune:
            # expecting npy_postfix be like 433h_lrs3
            self.mel_paths = [x+f"_{npy_postfix}.npy" for x in self.metadata]

        self.logmel = LogMelSpectrogram()

    def load_audio_visual_simple(self, manifest_path, max_keep, min_keep):
        n_long, n_short = 0, 0
        names, inds, sizes = [], [], []

        with open(manifest_path) as f:
            root = f.readline().strip()
            for ind, line in enumerate(f):
                items = line.strip().split("\t")
                sz = int(items[-2]) # 
                if min_keep is not None and sz < min_keep:
                    n_short += 1
                elif max_keep is not None and sz > max_keep:
                    n_long += 1
                else:
                    audio_id = items[0]
                    names.append(audio_id)
                    inds.append(ind)
                    sizes.append(sz)
        tot = ind + 1
        print(
            (
                f"max_keep={max_keep}, min_keep={min_keep}, "
                f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
                f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
            )
        )
        return names, inds, tot, sizes
    
    def __getitem__(self, index):
        wav_path = self.wav_paths[index]

        info = torchaudio.info(wav_path)
        if info.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
            )

        if self.finetune:
            mel_path = self.mel_paths[index]
            src_logmel = torch.from_numpy(np.load(mel_path)).transpose(-1, -2)
            src_logmel = src_logmel.unsqueeze(0)

            mel_frames_per_segment = math.ceil(self.segment_length / self.hop_length)
            mel_diff = src_logmel.size(-1) - mel_frames_per_segment if self.train else 0
            mel_offset = random.randint(0, max(mel_diff, 0))

            frame_offset = self.hop_length * mel_offset
        else:
            frame_diff = info.num_frames - self.segment_length
            frame_offset = random.randint(0, max(frame_diff, 0))

        wav, _ = torchaudio.load(
            wav_path,
            frame_offset=frame_offset if self.train else 0,
            num_frames=self.segment_length if self.train else -1,
        )

        if wav.size(-1) < self.segment_length:
            wav = F.pad(wav, (0, self.segment_length - wav.size(-1)))

        if not self.finetune and self.train:
            gain = random.random() * (0.99 - 0.4) + 0.4
            flip = -1 if random.random() > 0.5 else 1
            wav = flip * gain * wav / max(wav.abs().max(), 1e-5)

        tgt_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)

        if self.finetune:
            if self.train:
                src_logmel = src_logmel[
                    :, :, mel_offset : mel_offset + mel_frames_per_segment
                ]

            if src_logmel.size(-1) < mel_frames_per_segment:
                src_logmel = F.pad(
                    src_logmel,
                    (0, mel_frames_per_segment - src_logmel.size(-1)),
                    "constant",
                    src_logmel.min(),
                )
        else:
            src_logmel = tgt_logmel.clone()

        return wav, src_logmel, tgt_logmel
    
    def __len__(self):
        return len(self.metadata)