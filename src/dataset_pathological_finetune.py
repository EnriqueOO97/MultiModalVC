# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset for pathological -> healthy voice-conversion fine-tuning.

Manifest format (tab-separated, with a root path on the first line):

    <root_path>
    <video.mp4>  <patho.wav>  <healthy.wav>  <synth1.wav> ... <synth6.wav>
    ...

Per row: 1 pathological audio (always the INPUT) and 7 candidate TARGETS
(healthy + 6 synthetic-of-healthy variants). Per ``__getitem__`` call one
target is sampled uniformly at random; after ~7 epochs each (input, target)
pair has been visited at least once.

xvectors are NOT in the manifest — they are derived from the chosen target's
wav path by replacing ``.wav`` with ``_xvector.pt``.
"""

import logging
import os
import random
import sys
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from fairseq.data.fairseq_dataset import FairseqDataset
from scipy.io import wavfile
from transformers import WhisperProcessor

from . import utils as custom_utils

logger = logging.getLogger(__name__)


def _xvector_path(wav_path: str) -> str:
    """Convert ``<stem>.wav`` -> ``<stem>_xvector.pt`` (naming convention)."""
    if wav_path.endswith(".wav"):
        return wav_path[:-4] + "_xvector.pt"
    raise ValueError(f"Expected .wav path, got: {wav_path}")


def _count_video_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n if n > 0 else 1


def load_pathological_manifest(manifest_path: str):
    """Parse the 9-column pathological fine-tune manifest.

    Returns:
        root:        Root path prefix (first line of manifest).
        video_paths: list[str]            — video for each entry
        patho_paths: list[str]            — single pathological input per entry
        target_lists: list[list[str]]     — 7 candidate target paths per entry
        sizes:       list[int]            — video frame counts (for fairseq batching)
    """
    video_paths, patho_paths, target_lists, sizes = [], [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for line_no, line in enumerate(f, start=2):
            items = line.strip().split("\t")
            if len(items) != 9:
                raise ValueError(
                    f"{manifest_path}:{line_no}: expected 9 columns, got {len(items)}"
                )
            video, patho, healthy, *synths = items
            video_paths.append(video)
            patho_paths.append(patho)
            target_lists.append([healthy] + synths)

    if not video_paths:
        raise RuntimeError(f"No entries in {manifest_path}")

    for vp in video_paths:
        full = vp if os.path.isabs(vp) else os.path.join(root, vp)
        sizes.append(_count_video_frames(full))

    logger.info(
        f"Loaded {len(video_paths)} entries from {manifest_path} "
        f"(root={root!r}, max_frames={max(sizes)}, min_frames={min(sizes)})"
    )
    return root, video_paths, patho_paths, target_lists, sizes


class mms_pathological_finetune_dataset(FairseqDataset):
    """Pathological -> healthy fine-tune dataset.

    Batch contract is identical to ``mms_synthvc_dataset`` (same keys / shapes),
    so the existing model and criterion consume it unchanged. The only
    semantic difference: the ``synth_audio`` slot carries the pathological
    input, and ``target_waveform_clean`` carries a randomly-rotated healthy
    target (1 of 7) per call.
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16_000,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        normalize: bool = False,
        image_mean: float = 0.0,
        image_std: float = 1.0,
        image_crop_size: int = 88,
        image_aug: bool = False,
        modalities: Optional[List[str]] = None,
        subset_name: Optional[str] = None,
    ):
        super().__init__()
        self.modalities = set(modalities) if modalities is not None else {"audio", "video"}

        (
            self.audio_root,
            self.video_paths,
            self.patho_paths,
            self.target_lists,
            self.sizes,
        ) = load_pathological_manifest(manifest_path)

        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.normalize = normalize
        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize

        self.subset = manifest_path.split("/")[-1].split(".")[0]
        self.subset_name = subset_name if subset_name is not None else self.subset

        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize(0.0, 255.0),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std),
            ])
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize(0.0, 255.0),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std),
            ])
        logger.info(f"[pathological-finetune] subset={self.subset_name} image_aug={image_aug}")

    # ---------- helpers (mirror mms_synthvc_dataset) ----------

    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.audio_root, path)

    def _load_video(self, path: str):
        feats = custom_utils.load_video(self._resolve(path))
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def _load_wav(self, path: str):
        sr, wav_data = wavfile.read(self._resolve(path))
        assert sr == 16_000 and wav_data.ndim == 1, f"bad wav: {path}"
        if wav_data.dtype == np.int16:
            wav_data = wav_data / 32768.0
        return wav_data.astype(np.float32), sr

    def _wav_to_whisper(self, wav_data, sample_rate):
        return self.whisper_processor(
            wav_data, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features

    # ---------- core ----------

    def __getitem__(self, index):
        # Video
        video_feats = None
        if "video" in self.modalities:
            video_feats = self._load_video(self.video_paths[index])
            video_feats = torch.from_numpy(video_feats.astype(np.float32))

        # Pathological audio -> goes into BOTH `audio_source` and `synth_audio_source`
        # slots so the model receives it whichever forward path it uses.
        patho_path = self.patho_paths[index]
        patho_wav, sr = self._load_wav(patho_path)
        patho_len = int(len(patho_wav))
        patho_whisper = self._wav_to_whisper(patho_wav, sr)

        # Pick 1 of 7 targets uniformly at random per __getitem__ call
        target_paths = self.target_lists[index]
        target_idx = random.randint(0, len(target_paths) - 1)
        target_path = target_paths[target_idx]
        target_wav, _ = self._load_wav(target_path)
        target_waveform = torch.from_numpy(target_wav).float()

        # Speaker embedding from the chosen target (matches the target speaker timbre)
        spk_path = _xvector_path(self._resolve(target_path))
        spk_embedding = torch.load(spk_path, map_location="cpu", weights_only=True)
        if spk_embedding.dim() > 1:
            spk_embedding = spk_embedding.squeeze()

        fid = os.path.splitext(os.path.basename(patho_path))[0]
        # Speech rate is unused at training time — populate with a neutral placeholder
        sr_label = torch.tensor(0.0, dtype=torch.float32)

        return {
            "id": index,
            "fid": fid,
            "video_source": video_feats,
            "audio_source": patho_whisper,
            "audio_len_samples": patho_len,
            "sr_label": sr_label,
            "target_waveform": target_waveform,         # Phase-1-style GT slot (unused in Phase 2)
            "target_waveform_clean": target_waveform,   # Phase 2 GT
            "spk_embedding": spk_embedding,
            "synth_audio_source": patho_whisper,
            "synth_audio_len_samples": patho_len,
        }

    def __len__(self):
        return len(self.sizes)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    # ---------- collater (matches mms_synthvc_dataset batch contract) ----------

    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        audio_source = [s["audio_source"] for s in samples]
        video_source = [s["video_source"] for s in samples]
        audio_len_samples = [s["audio_len_samples"] for s in samples]

        if video_source[0] is None:
            raise RuntimeError("video modality is required for pathological finetune")

        video_sizes = [len(v) for v in video_source]
        video_size = min(max(video_sizes), self.max_sample_size)

        collated_audios = self._collater_whisper(audio_source)
        collated_videos, padding_mask, _ = self._collater_video(video_source, video_size)

        sr_labels = torch.stack([s["sr_label"] for s in samples])

        source = {
            "audio": collated_audios,
            "video": collated_videos,
            "audio_lengths": torch.tensor(audio_len_samples, dtype=torch.long),
        }
        net_input = {"source": source, "padding_mask": padding_mask}

        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "utt_id": [s["fid"] for s in samples],
            "subset_name": self.subset_name,
            "sr_labels": sr_labels,
        }

        # Targets — pad to max length in batch
        targets = [s["target_waveform"] for s in samples]
        wav_lens = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
        max_wav = int(wav_lens.max().item())
        padded = targets[0].new_zeros(len(targets), max_wav)
        for i, t in enumerate(targets):
            padded[i, : t.size(0)] = t
        batch["target_waveform"] = padded
        batch["waveform_lengths"] = wav_lens
        batch["target_waveform_clean"] = padded  # same tensor — no noise in this stage

        # Speaker embeddings
        batch["spk_embeddings"] = torch.stack([s["spk_embedding"] for s in samples])

        # Pathological input in the synth_audio slot (same Whisper features as `audio`)
        synth_sources = [s["synth_audio_source"] for s in samples]
        batch["synth_audio"] = self._collater_whisper(synth_sources)
        batch["synth_audio_lengths"] = torch.tensor(
            [s["synth_audio_len_samples"] for s in samples], dtype=torch.long
        )

        return batch

    def _collater_whisper(self, audios):
        return torch.cat(audios, dim=0)

    def _collater_video(self, audios, audio_size):
        feat_shape = list(audios[0].shape[1:])
        out = audios[0].new_zeros([len(audios), audio_size] + feat_shape)
        padding_mask = torch.BoolTensor(len(audios), audio_size).fill_(False)
        audio_starts = [0] * len(audios)
        for i, a in enumerate(audios):
            diff = len(a) - audio_size
            if diff == 0:
                out[i] = a
            elif diff < 0:
                out[i] = torch.cat([a, a.new_full([-diff] + feat_shape, 0.0)])
                padding_mask[i, diff:] = True
            else:
                start = np.random.randint(0, diff + 1)
                out[i] = a[start : start + audio_size]
                audio_starts[i] = start
        return out, padding_mask, audio_starts
