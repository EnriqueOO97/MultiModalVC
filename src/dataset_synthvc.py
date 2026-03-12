# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset for SynthVC-inspired training with augmented manifest.

Handles the augmented manifest format:
  audio_id \t video \t canon_audio \t spk_emb \t synth1..synth6 \t frame_count \t size \t speech_rate

Extends mms_llama_dataset to additionally load:
- Speaker embedding (x-vector .pt file)
- Randomly selected 1-of-6 synthetic audio (Whisper features + raw waveform)
"""

import itertools
import logging
import os
import sys
import random
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data.fairseq_dataset import FairseqDataset
from scipy.io import wavfile
from transformers import WhisperProcessor
import math
import torchaudio
import soundfile as sf

DBG = True if len(sys.argv) == 1 else False

from . import utils as custom_utils

logger = logging.getLogger(__name__)


def load_audio_visual_synthvc(manifest_path, max_keep, min_keep, frame_rate, label_paths, label_rates, tol=0.1):
    """Parse augmented manifest with speaker embeddings and 6 synthetic paths.

    Manifest columns (tab-separated):
        items[0]:  audio_id
        items[1]:  video_path
        items[2]:  canon_audio_path
        items[3]:  spk_embedding_path
        items[4..9]: 6 synthetic audio paths
        items[10]: video frame count
        items[11]: audio sample count (size)
        items[12]: speech_rate
    """
    def is_audio_label_aligned(audio_dur, label_durs):
        return all([abs(audio_dur - label_dur) < tol for label_dur in label_durs])

    n_long, n_short, n_unaligned = 0, 0, 0
    names, inds, sizes, speech_rates = [], [], [], []
    spk_emb_paths, synth_paths_list = [], []
    dur_from_label_list = []
    is_seq_label = any([x == -1 for x in label_rates])

    for label_path, label_rate in zip(label_paths, label_rates):
        label_lengths = [len(line.rstrip().split()) / label_rate for line in open(label_path).readlines()]
        dur_from_label_list.append(label_lengths)
    dur_from_label_list = list(zip(*dur_from_label_list))

    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[-3])  # video sample count (second to last)
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            elif (not is_seq_label) and (not is_audio_label_aligned(sz / frame_rate, dur_from_label_list[ind])):
                n_unaligned += 1
            else:
                audio_id = items[0]
                video_path = items[1]
                audio_path = items[2]
                spk_emb_path = items[3]
                synth_audio_paths = items[4:10]  # 6 synthetic paths
                speech_rate = items[-1]

                names.append((video_path, audio_path + ':' + audio_id))
                inds.append(ind)
                sizes.append(sz)
                speech_rates.append(speech_rate)
                spk_emb_paths.append(spk_emb_path)
                synth_paths_list.append(synth_audio_paths)

    tot = ind + 1
    logger.info(
        f"max_keep={max_keep}, min_keep={min_keep}, "
        f"loaded {len(names)}, skipped {n_short} short and {n_long} long and {n_unaligned} unaligned, "
        f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
    )
    return root, names, inds, tot, sizes, speech_rates, spk_emb_paths, synth_paths_list


class mms_synthvc_dataset(FairseqDataset):
    """Dataset for SynthVC-inspired training.

    Identical to mms_llama_dataset except:
    - Loads speaker embeddings per sample
    - Randomly selects 1-of-6 synthetic audio per __getitem__ call
    - Returns synthetic Whisper features and waveform in the batch
    """

    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            label_paths: List[str],
            label_rates: Union[List[float], float],
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int = 1,
            skip_verify: bool = False,
            image_mean: float = 0,
            image_std: float = 1,
            image_crop_size: int = 88,
            image_aug: bool = False,
            modalities: Optional[List[str]] = None,
            is_s2s=False,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1,
            snr_target=None
    ):
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        self.modalities = set(modalities)

        # Use augmented manifest loader
        (self.audio_root, self.names, inds, tot, self.sizes,
         self.speech_rates, self.spk_emb_paths, self.synth_paths_list
        ) = load_audio_visual_synthvc(
            manifest_path, max_keep_sample_size, min_keep_sample_size,
            frame_rate=sample_rate, label_paths=label_paths, label_rates=self.label_rates
        )

        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.num_labels = len(label_paths)
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s

        self.subset = manifest_path.split('/')[-1].split('.')[0]
        self.snr_target = snr_target
        self.snr_levels = [-5, 0, 5, 10, 15, 20]

        # Noise setup
        noise_audio, noise_sr = sf.read(noise_fn, dtype='float32')
        if noise_audio.ndim == 1:
            noise_audio = noise_audio[np.newaxis, :]
        else:
            noise_audio = noise_audio.T
        self.noise = torch.from_numpy(noise_audio)
        self.noise_prob = noise_prob

        assert noise_sr == 16000
        assert self.single_target == (self.label_rates[0] == -1)

        if store_labels:
            from .dataset import load_label
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            from .dataset import load_label_offset
            self.label_paths = label_paths
            self.label_offsets_list = [load_label_offset(p, inds, tot) for p in label_paths]

        if not skip_verify:
            from .dataset import verify_label_lengths
            for label_path, label_rate in zip(label_paths, self.label_rates):
                verify_label_lengths(self.sizes, self.sample_rate, label_path, label_rate, inds, tot)
        else:
            logger.info("Skip label alignment verifying")

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize

        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize(0.0, 255.0),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std)])
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize(0.0, 255.0),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std)])
        logger.info(f"image transform: {self.transform}")
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")

    def add_noise(self, speech):
        speech = torch.from_numpy(speech)
        speech = speech.unsqueeze(1)
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx: start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.squeeze(1).numpy()

    def load_video(self, audio_name):
        feats = custom_utils.load_video(os.path.join(self.audio_root, audio_name))
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def _load_wav(self, audio_path):
        """Load and normalize a wav file, return (wav_data_float32, sample_rate)."""
        sample_rate, wav_data = wavfile.read(audio_path)
        assert sample_rate == 16_000 and len(wav_data.shape) == 1
        if wav_data.dtype == np.int16:
            wav_data = wav_data / 32768.0
        wav_data = wav_data.astype(np.float32)
        return wav_data, sample_rate

    def _wav_to_whisper(self, wav_data, sample_rate):
        """Convert raw waveform to Whisper input features."""
        return self.whisper_processor(wav_data, sampling_rate=sample_rate, return_tensors="pt").input_features

    def __getitem__(self, index):
        # === Canonical video ===
        video_fn, audio_fn = self.names[index]
        if 'video' in self.modalities:
            video_feats = self.load_video(video_fn)
        else:
            video_feats = None

        # === Canonical audio ===
        canon_wav_data = None
        canon_audio_feats = None
        canon_audio_len_samples = None
        if 'audio' in self.modalities:
            audio_path = audio_fn.split(':')[0]
            canon_wav_data, sr = self._load_wav(audio_path)
            if self.subset == 'train' and np.random.rand() < self.noise_prob:
                canon_wav_data = self.add_noise(canon_wav_data)
            elif self.subset == 'test' and self.snr_target is not None:
                if self.noise_prob != 0:
                    canon_wav_data = self.add_noise(canon_wav_data)
            canon_audio_len_samples = int(len(canon_wav_data))
            canon_audio_feats = self._wav_to_whisper(canon_wav_data, sr)

        # === Speaker embedding ===
        spk_emb_path = self.spk_emb_paths[index]
        spk_embedding = torch.load(spk_emb_path, map_location="cpu", weights_only=True)
        if spk_embedding.dim() > 1:
            spk_embedding = spk_embedding.squeeze()  # ensure (512,)

        # === Synthetic audio (randomly pick 1 of 6) ===
        synth_paths = self.synth_paths_list[index]
        synth_idx = random.randint(0, len(synth_paths) - 1)
        synth_path = synth_paths[synth_idx]

        synth_wav_data = None
        synth_audio_feats = None
        synth_audio_len_samples = None
        if 'audio' in self.modalities:
            synth_wav_data, sr = self._load_wav(synth_path)
            synth_audio_len_samples = int(len(synth_wav_data))
            synth_audio_feats = self._wav_to_whisper(synth_wav_data, sr)

        # === Video features to tensor ===
        video_feats = torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None

        fid = self.names[index][1].split(':')[1]
        speech_rate = self.speech_rates[index]
        sr_label = torch.tensor(float(speech_rate), dtype=torch.float32)

        # === Target waveform (canonical, for mel loss) ===
        target_waveform = torch.from_numpy(canon_wav_data).float() if canon_wav_data is not None else None

        return {
            "id": index,
            "fid": fid,
            "video_source": video_feats,
            "audio_source": canon_audio_feats,
            "audio_len_samples": canon_audio_len_samples,
            "sr_label": sr_label,
            "target_waveform": target_waveform,
            # SynthVC-specific
            "spk_embedding": spk_embedding,           # (512,)
            "synth_audio_source": synth_audio_feats,   # (1, 80, T_whisper)
            "synth_audio_len_samples": synth_audio_len_samples,
        }

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        audio_source = [s["audio_source"] for s in samples]
        video_source = [s["video_source"] for s in samples]
        audio_len_samples = [s["audio_len_samples"] for s in samples]

        if audio_source[0] is None:
            audio_source = None
            audio_len_samples = None
        if video_source[0] is None:
            video_source = None

        video_sizes = [len(s) for s in video_source]
        video_size = min(max(video_sizes), self.max_sample_size)

        if audio_source is not None:
            collated_audios = self.collater_whisper_input(audio_source)
        else:
            collated_audios = None

        if video_source is not None:
            collated_videos, padding_mask, audio_starts = self.collater_audio(video_source, video_size)
        else:
            collated_videos = None

        sr_labels = torch.stack([s["sr_label"] for s in samples])

        source = {
            "audio": collated_audios,
            "video": collated_videos,
        }
        if audio_len_samples is not None:
            source["audio_lengths"] = torch.tensor(audio_len_samples, dtype=torch.long)

        net_input = {"source": source, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "utt_id": [s['fid'] for s in samples],
        }
        batch['sr_labels'] = sr_labels

        # === Target waveform (canonical) ===
        target_waveforms = [s.get("target_waveform") for s in samples]
        if target_waveforms[0] is not None:
            waveform_lengths = torch.tensor([w.size(0) for w in target_waveforms], dtype=torch.long)
            max_wav_len = waveform_lengths.max().item()
            collated_waveforms = target_waveforms[0].new_zeros(len(target_waveforms), max_wav_len)
            for i, w in enumerate(target_waveforms):
                collated_waveforms[i, :w.size(0)] = w
            batch['target_waveform'] = collated_waveforms
            batch['waveform_lengths'] = waveform_lengths

        # === SynthVC-specific: speaker embeddings ===
        spk_embeddings = [s["spk_embedding"] for s in samples]
        batch['spk_embeddings'] = torch.stack(spk_embeddings)  # (B, 512)

        # === SynthVC-specific: synthetic audio ===
        synth_sources = [s["synth_audio_source"] for s in samples]
        if synth_sources[0] is not None:
            batch['synth_audio'] = self.collater_whisper_input(synth_sources)  # (B, 80, T_whisper)
            batch['synth_audio_lengths'] = torch.tensor(
                [s["synth_audio_len_samples"] for s in samples], dtype=torch.long
            )

        return batch

    def collater_whisper_input(self, audios):
        return torch.cat(audios, dim=0)

    def collater_audio(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros([len(audios), audio_size] + audio_feat_shape)
        padding_mask = torch.BoolTensor(len(audios), audio_size).fill_(False)
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff] + audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2)
        else:
            collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous()
        return collated_audios, padding_mask, audio_starts



    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)[::-1]
