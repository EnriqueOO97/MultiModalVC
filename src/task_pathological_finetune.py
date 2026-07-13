# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Task for the pathological -> healthy fine-tune stage.

Reads ``trainAugmented.tsv`` / ``validAugmented.tsv`` (and optionally
``testAugmented.tsv``) from ``task.data`` in the 9-column pathological
fine-tune format. Wraps ``mms_pathological_finetune_dataset``.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from fairseq.data import Dictionary
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask

DBG = True if len(sys.argv) == 1 else False

if DBG:
    from src.dataset_pathological_finetune import mms_pathological_finetune_dataset
    from src.task_synthvc import MMS_LLaMA_TrainingSynthVCConfig
else:
    from .dataset_pathological_finetune import mms_pathological_finetune_dataset
    from .task_synthvc import MMS_LLaMA_TrainingSynthVCConfig

logger = logging.getLogger(__name__)


@dataclass
class MMS_PathologicalFinetuneConfig(MMS_LLaMA_TrainingSynthVCConfig):
    """Inherits all task fields from the synth-VC training config so the existing
    YAML (mms-speech-nollm-e2e-synthvc.yaml) loads without errors.  Fields like
    label_dir / labels / tokenizer / noise_* are accepted but ignored — the
    pathological dataset doesn't use them.

    Adds per-module freeze knobs that the task applies to the model right after
    build_model. AV-HuBERT, Whisper (except top-N), SR predictor and mel_head
    are always frozen and not exposed as knobs.
    """

    afeat_1d_conv_trainable: bool = field(
        default=True,
        metadata={"help": "Trainable: afeat_1d_conv + vfeat_1d_conv (audio/video temporal aligners)"},
    )
    fusion_trainable: bool = field(
        default=True,
        metadata={"help": "Trainable: multimodal_attention_layer + audio_mask_emb + video_mask_emb"},
    )
    qformer_trainable: bool = field(
        default=True,
        metadata={"help": "Trainable: Qformer + query_tokens + avfeat_to_llm"},
    )
    proj_trainable: bool = field(
        default=True,
        metadata={"help": "Trainable: proj1/proj2 and their layer norms ln1/ln2"},
    )
    conformer_trainable: bool = field(
        default=True,
        metadata={"help": "Trainable: conformer + ln3 (post-conformer layer norm)"},
    )
    vocoder_trainable: bool = field(
        default=False,
        metadata={"help": "Trainable: HiFi-GAN generator (vocoder_* + _full_vocoder.*)"},
    )
    whisper_top_n_trainable: int = field(
        default=0,
        metadata={"help": "Unfreeze the last N transformer layers of the Whisper encoder (0 = fully frozen)"},
    )
    whisper_layernorm_trainable: bool = field(
        default=False,
        metadata={"help": "Unfreeze the final layer_norm of the Whisper encoder (only meaningful when whisper_top_n_trainable > 0)"},
    )
    avhubert_top_n_trainable: int = field(
        default=0,
        metadata={"help": "Unfreeze the last N transformer layers of the AV-HuBERT encoder (0 = fully frozen)"},
    )
    avhubert_layernorm_trainable: bool = field(
        default=False,
        metadata={"help": "Unfreeze the AV-HuBERT encoder's final layer_norm (only meaningful when avhubert_top_n_trainable > 0)"},
    )
    number_of_synths: int = field(
        default=6,
        metadata={"help": "How many synth targets (1..N) to pair with each input, in addition to the always-included real healthy target. 0 = healthy only. Applies to train and valid."},
    )
    whisper_pretrained_path: str = field(
        default="",
        metadata={"help": "Path to an externally-finetuned Whisper checkpoint (HF dir containing model.safetensors, or a direct .safetensors file). Its encoder weights are swapped in at build time. Empty = stock openai/whisper-medium."},
    )
    disc_init_checkpoint: str = field(
        default="",
        metadata={"help": "Optional fairseq checkpoint to warm-start ONLY the discriminator(s) (mpd./msstftd./cqtd.* keys). The generator side is ignored. Used to rescue a clean (bug-independent) discriminator when the rest of the model is cold-started. Empty = disc from scratch."},
    )
    ogm_enabled: bool = field(
        default=False,
        metadata={"help": "On-the-fly gradient modulation (Peng et al. 2022): after backward, throttle whichever modality branch (audio=whisper/afeat, video=avhubert/vfeat) has the larger gradient, pushing the two toward balance so video isn't ignored. Skips modality-dropout steps automatically (one branch has ~0 grad). With ogm_alpha=0 it only measures+logs (no modulation)."},
    )
    ogm_alpha: float = field(
        default=0.3,
        metadata={"help": "OGM strength. Throttle coefficient k = 1 - tanh(alpha * rho), rho = max(ga/gv, gv/ga). Higher = harder throttle on the dominant branch; 0 = measure-only. Pick from a measured rho via alpha = atanh(1-k_target)/rho; ~0.05-0.1 is a gentle start."},
    )


@register_task("MMS_LLaMA_pathological_finetune", dataclass=MMS_PathologicalFinetuneConfig)
class MMS_PathologicalFinetuneTask(FairseqTask):
    cfg: MMS_PathologicalFinetuneConfig

    def __init__(self, cfg: MMS_PathologicalFinetuneConfig) -> None:
        super().__init__(cfg)
        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"MMS_PathologicalFinetuneTask Config {cfg}")
        self.fine_tuning = cfg.fine_tuning
        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def dictionaries(self) -> Optional[List[Dictionary]]:
        return None

    # ---- OGM: on-the-fly gradient modulation (Peng et al., CVPR 2022) ----------
    # Audio branch = whisper(+afeat) ; video branch = avhubert(+vfeat). Only the
    # TRAINABLE params of each have a grad (frozen layers -> grad None -> ignored).
    _OGM_AUDIO = ("whisper.", "afeat_1d_conv.")
    _OGM_VIDEO = ("avhubert.", "vfeat_1d_conv.")

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        # super() runs forward + backward; grads are populated on return.
        out = super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        # Call OGM on EVERY rank every step (drop the `not ignore_grad` guard): its
        # internal all-reduce must be rank-symmetric, and ignore_grad is set per-rank
        # on dummy-batch steps. Skipping it on dummy steps desynced the collective
        # count across ranks -> NCCL all-reduce timeout / crash. On dummy steps grads
        # are ~0, so the eps guard inside skips the actual modulation anyway.
        if getattr(self.cfg, "ogm_enabled", False):
            self._ogm_modulate(model, update_num)
        return out

    @staticmethod
    def _ogm_base_name(n):
        # The training model is wrapped (DDP + an inner wrapper), so named_parameters()
        # are prefixed with "module.module." — strip ALL leading "module." levels so the
        # branch prefixes ("avhubert.", "whisper.", ...) match.
        while n.startswith("module."):
            n = n[7:]
        return n

    def _ogm_modulate(self, model, update_num) -> None:
        import torch
        import torch.distributed as dist

        dev = next(model.parameters()).device

        def branch_sqnorm(prefixes):
            # Device scalar, 0.0 if this branch has no local grad this step -- NEVER
            # None. The all-reduce below MUST be reached identically on every rank;
            # if a rank whose modality-dropout masked a branch returned early while
            # another rank called the collective, NCCL would deadlock (this is the
            # bug that hung the run once avhubert started receiving gradients).
            s = torch.zeros((), device=dev)
            for n, p in model.named_parameters():
                if p.grad is not None and self._ogm_base_name(n).startswith(prefixes):
                    s = s + p.grad.detach().pow(2).sum()
            return s

        ga2 = branch_sqnorm(self._OGM_AUDIO)
        gv2 = branch_sqnorm(self._OGM_VIDEO)
        # UNCONDITIONAL, rank-symmetric collective: every rank reaches this exactly
        # once per step regardless of its own modality dropout. Summed norms make
        # rho/k global and identical everywhere; scaling local grads by that global
        # scalar commutes with the later DDP all-reduce, so workers stay consistent.
        if dist.is_available() and dist.is_initialized():
            packed = torch.stack([ga2, gv2])
            dist.all_reduce(packed)
            ga2, gv2 = packed[0], packed[1]
        ga = ga2.sqrt()
        gv = gv2.sqrt()
        eps = 1e-8
        # ga/gv are GLOBAL now -> this branch is identical on all ranks, so every
        # rank skips or continues together (no collective past this point).
        if ga < eps or gv < eps:
            return
        alpha = float(self.cfg.ogm_alpha)
        if ga >= gv:
            rho, prefixes, dom = ga / gv, self._OGM_AUDIO, "audio"
        else:
            rho, prefixes, dom = gv / ga, self._OGM_VIDEO, "video"
        k = float(1.0 - torch.tanh(alpha * rho))
        if alpha > 0.0 and k < 1.0:           # alpha==0 => measure-only (k==1, no-op)
            for n, p in model.named_parameters():
                if p.grad is not None and self._ogm_base_name(n).startswith(prefixes):
                    p.grad.mul_(k)
        rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        if rank0 and update_num % 50 == 0:
            logger.info(
                f"[OGM] upd={update_num} ga={float(ga):.4f} gv={float(gv):.4f} "
                f"rho={float(rho):.3f} dominant={dom} k={k:.4f}"
            )

    @classmethod
    def setup_task(cls, cfg: MMS_PathologicalFinetuneConfig, **kwargs):
        if cfg.pdb:
            import pdb
            pdb.set_trace()
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}PATH-HE.tsv"
        logger.info(f"[pathological-finetune] loading manifest '{split}' from {manifest}")

        image_aug = self.cfg.image_aug if split == "train" else False

        self.datasets[split] = mms_pathological_finetune_dataset(
            manifest_path=manifest,
            sample_rate=self.cfg.sample_rate,
            max_sample_size=self.cfg.max_sample_size,
            shuffle=(split == "train"),
            normalize=self.cfg.normalize,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            subset_name=split,
            number_of_synths=self.cfg.number_of_synths,
            noise_wav=self.cfg.noise_wav,
            noise_prob=self.cfg.noise_prob,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices

    def build_model(self, cfg):
        model = super().build_model(cfg)
        # Swap in externally-finetuned Whisper encoder BEFORE the freeze plan so
        # top-N selection operates on the adapted encoder. Runs before fairseq's
        # finetune_from_model restore; if the restored checkpoint contains
        # whisper.* keys (trained runs), those override this swap — which is what
        # we want (saved weights win, external file is only the cold-start source).
        self._load_finetuned_whisper(model)
        self._load_disc_init(model)
        self._apply_finetune_freeze_plan(model)
        # Tell the criterion that this task owns the freeze plan — its legacy
        # disc-activation auto-freeze must not run for this finetune.
        model._finetune_owns_freeze = True
        # Force the model into the single-pass forward branch. Without this,
        # the model's gate routes "disc off + training mode" into the Phase-1
        # dual-pass (canonical + synthetic), which is not what this finetune is.
        model._force_single_pass = True
        # Disable modality dropout: pathological audio IS the signal we need to
        # learn from. Masking it 25% of the time forces the model to hallucinate
        # healthy speech from video alone — wrong objective for this task.
        if hasattr(model, "cfg"):
            model.cfg.p_modality_av = 1.0
            model.cfg.p_modality_video_only = 0.0
            model.cfg.p_modality_audio_only = 0.0
        logger.info(
            "[finetune-freeze] forced single-pass forward; modality dropout disabled (p_av=1.0)"
        )
        # Decide what to EXCLUDE from the saved checkpoint. Exclude ONLY the modules
        # that are deterministically REBUILT at load time:
        #   * avhubert      -> rebuilt from w2v_path in build_model
        #   * sr_predictor  -> reloaded from its fixed pretrained file in __init__
        # Everything else is persisted — including a frozen backbone in a head-only
        # salvage (its weights live nowhere else) and the finetuned whisper. This is
        # what makes EVERY checkpoint self-contained: the earlier "exclude all frozen"
        # rule stripped the frozen backbone in salvage runs, so those checkpoints were
        # missing conformer/Qformer/proj and could not be used for inference alone.
        # named_parameters() excludes buffers, so BatchNorm running stats (incl. the
        # drifted avhubert stats the model adapted to) are still saved.
        # Exclude FROZEN avhubert params (rebuilt from w2v_path at load) and the
        # always-frozen sr_predictor. TRAINABLE avhubert params (the unfrozen top-N
        # layers + final layer_norm) are NOT excluded — they must persist, exactly
        # like the trainable top-N whisper layers do, or the adaptation is lost on
        # reload (build_model rebuilds avhubert from w2v_path, overwriting them).
        model.freeze_params = [
            n for n, p in model.named_parameters()
            if (n.startswith("avhubert.") and not p.requires_grad)
            or n.startswith("sr_predictor.")
        ]
        logger.info(
            f"[finetune-freeze] save-excludes {len(model.freeze_params)} param keys "
            f"(FROZEN avhubert + sr_predictor, rebuilt at load); trainable avhubert "
            f"layers + everything else persisted -> self-contained checkpoint"
        )
        return model

    def _load_finetuned_whisper(self, model) -> None:
        """Swap the Whisper encoder weights with an externally-finetuned checkpoint.

        Reads ``model.encoder.*`` tensors from the safetensors file, strips the
        prefix, and loads them into ``model.whisper.whisper`` (the HF
        WhisperEncoder). No-op when whisper_pretrained_path is empty.
        """
        path = self.cfg.whisper_pretrained_path
        if not path:
            # Hard-fail rather than silently using stock openai/whisper-medium: this
            # run REQUIRES the externally-finetuned Whisper encoder, and a silent HF
            # fallback would train against the wrong weights without warning.
            raise RuntimeError(
                "[finetune-whisper] whisper_pretrained_path is empty. This run requires "
                "the finetuned Whisper encoder; refusing to fall back to stock "
                "openai/whisper-medium. Set task.whisper_pretrained_path explicitly."
            )

        from safetensors.torch import load_file

        st_path = path if path.endswith(".safetensors") else os.path.join(path, "model.safetensors")
        if not os.path.isfile(st_path):
            raise FileNotFoundError(f"[finetune-whisper] model.safetensors not found at {st_path}")

        full = load_file(st_path)
        prefix = "model.encoder."
        enc_state = {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
        if not enc_state:
            raise RuntimeError(f"[finetune-whisper] no 'model.encoder.*' tensors in {st_path}")

        target = model.whisper.whisper  # HF WhisperEncoder
        missing, unexpected = target.load_state_dict(enc_state, strict=False)
        target.to(next(model.parameters()).dtype)

        logger.info(
            f"[finetune-whisper] loaded {len(enc_state)} encoder tensors from {st_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing:
            logger.warning(f"[finetune-whisper] keys left at stock (missing in file): {list(missing)[:8]}")
        if unexpected:
            logger.warning(f"[finetune-whisper] ignored unexpected keys: {list(unexpected)[:8]}")

    def _load_disc_init(self, model) -> None:
        """Warm-start ONLY the discriminator(s) from a fairseq checkpoint.

        The discriminators (mpd./msstftd./cqtd.*) only ever consume waveforms /
        spectra — they are independent of the generator and of modality_fuse — so
        they can be rescued from a checkpoint whose generator we otherwise discard.
        Loads with strict=False (every non-disc key is reported missing and ignored);
        shape mismatches still raise, which is the desired guard if the disc arch
        differs.
        """
        import torch
        from torch import nn

        path = self.cfg.disc_init_checkpoint
        if not path:
            logger.info("[disc-init] disc_init_checkpoint empty — discriminator starts from scratch")
            return
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[disc-init] disc_init_checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        disc_prefixes = ("mpd.", "msstftd.", "cqtd.")
        disc_state = {k: v for k, v in sd.items() if k.startswith(disc_prefixes)}
        if not disc_state:
            logger.warning(f"[disc-init] no discriminator tensors (mpd./msstftd./cqtd.*) found in {path}")
            return

        # Use nn.Module.load_state_dict directly: fairseq's override prunes/returns None.
        missing, unexpected = nn.Module.load_state_dict(model, disc_state, strict=False)
        # The model has disc params that weren't in the checkpoint? Surface them.
        model_disc_keys = {n for n, _ in model.named_parameters() if n.startswith(disc_prefixes)}
        loaded_keys = set(disc_state.keys())
        disc_not_loaded = sorted(model_disc_keys - loaded_keys)
        by_prefix = {p: sum(1 for k in disc_state if k.startswith(p)) for p in disc_prefixes}
        logger.info(
            f"[disc-init] loaded {len(disc_state)} discriminator tensors from {path} "
            f"({ {p: c for p, c in by_prefix.items() if c} })"
        )
        if disc_not_loaded:
            logger.warning(f"[disc-init] model disc params with no match in checkpoint (left at init): {disc_not_loaded[:8]}")

    def _apply_finetune_freeze_plan(self, model) -> None:
        """Apply per-module trainable flags to ``model`` based on this task's config.

        Groups are disjoint prefix sets. Each named parameter is matched against
        exactly one group; parameters that don't match any group fall into the
        unconditionally-frozen set.
        """

        # Group → (trainable flag, list of parameter-name prefixes / exact names).
        # mel_head is dead code in the E2E pipeline (no gradient path) — frozen
        # always for cleanliness, no knob exposed.
        groups = {
            "afeat_1d_conv": (
                self.cfg.afeat_1d_conv_trainable,
                ("afeat_1d_conv.", "vfeat_1d_conv."),
            ),
            "fusion": (
                self.cfg.fusion_trainable,
                ("multimodal_attention_layer.", "audio_mask_emb", "video_mask_emb"),
            ),
            "qformer": (
                self.cfg.qformer_trainable,
                ("Qformer.", "query_tokens", "avfeat_to_llm."),
            ),
            "proj": (
                self.cfg.proj_trainable,
                ("proj1.", "proj2.", "ln1.", "ln2."),
            ),
            "conformer": (
                self.cfg.conformer_trainable,
                ("conformer.", "ln3."),
            ),
            "vocoder": (
                self.cfg.vocoder_trainable,
                ("vocoder_", "_full_vocoder."),
            ),
            # mel_head is the LIVE output head in MelVC (512 -> mel bands). Trainable
            # by default; the knob lets the head-only salvage be explicit. (In the
            # E2E/SynthVC pipelines this head is dead code, but leaving it trainable
            # there is harmless — those forwards never touch it, so it gets no grad.)
            "mel_head": (
                getattr(self.cfg, "mel_head_trainable", True),
                ("mel_head.",),
            ),
        }

        always_frozen_prefixes = (
            "avhubert.",
            "whisper.",          # re-opened selectively below if whisper_top_n > 0
            "sr_predictor.",
            "mpd.", "msstftd.", "cqtd.", "mel_disc.",  # disc params managed by criterion
        )

        def match_group(name: str) -> Optional[str]:
            for gname, (_flag, prefixes) in groups.items():
                if any(name.startswith(p) or name == p for p in prefixes):
                    return gname
            return None

        unmatched: List[str] = []
        group_counts = {g: {"params": 0, "trainable": 0} for g in groups}
        always_frozen_count = 0

        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in always_frozen_prefixes):
                param.requires_grad = False
                always_frozen_count += param.numel()
                continue

            g = match_group(name)
            if g is None:
                # No group claims it — leave whatever upstream init decided,
                # but record it so we surface the diagnostic in the log.
                unmatched.append(name)
                continue

            flag = groups[g][0]
            param.requires_grad = bool(flag)
            group_counts[g]["params"] += param.numel()
            if flag:
                group_counts[g]["trainable"] += param.numel()

        # Whisper top-N: unfreeze the last N transformer layers (HF WhisperEncoder.layers).
        if self.cfg.whisper_top_n_trainable > 0:
            whisper_layers = getattr(getattr(model, "whisper", None), "whisper", None)
            whisper_layers = getattr(whisper_layers, "layers", None)
            if whisper_layers is None:
                logger.warning(
                    "[finetune-freeze] whisper_top_n_trainable=%d requested but model.whisper.whisper.layers not found — skipping",
                    self.cfg.whisper_top_n_trainable,
                )
            else:
                total = len(whisper_layers)
                n = min(self.cfg.whisper_top_n_trainable, total)
                opened = 0
                for layer in whisper_layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                        opened += p.numel()
                logger.info(
                    "[finetune-freeze] whisper top %d/%d encoder layers trainable (%s params)",
                    n, total, f"{opened:,}",
                )

        if self.cfg.whisper_layernorm_trainable:
            whisper_ln = getattr(getattr(model, "whisper", None), "whisper", None)
            whisper_ln = getattr(whisper_ln, "layer_norm", None)
            if whisper_ln is None:
                logger.warning("[finetune-freeze] whisper_layernorm_trainable=True but model.whisper.whisper.layer_norm not found — skipping")
            else:
                for p in whisper_ln.parameters():
                    p.requires_grad = True
                logger.info("[finetune-freeze] whisper final layer_norm trainable")

        # AV-HuBERT top-N: unfreeze the last N transformer layers of the encoder
        # (fairseq wav2vec2 TransformerEncoder.layers). Mirrors the whisper block;
        # the video frontend (ResNet) stays frozen. Param prefix:
        # avhubert.w2v_model.encoder.layers.{i}.*
        if self.cfg.avhubert_top_n_trainable > 0:
            av_layers = getattr(getattr(model, "avhubert", None), "w2v_model", None)
            av_layers = getattr(getattr(av_layers, "encoder", None), "layers", None)
            if av_layers is None:
                logger.warning(
                    "[finetune-freeze] avhubert_top_n_trainable=%d requested but "
                    "model.avhubert.w2v_model.encoder.layers not found — skipping",
                    self.cfg.avhubert_top_n_trainable,
                )
            else:
                total = len(av_layers)
                n = min(self.cfg.avhubert_top_n_trainable, total)
                opened = 0
                for layer in av_layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                        opened += p.numel()
                logger.info(
                    "[finetune-freeze] avhubert top %d/%d encoder layers trainable (%s params)",
                    n, total, f"{opened:,}",
                )

        if self.cfg.avhubert_layernorm_trainable:
            av_ln = getattr(getattr(model, "avhubert", None), "w2v_model", None)
            av_ln = getattr(getattr(av_ln, "encoder", None), "layer_norm", None)
            if av_ln is None:
                logger.warning("[finetune-freeze] avhubert_layernorm_trainable=True but model.avhubert.w2v_model.encoder.layer_norm not found — skipping")
            else:
                for p in av_ln.parameters():
                    p.requires_grad = True
                logger.info("[finetune-freeze] avhubert encoder final layer_norm trainable")

        # Put fully-frozen groups into eval() mode so dropout/BN-like state is inert.
        eval_targets = {
            "conformer": ("conformer",),
            "vocoder": ("_full_vocoder",),
            "qformer": ("Qformer",),
            "fusion": ("multimodal_attention_layer",),
        }
        for gname, attr_names in eval_targets.items():
            if not groups[gname][0]:
                for attr in attr_names:
                    submod = getattr(model, attr, None)
                    if submod is not None and hasattr(submod, "eval"):
                        submod.eval()

        # Logging summary.
        for gname, stats in group_counts.items():
            flag = groups[gname][0]
            logger.info(
                "[finetune-freeze] group=%-13s trainable=%s  params=%s  (matched=%s)",
                gname, str(flag).lower(), f"{stats['trainable']:,}", f"{stats['params']:,}",
            )
        if unmatched:
            logger.warning(
                "[finetune-freeze] %d parameter(s) not matched by any group (left untouched). First few: %s",
                len(unmatched), unmatched[:6],
            )

        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "[finetune-freeze] FINAL trainable=%s / total=%s  (always-frozen base=%s)",
            f"{total_trainable:,}", f"{total:,}", f"{always_frozen_count:,}",
        )
