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
from dataclasses import dataclass
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
    pathological dataset doesn't use them."""
    pass


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
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
