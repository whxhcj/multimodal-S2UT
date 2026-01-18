import json
import logging
import math
from argparse import Namespace
from pathlib import Path
from typing import List
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.data.audio.data_cfg import MultitaskConfig, S2SDataConfig
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    TextTargetMultitaskData,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import DummyMultiTask, SpeechToTextTask
from fairseq.tasks.text_to_speech import batch_mel_cepstral_distortion
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask

from mm_s2ut.data.speech_to_text_dataset import MultiModalSpeechToTextDatasetCreator

logger = logging.getLogger(__name__)


@register_task("multimodal_speech_to_text")
class MultiModalSpeechToTextTask(SpeechToTextTask):
    @staticmethod
    def add_args(parser):
        SpeechToTextTask.add_args(parser)
        parser.add_argument(
            "--multimodal-translation-config-yaml",
            type=str,
            default=None,
            help="Configuration YAML filename for multimodal translation (under manifest root)",
        )

    def __init__(self, args, tgt_dict):
        super(). __init__(args, tgt_dict)
        self.multimodal_translation_config = None
        if getattr(args, "multimodal_translation_config_yaml", None) is not None:
            self.multimodal_translation_config = OmegaConf.load(Path(args.multimodal_translation_config_yaml))
            logger.info(f"load multimodal_translation_config_yaml from {args.multimodal_translation_config_yaml}")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        image_feat_path = None
        if self.multimodal_translation_config:
            image_feat_path = self.multimodal_translation_config.image_feat_path
        self.datasets[split] = MultiModalSpeechToTextDatasetCreator.from_tsv(
            root=self.args.data,
            cfg=self.data_cfg,
            splits=split,
            tgt_dict=self.target_dictionary,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            is_train_split=split.startswith("train"),
            epoch=epoch,
            seed=self.args.seed,
            multitask=self.multitask_tasks,
            image_feat_path=image_feat_path, 
        )
