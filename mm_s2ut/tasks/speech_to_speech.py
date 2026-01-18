import os
import torch
import random
import numpy as np
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
from fairseq.data.audio.speech_to_speech_dataset import SpeechToSpeechDatasetCreator
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    TextTargetMultitaskData,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import DummyMultiTask
from fairseq.tasks.text_to_speech import batch_mel_cepstral_distortion
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask

from mm_s2ut.data.speech_to_speech_dataset import MultiModalSpeechToSpeechDatasetCreator

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  
    # torch.backends.cudnn.benchmark = False  
    # torch.backends.cudnn.enabled = False


@register_task("multimodal_speech_to_speech")
class MultiModalSpeechToSpeechTask(SpeechToSpeechTask):
    @staticmethod
    def add_args(parser):
        SpeechToSpeechTask.add_args(parser)
        # multimodal translation
        parser.add_argument(
            "--multimodal-translation-config-yaml",
            type=str,
            default=None,
            help="Configuration YAML filename for multimodal translation (under manifest root)",
        )
        parser.add_argument(
            "--mhubert-ckpt-path",
            type=str,
            default=None,
            help="the checkpoint path of mhubert",
        )
        parser.add_argument(
            "--wav2vec2-model-dir",
            type=str,
            default=None,
            help="the model directory of wav2vec2",
        )
        parser.add_argument(
            "--freezing-updates",
            type=int,
            default=-1,
            help="freezing updates of mhubert",
        )
        # add noise
        parser.add_argument(
            "--noise-config-yaml",
            type=str,
            default=None,
            help="Configuration YAML filename for noise (under manifest root)",
        )

    def __init__(self, args, tgt_dict, infer_tgt_lang_id=None):
        super(). __init__(args, tgt_dict, infer_tgt_lang_id=infer_tgt_lang_id)
        set_seed(args.seed)
        self.multimodal_translation_config = None
        if getattr(args, "multimodal_translation_config_yaml", None) is not None:
            self.multimodal_translation_config = OmegaConf.load(Path(args.multimodal_translation_config_yaml))
        self.noise_config = None
        if getattr(args, "noise_config_yaml", None) is not None:
            self.noise_config = OmegaConf.load(Path(args.noise_config_yaml))
    
    def get_attr_from_config(self, config: str, attr: str, default=None):
        if not hasattr(self, config):
            return default
        if not hasattr(getattr(self, config), attr):
            return default
        return getattr(getattr(self, config), attr)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = MultiModalSpeechToSpeechDatasetCreator.from_tsv(
            root=self.args.data,
            data_cfg=self.data_cfg,
            splits=split,
            is_train_split=split.startswith("train"),
            epoch=epoch,
            seed=self.args.seed,
            target_is_code=self.args.target_is_code,
            tgt_dict=self.target_dictionary,
            n_frames_per_step=self.args.n_frames_per_step,
            multitask=self.multitask_tasks,
            noise_wav=self.get_attr_from_config("noise_config", "noise_wav", []),
            noise_prob=self.get_attr_from_config("noise_config", "noise_prob", 0.0),
            noise_snr=self.get_attr_from_config("noise_config", "noise_snr", 0.0),
            noise_num=self.get_attr_from_config("noise_config", "noise_num", 0),
            image_feat_path=self.get_attr_from_config("multimodal_translation_config", "image_feat_path"), 
            flickr30k_root=self.get_attr_from_config("multimodal_translation_config", "flickr30k_root"), 
            load_visual_extractor_type=self.get_attr_from_config("multimodal_translation_config", "load_visual_extractor_type"), 
            load_visual_extractor=self.get_attr_from_config("multimodal_translation_config", "load_visual_extractor"),
            image_input_size=self.get_attr_from_config("multimodal_translation_config", "image_input_size"), 
            image_mean=self.get_attr_from_config("multimodal_translation_config", "image_mean"), 
            image_std=self.get_attr_from_config("multimodal_translation_config", "image_std"), 
        )
