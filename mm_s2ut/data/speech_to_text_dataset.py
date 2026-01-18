import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from fairseq.data import ConcatDataset, Dictionary, FairseqDataset, ResamplingDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.data_cfg import S2SDataConfig, S2TDataConfig
from fairseq.data.audio.dataset_transforms.noisyoverlapaugment import (
    NoisyOverlapAugment,
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.waveform_transforms import CompositeAudioWaveformTransform
from fairseq.data.audio.dataset_transforms import CompositeAudioDatasetTransform
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextMultitaskDataset, 
    SpeechToTextDatasetCreator,
    TextTargetMultitaskData,
    _collate_frames,
)
from mm_s2ut.data.speech_to_speech_dataset import ImageDataset

logger = logging.getLogger(__name__)


@dataclass
class MultiModalSpeechToTextDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    speaker_id: Optional[int] = None
    tgt_lang_tag: Optional[int] = None
    img_list: Optional[torch.Tensor] = None
    img_mask_list: Optional[torch.Tensor] = None


class MultiModalSpeechToTextDataset(SpeechToTextDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2TDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        n_frames_per_step=1,
        speaker_to_id=None,
        append_eos=True,
        image_dataset_list: Optional[List[ImageDataset]] = None, 
    ):
        self.image_dataset_list = image_dataset_list
        super().__init__(
            split,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id,
            append_eos,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"image_dataset_list={self.image_dataset_list}, "
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"n_frames_per_step={self.n_frames_per_step}, "
            f"shuffle={self.shuffle}, "
            f"feature_transforms={self.feature_transforms}, "
            f"waveform_transforms={self.waveform_transforms}, "
            f"dataset_transforms={self.dataset_transforms})"
        )

    def __getitem__(self, index: int) -> MultiModalSpeechToTextDatasetItem:
        item = super().__getitem__(index)
        item = MultiModalSpeechToTextDatasetItem(
            item.index, 
            item.source, 
            item.target, 
            item.speaker_id, 
        )
        if self.image_dataset_list:
            img_item = [i[index][0] for i in self.image_dataset_list]         # list for image data
            img_mask_item = [i[index][1] for i in self.image_dataset_list]    # list for image mask data
            item.img_list = img_item
            item.img_mask_list = img_mask_item
        return item
    
    def collater(
        self, samples: List[MultiModalSpeechToTextDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)

        sources = [x.source for x in samples]
        has_NOAug = self.dataset_transforms.has_transform(NoisyOverlapAugment)
        if has_NOAug and self.cfg.use_audio_input:
            NOAug = self.dataset_transforms.get_transform(NoisyOverlapAugment)
            sources = NOAug(sources)

        frames = _collate_frames(sources, self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.size(0) for x in sources], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                eos_idx=None,
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(x.target.size(0) for x in samples)

        speaker = None
        if self.speaker_to_id is not None:
            speaker = (
                torch.tensor([s.speaker_id for s in samples], dtype=torch.long)
                .index_select(0, order)
                .view(-1, 1)
            )

        # extra code
        imgs_list, img_masks_list = [], []
        if self.image_dataset_list:
            imgs_list = [[] for i in range(len(samples[0].img_list))]
            for s in samples:
                for idx, img in enumerate(s.img_list):
                    imgs_list[idx].append(img)
            for idx, i in enumerate(imgs_list):
                img = torch.stack(i, dim=0)
                img = img.index_select(0, order)
                imgs_list[idx] = img
            img_masks_list = []
            img_masks_pos = []
            for idx, i in enumerate(samples[0].img_mask_list):
                if i is not None:
                    img_masks_list.append([])
                    img_masks_pos.append(idx)
                else:
                    img_masks_list.append(None)
            for s in samples:
                for idx, img_mask in enumerate(s.img_mask_list):
                    if idx in img_masks_pos:
                        img_masks_list[idx].append(img_mask)
            for idx, i in enumerate(img_masks_list):
                if idx in img_masks_pos:
                    img_mask = torch.stack(i, dim=0)
                    img_mask = img_mask.index_select(0, order)
                    img_masks_list[idx] = img_mask

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
            "imgs_list": imgs_list,
            "img_masks_list": img_masks_list,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": speaker,
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out
            

class MultiModalSpeechToTextMultitaskDataset(SpeechToTextMultitaskDataset):
    def __getitem__(
        self, index: int
    ) -> Tuple[MultiModalSpeechToTextDatasetItem, Dict[str, torch.Tensor]]:
        s2t_data = super().__getitem__(index)
        multitask_target = {}
        sample_id = self.ids[index]
        tgt_lang = self.tgt_langs[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id, tgt_lang)
        return s2t_data, multitask_target

    def collater(
        self, samples: List[Tuple[MultiModalSpeechToTextDatasetItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}
        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        del out["order"]
        for task_name, task_dataset in self.multitask_data.items():
            if "multitask" not in out:
                out["multitask"] = {}
            d = [s[task_name] for _, s in samples]
            task_target = task_dataset.collater(d)
            out["multitask"][task_name] = {
                "target": task_target["target"].index_select(0, order),
                "target_lengths": task_target["target_lengths"].index_select(0, order),
                "ntokens": task_target["ntokens"],
            }
            out["multitask"][task_name]["net_input"] = {
                "prev_output_tokens": task_target["prev_output_tokens"].index_select(
                    0, order
                ),
            }
        return out
    

class MultiModalSpeechToTextDatasetCreator(SpeechToTextDatasetCreator):
    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
        multitask: Optional[Dict] = None,
        image_feat_path: Optional[List] = None, 
    ) -> MultiModalSpeechToTextDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        image_dataset_list = []
        for feat_path in image_feat_path:
            # get image feature path
            assert os.path.exists(feat_path) == True, 'not found image feature'
            img_dataset = ImageDataset(
                feat_path=os.path.join(feat_path, split_name + '.pth'), 
                mask_path=os.path.join(feat_path, split_name + '_mask.pth'), 
            )
            image_dataset_list.append(img_dataset)

        has_multitask = multitask is not None and len(multitask.keys()) > 0
        dataset_cls = (
            MultiModalSpeechToTextMultitaskDataset if has_multitask else MultiModalSpeechToTextDataset
        )

        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            cfg=cfg,
            audio_paths=audio_paths,
            n_frames=n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id, 
            image_dataset_list=image_dataset_list, 
        )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        split: str,
        tgt_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
        multitask: Optional[Dict] = None,
        image_feat_path: Optional[List] = None, 
    ) -> MultiModalSpeechToTextDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id,
            multitask,
            image_feat_path, 
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        n_frames_per_step: int = 1,
        speaker_to_id=None,
        multitask: Optional[Dict] = None,
        image_feat_path: Optional[List] = None, 
    ) -> MultiModalSpeechToTextDataset:
        datasets = [
            cls._from_tsv(
                root=root,
                cfg=cfg,
                split=split,
                tgt_dict=tgt_dict,
                is_train_split=is_train_split,
                pre_tokenizer=pre_tokenizer,
                bpe_tokenizer=bpe_tokenizer,
                n_frames_per_step=n_frames_per_step,
                speaker_to_id=speaker_to_id,
                multitask=multitask,
                image_feat_path=image_feat_path, 
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]