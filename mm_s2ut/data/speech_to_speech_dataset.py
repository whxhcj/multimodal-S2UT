import os
import numpy as np
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
import soundfile as sf
import timm
import transformers
from omegaconf import OmegaConf
from typing import BinaryIO, List, Optional, Tuple, Union

from fairseq.data import ConcatDataset, Dictionary
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.data_cfg import S2SDataConfig
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    TextTargetMultitaskData,
    _collate_frames,
    _is_int_or_np_int, 
)
from mm_s2ut.data.audio_utils import (
    select_noise, 
    add_noise, 
    add_noise_v2, 
    get_features_or_waveform, 
)

logger = logging.getLogger(__name__)


class ImageDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(
        self, 
        feat_path: str, 
        mask_path: str, 
        img_path_list: Optional[str] = None, 
        img_dir: Optional[str] = None, 
    ):
        self.img_feat = torch.load(feat_path)
        self.img_feat_mask = None
        self.img_path_list = img_path_list
        self.img_dir = img_dir
        if os.path.exists(mask_path):
            self.img_feat_mask = torch.load(mask_path)
        self.size = self.img_feat.shape[0]
        logger.info(f"ImageDataset load {self.size} image features")
        logger.info(f"img_path_list = {len(self.img_path_list) if self.img_path_list else None}")
        logger.info(f"img_dir = {img_dir}")

    def __getitem__(self, idx):
        img_path = None
        if self.img_path_list is not None and self.img_dir is not None:
            img_path = os.path.join(self.img_dir, self.img_path_list[idx])
        img_feat_mask = None
        if self.img_feat_mask is not None:
            img_feat_mask = self.img_feat_mask[idx]
        return img_path, self.img_feat[idx], img_feat_mask, 

    def __len__(self):
        return self.size


@dataclass
class MultiModalSpeechToSpeechDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    target_speaker: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None
    src_audio_path: Optional[str] = None
    img_path: Optional[str] = None
    img_tensor: Optional[torch.Tensor] = None
    img_list: Optional[torch.Tensor] = None
    img_mask_list: Optional[torch.Tensor] = None


class MultiModalSpeechToSpeechDataset(SpeechToTextDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2SDataConfig,
        src_audio_paths: List[str],
        src_n_frames: List[int],
        tgt_audio_paths: List[str],
        tgt_n_frames: List[int],
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        noise_wav: Optional[List[str]] = None,
        noise_prob: Optional[Union[int, float]] = 0.0,
        noise_snr: Optional[Union[int, float]] = 0.0,
        noise_num: Optional[int] = 0,
        image_dataset_list: Optional[List[ImageDataset]] = None, 
        load_visual_extractor_type: Optional[str] = None, 
        load_visual_extractor: Optional[str] = None, 
        image_input_size: Optional[List] = None, 
        image_mean: Optional[List] = None, 
        image_std: Optional[List] = None, 
    ):
        tgt_texts = tgt_audio_paths if target_is_code else None
        super().__init__(
            split=split,
            is_train_split=is_train_split,
            cfg=data_cfg,
            audio_paths=src_audio_paths,
            n_frames=src_n_frames,
            ids=ids,
            tgt_dict=tgt_dict,
            tgt_texts=tgt_texts,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            n_frames_per_step=n_frames_per_step,
        )

        self.tgt_audio_paths = tgt_audio_paths
        self.tgt_lens = [t // self.n_frames_per_step for t in tgt_n_frames]

        assert not target_is_code or tgt_dict is not None
        self.target_is_code = target_is_code

        assert len(tgt_audio_paths) == self.n_samples
        assert len(tgt_n_frames) == self.n_samples

        self.tgt_speakers = None
        if self.cfg.target_speaker_embed:
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(
                self.cfg.target_speaker_embed, split
            )
            spk_emb_dict = {s["id"]: s["speaker_embed"] for s in samples}
            self.tgt_speakers = [spk_emb_dict[id] for id in self.ids]
            assert len(self.tgt_speakers) == self.n_samples

        self.noise_wav = noise_wav
        self.noise_prob = noise_prob
        self.noise_snr = noise_snr
        self.noise_num = noise_num
        logger.info(f"num of noise_wav = {len(noise_wav)}")
        logger.info(f"noise_prob = {noise_prob}")
        logger.info(f"noise_snr = {noise_snr}")
        logger.info(f"noise_num = {noise_num}")
        
        self.image_dataset_list = image_dataset_list
        self.load_visual_extractor_type = load_visual_extractor_type
        self.transforms = None
        if self.image_dataset_list:
            if load_visual_extractor_type is None or load_visual_extractor_type == "":
                data_config = {
                    'input_size': tuple(OmegaConf.to_container(image_input_size, resolve=True)), 
                    'interpolation': 'bicubic', 
                    'mean': tuple(OmegaConf.to_container(image_mean, resolve=True)), 
                    'std': tuple(OmegaConf.to_container(image_std, resolve=True)), 
                    'crop_pct': 1.0, 
                    'crop_mode': 'squash', 
                }
                logger.info(f"transform_config = {data_config}")
                self.transforms = timm.data.create_transform(**self.transform_config, is_training=False)
                logger.info("load vit transform from default")
            elif load_visual_extractor_type == "vit_timm":
                model_dir = load_visual_extractor
                model_name = os.path.split(model_dir)[-1]
                model_path = os.path.join(model_dir, "pytorch_model.bin")
                model = timm.create_model(
                    model_name,
                    pretrained=False,
                )
                timm.models.load_checkpoint(model, model_path)
                model = model.eval()
                data_config = timm.data.resolve_model_data_config(model)
                self.transforms = timm.data.create_transform(**data_config, is_training=False)
                logger.info(f"transform_config = {data_config}")
                self.transforms = timm.data.create_transform(**data_config, is_training=False)
                logger.info("load vit transform from timm")
            elif load_visual_extractor_type == "vit_openai":
                self.image_processor = transformers.CLIPProcessor.from_pretrained(load_visual_extractor)
                logger.info("load vit processor from openai")
            elif load_visual_extractor_type == "vit_huggingface":
                self.image_processor = transformers.ViTImageProcessor.from_pretrained(load_visual_extractor)
                logger.info("load vit processor from huggingface")

        logger.info(self.__repr__())

    def pack_units(self, input: torch.Tensor) -> torch.Tensor:
        if self.n_frames_per_step <= 1:
            return input

        offset = 4
        vocab_size = (
            len(self.tgt_dict) - offset
        )  # remove offset from <bos>, <pad>, <eos>, <unk>, which is specific to fairseq dictionary

        assert input.dim() == 1
        stacked_input = (
            input[:-1].view(-1, self.n_frames_per_step) - offset
        )  # remove <eos>
        scale = [
            pow(vocab_size, self.n_frames_per_step - 1 - i)
            for i in range(self.n_frames_per_step)
        ]
        scale = torch.LongTensor(scale).squeeze(0)
        res = input.new((len(input) - 1) // self.n_frames_per_step + 1).fill_(input[-1])
        res[:-1] = (stacked_input * scale).sum(dim=1) + offset

        return res
    
    def add_noise(self, waveform):
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform.transpose())
        noise_waveform = select_noise(self.noise_wav, self.noise_num)
        noise_waveform = torch.from_numpy(noise_waveform.transpose())
        waveform = add_noise_v2(
            waveforms=waveform, 
            noise_waveform=noise_waveform, 
            snr_low=self.noise_snr[0], 
            snr_high=self.noise_snr[1], 
            noise_waveform_start=-1, 
            add_white_noise=False, 
            normalize=True, 
        )
        waveform = waveform.transpose(0, 1).cpu().numpy()
        return waveform

    def get_source_audio(self, index: Union[int, List[int]]) -> torch.Tensor:
        if _is_int_or_np_int(index):
            waveform, sample_rate = sf.read(
                self.audio_paths[index], dtype="float32", always_2d=True, frames=-1, start=0
            )
            if np.random.rand() < self.noise_prob:
                waveform = self.add_noise(waveform)
            source = get_features_or_waveform(
                waveform=waveform, sample_rate=sample_rate, 
                path=None,
                need_waveform=self.cfg.use_audio_input,
                use_sample_rate=self.cfg.use_sample_rate,
                waveform_transforms=self.waveform_transforms,
            )
        else:
            source = []
            for i in index:
                waveform, sample_rate = sf.read(
                    self.audio_paths[i], dtype="float32", always_2d=True, frames=-1, start=0
                )
                if np.random.rand() < self.noise_prob:
                    waveform = self.add_noise(waveform)
                waveform = get_features_or_waveform(
                    waveform=waveform, sample_rate=sample_rate,
                    path=None, 
                    need_waveform=self.cfg.use_audio_input,
                    use_sample_rate=self.cfg.use_sample_rate,
                    waveform_transforms=self.waveform_transforms,
                )
                source.append(waveform)
            source = np.concatenate(source)
        if self.cfg.use_audio_input:
            source = torch.from_numpy(source).float()
            if self.cfg.standardize_audio:
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
        else:
            if self.feature_transforms is not None:
                source = self.feature_transforms(source)
            source = torch.from_numpy(source).float()
        return source

    def __getitem__(self, index: int) -> MultiModalSpeechToSpeechDatasetItem:
        # source = self._get_source_audio(index)
        source = self.get_source_audio(index)

        tgt_lang_tag = None
        if self.cfg.prepend_tgt_lang_tag_as_bos:
            # prepend_tgt_lang_tag_as_bos: put tgt_lang_tag as bos of target
            tgt_lang_tag = self.get_lang_tag_idx(self.tgt_langs[index], self.tgt_dict)

        if not self.target_is_code:
            target = get_features_or_waveform(waveform=None, sample_rate=None, path=self.tgt_audio_paths[index])
            target = torch.from_numpy(target).float()
            target = self.pack_frames(target)
        else:
            target = self.tgt_dict.encode_line(
                self.tgt_audio_paths[index],
                add_if_not_exist=False,
                append_eos=True,
            ).long()
            if self.n_frames_per_step > 1:
                n_tgt_frame = target.size(0) - 1  # exclude <eos>
                keep_n_tgt_frame = n_tgt_frame - n_tgt_frame % self.n_frames_per_step
                target = torch.cat(
                    (
                        target[:keep_n_tgt_frame],
                        target.new_full((1,), self.tgt_dict.eos()),
                    ),
                    dim=0,
                )

        if self.tgt_speakers:
            tgt_spk = get_features_or_waveform(waveform=None, sample_rate=None, path=self.tgt_speakers[index])
            tgt_spk = torch.from_numpy(tgt_spk).float()
        else:
            tgt_spk = torch.FloatTensor([])
        
        img_path, img_tensor = None, None
        img_item, img_mask_item = None, None
        # if self.image_dataset_list:
        #     img_item = [i[index][0] for i in self.image_dataset_list]         # list for image data
        #     img_mask_item = [i[index][1] for i in self.image_dataset_list]    # list for image mask data
        #     logger.info(f"666 {index} {self.audio_paths[index]}")
        if self.image_dataset_list:
            _index = os.path.splitext(os.path.split(self.audio_paths[index])[1])[0]
            _index = int(_index) - 1
            img_path = self.image_dataset_list[0][_index][0]
            img = Image.open(img_path)
            if self.load_visual_extractor_type is None or self.load_visual_extractor_type in ("", "vit_timm"):
                img = img.convert("RGB")
                img_tensor = self.transforms(img)
            elif self.load_visual_extractor_type in ("vit_openai", "vit_huggingface"):
                pass
            img_item = [i[_index][1] for i in self.image_dataset_list]         # list for image data
            img_mask_item = [i[_index][2] for i in self.image_dataset_list]    # list for image mask data
            # logger.info(f"666 {_index} {self.audio_paths[index]}")
        return MultiModalSpeechToSpeechDatasetItem(
            index=index,
            source=source,
            target=target,
            target_speaker=tgt_spk,
            tgt_lang_tag=tgt_lang_tag,
            src_audio_path=self.audio_paths[index], 
            img_path=img_path, 
            img_tensor=img_tensor, 
            img_list=img_item, 
            img_mask_list=img_mask_item, 
        )

    def _collate_target(self, samples: List[MultiModalSpeechToSpeechDatasetItem]) -> torch.Tensor:
        if self.target_is_code:
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            # convert stacked units to a single id
            pack_targets = [self.pack_units(x.target) for x in samples]
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                pack_targets,
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            target_lengths = torch.tensor(
                [x.size(0) for x in pack_targets], dtype=torch.long
            )
        else:
            target = _collate_frames([x.target for x in samples], is_audio_input=False)
            bsz, _, d = target.size()
            prev_output_tokens = torch.cat(
                (target.new_full((bsz, 1, d), 0.0), target[:, :-1, :]), dim=1
            )
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            )

        return target, prev_output_tokens, target_lengths

    def collater(
        self, samples: List[MultiModalSpeechToSpeechDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = _collate_frames([x.source for x in samples], self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, prev_output_tokens, target_lengths = self._collate_target(samples)
        target = target.index_select(0, order)
        target_lengths = target_lengths.index_select(0, order)
        prev_output_tokens = prev_output_tokens.index_select(0, order)
        ntokens = sum(x.target.size(0) for x in samples)

        tgt_speakers = None
        if self.cfg.target_speaker_embed:
            tgt_speakers = _collate_frames(
                [x.target_speaker for x in samples], is_audio_input=True
            ).index_select(0, order)

        # extra code
        _order = order.cpu().numpy().tolist()
        img_path = [x.img_path for x in samples]
        img_path = [img_path[i] for i in _order]
        src_audio_path = [x.src_audio_path for x in samples]
        src_audio_path = [src_audio_path[i] for i in _order]
        img_tensor = [x.img_tensor for x in samples]
        if img_tensor[0] is not None:
            img_tensor = torch.stack(img_tensor, dim=0)
            img_tensor = img_tensor.index_select(0, order)
        if self.load_visual_extractor_type in ("vit_openai", "vit_huggingface"):
            images = [Image.open(image_path) for image_path in img_path]
            inputs = self.image_processor(images=images, return_tensors="pt")
            img_tensor = inputs["pixel_values"]
            # logging.info(f"666 vit image processor")
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
            # img_tensor
        
        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
            "tgt_speaker": tgt_speakers,  # TODO: unify "speaker" and "tgt_speaker"
            "src_audio_path": src_audio_path,
            "img_path": img_path, 
            "img_tensor": img_tensor, 
            "imgs_list": imgs_list,
            "img_masks_list": img_masks_list,
        }
        if self.tgt_texts is not None and samples[0].tgt_lang_tag is not None:
            for i in range(len(samples)):
                net_input["prev_output_tokens"][i][0] = samples[order[i]].tgt_lang_tag
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": tgt_speakers,  # to support Tacotron2 loss for speech-to-spectrogram model
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out


class MultiModalSpeechToSpeechMultitaskDataset(MultiModalSpeechToSpeechDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multitask_data = {}

    def add_multitask_dataset(self, task_name, task_data):
        self.multitask_data[task_name] = task_data

    def __getitem__(
        self, index: int
    ) -> Tuple[MultiModalSpeechToSpeechDatasetItem, Dict[str, torch.Tensor]]:
        s2s_data = super().__getitem__(index)

        multitask_target = {}
        sample_id = self.ids[index]
        tgt_lang = self.tgt_langs[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id, tgt_lang)

        return s2s_data, multitask_target

    def collater(
        self, samples: List[Tuple[MultiModalSpeechToSpeechDatasetItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}

        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        # del out["order"]

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


class MultiModalSpeechToSpeechDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    # KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_text", "tgt_n_frames"
    # optional columns
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_LANG = ""
    dic_img_dir = {
        'test2017': 'test2017', 
        'testcoco': 'testcoco',
        'test2016': 'flickr30k',
        'train': 'flickr30k',
        'val': 'flickr30k',
        'valid': 'flickr30k',
        'test.2017': 'test2017', 
        'test.coco': 'testcoco',
        'test.2016': 'flickr30k',
    }
    dic_txt = {
        'test2017': 'test_2017_flickr.txt',
        'testcoco': 'test_2017_mscoco.txt',
        'test2016': 'test_2016_flickr.txt',
        'train': 'train.txt',
        'val': 'val.txt',
        'valid': 'val.txt',
        'test.2017': 'test_2017_flickr.txt',
        'test.coco': 'test_2017_mscoco.txt',
        'test.2016': 'test_2016_flickr.txt',
    }

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        data_cfg: S2SDataConfig,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
        noise_wav: Optional[List[str]] = None,
        noise_prob: Optional[Union[int, float]] = 0.0,
        noise_snr: Optional[Union[int, float]] = 0.0,
        noise_num: Optional[int] = 0,
        image_feat_path: Optional[List] = None, 
        img_dir: Optional[str] = None, 
        flickr30k_root: Optional[str] = None, 
        load_visual_extractor_type: Optional[str] = None,
        load_visual_extractor: Optional[str] = None, 
        image_input_size: Optional[List] = None, 
        image_mean: Optional[List] = None, 
        image_std: Optional[List] = None, 
    ) -> MultiModalSpeechToSpeechDataset:
        audio_root = Path(data_cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        src_audio_paths = [
            (audio_root / s[cls.KEY_SRC_AUDIO]).as_posix() for s in samples
        ]
        tgt_audio_paths = [
            s[cls.KEY_TGT_AUDIO]
            if target_is_code
            else (audio_root / s[cls.KEY_TGT_AUDIO]).as_posix()
            for s in samples
        ]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_n_frames = [int(s[cls.KEY_TGT_N_FRAMES]) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        image_dataset_list = []
        img_path_list = None
        img_dir = None
        if flickr30k_root is not None:
            img_dir = os.path.join(flickr30k_root, cls.dic_img_dir[split_name] + "-images")
            img_path_list = []
            with open(os.path.join(flickr30k_root, cls.dic_txt[split_name]), mode="r+") as f:
                for line in f.readlines():
                    line = line.strip()
                    img_path_list.append(line)
        if image_feat_path is not None:
            for feat_path in image_feat_path:
                # get image feature path
                assert os.path.exists(feat_path) == True, 'not found image feature'
                img_dataset = ImageDataset(
                    feat_path=os.path.join(feat_path, split_name + '.pth'), 
                    mask_path=os.path.join(feat_path, split_name + '_mask.pth'), 
                    img_path_list=img_path_list, 
                    img_dir=img_dir, 
                )
                image_dataset_list.append(img_dataset)

        has_multitask = multitask is not None and len(multitask.keys()) > 0
        dataset_cls = (
            MultiModalSpeechToSpeechMultitaskDataset if has_multitask else MultiModalSpeechToSpeechDataset
        )

        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            data_cfg=data_cfg,
            src_audio_paths=src_audio_paths,
            src_n_frames=src_n_frames,
            tgt_audio_paths=tgt_audio_paths,
            tgt_n_frames=tgt_n_frames,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            target_is_code=target_is_code,
            tgt_dict=tgt_dict,
            n_frames_per_step=n_frames_per_step,
            noise_wav=noise_wav,
            noise_prob=noise_prob,
            noise_snr=noise_snr,
            noise_num=noise_num,
            image_dataset_list=image_dataset_list, 
            load_visual_extractor_type=load_visual_extractor_type,
            load_visual_extractor=load_visual_extractor,
            image_input_size=image_input_size, 
            image_mean=image_mean, 
            image_std=image_std, 
        )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds

    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2SDataConfig,
        splits: str,
        is_train_split: bool,
        epoch: int,
        seed: int,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
        noise_wav: Optional[List[str]] = None,
        noise_prob: Optional[Union[int, float]] = 0.0,
        noise_snr: Optional[Union[int, float]] = 0.0,
        noise_num: Optional[int] = 0,
        image_feat_path: Optional[List] = None, 
        flickr30k_root: Optional[str] = None, 
        load_visual_extractor_type: Optional[str] = None,
        load_visual_extractor: Optional[str] = None, 
        image_input_size: Optional[List] = None, 
        image_mean: Optional[List] = None, 
        image_std: Optional[List] = None, 
    ) -> MultiModalSpeechToSpeechDataset:
        datasets = []
        for split in splits.split(","):
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(root, split)
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                data_cfg=data_cfg,
                target_is_code=target_is_code,
                tgt_dict=tgt_dict,
                n_frames_per_step=n_frames_per_step,
                multitask=multitask,
                noise_wav=noise_wav,
                noise_prob=noise_prob,
                noise_snr=noise_snr,
                noise_num=noise_num,
                image_feat_path=image_feat_path, 
                flickr30k_root=flickr30k_root, 
                load_visual_extractor_type=load_visual_extractor_type, 
                load_visual_extractor=load_visual_extractor,
                image_input_size=image_input_size, 
                image_mean=image_mean, 
                image_std=image_std, 
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
