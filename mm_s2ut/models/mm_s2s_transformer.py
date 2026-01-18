import os
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import OmegaConf
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
import timm
import transformers
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from fairseq import checkpoint_utils, utils
from fairseq.data.audio.data_cfg import get_config_from_yaml
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.hubert import HubertModel
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.models.speech_to_speech.s2s_transformer import (
    S2TTransformerEncoder, 
    S2UTTransformerModel, 
    s2ut_architecture_base, 
)
from fairseq.models.speech_to_speech.modules.ctc_decoder import CTCDecoder
from fairseq.models.speech_to_speech.modules.stacked_embedding import StackedEmbedding
from fairseq.models.speech_to_text import S2TTransformerEncoder, Conv1dAdaptor
from fairseq.models.text_to_speech import TTSTransformerDecoder
from fairseq.models.transformer import Linear, TransformerDecoder, TransformerModelBase
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from mm_s2ut.models.fuse import (
    SelectiveAttention,
    MultimodalAttention,
    TransformerLayerConfig,
    ExternalMultimodalTransformerEncoder,
    Wav2Vec2WithMultiModal,
    QFormerModel,
)

logger = logging.getLogger(__name__)


def unfreeze_module(model, num_updates, freezing_updates):
    if (
        freezing_updates is not None
        and num_updates > freezing_updates
    ):
        # for p in model.parameters():
        #     p.requires_grad = True
        model.requires_grad = True


def forward_padding_mask(
    features: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    extra = padding_mask.size(1) % features.size(1)
    if extra > 0:
        padding_mask = padding_mask[:, :-extra]
    padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
    padding_mask = padding_mask.all(-1)
    return padding_mask

def expand(t: torch.Tensor, repeat: int) -> torch.Tensor:
    shape = [repeat] + [-1] * len(t.shape)
    return t.unsqueeze(0).expand(*shape)


class MM_S2STransformerEncoder(S2TTransformerEncoder):
    """Based on S2T transformer encoder, with support
    to incorporate target speaker embedding and images."""

    def __init__(self, args):
        super().__init__(args)
        self.spk_emb_proj = None
        if args.target_speaker_embed:
            self.spk_emb_proj = Linear(
                args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim
            )
        # image MMT
        self.multimodal_translation_flag = False
        self.is_fusion_top = False
        self.load_visual_extractor_type = None
        self.only_img = False
        if getattr(args, "multimodal_translation_config_yaml", None) is not None:
            multimodal_translation_config = OmegaConf.load(Path(args.multimodal_translation_config_yaml))
            self.multimodal_translation_flag = True
            self.load_visual_extractor_type = multimodal_translation_config.load_visual_extractor_type
            self.only_img = multimodal_translation_config.only_img
        logger.info(f"only_img = {self.only_img}")
        logger.info(f"load_visual_extractor_type = {self.load_visual_extractor_type}")
        if self.load_visual_extractor_type is not None and self.load_visual_extractor_type != "":
            logger.info(f"path of visual extractor is {multimodal_translation_config.load_visual_extractor}")
            # self.image_feat_dim = multimodal_translation_config.image_feat_dim
            config = {
                "visual_extractor_type": self.load_visual_extractor_type, 
                "visual_extractor_path": multimodal_translation_config.load_visual_extractor, 
                "image_feat_dim": multimodal_translation_config.image_feat_dim, 
            }
            self.build_visual_extractor(**config)
        self.multimodal_attention_type = None
        if self.multimodal_translation_flag:
            # embed_dim = args.decoder_embed_dim
            embed_dim = args.encoder_embed_dim
            self.multimodal_attention_type = getattr(multimodal_translation_config, "multimodal_attention_type", None)
            logger.info(f"multimodal_attention_type = {self.multimodal_attention_type}")
            self.use_selective_gate = getattr(multimodal_translation_config, "use_selective_gate", None)
            logger.info(f"use_selective_gate = {self.use_selective_gate}")
            if self.multimodal_attention_type is None:
                pass
            elif self.multimodal_attention_type == "selective_attention":
                self.selective_attns = nn.ModuleList(
                    [
                        SelectiveAttention(
                            qdim=embed_dim, kdim=i,
                            vdim=i, attn_dim=embed_dim,
                            intermediate_dim=embed_dim, output_dim=embed_dim,
                            num_heads=1, attn_drop=multimodal_translation_config.SA_attention_dropout
                        ) \
                            for i in multimodal_translation_config.image_feat_dim
                    ]
                )
            elif self.multimodal_attention_type == "multimodal_attention":
                self.is_merge_text_img = getattr(multimodal_translation_config, "is_merge_text_img", False)
                logger.info(f"is_merge_text_img = {self.is_merge_text_img}")
                self.multimodal_attns = nn.ModuleList(
                    [
                        MultimodalAttention(
                            embed_dim=embed_dim, kdim=i, vdim=i, 
                            # num_heads=8, 
                            num_heads=1, 
                            dropout=multimodal_translation_config.SA_attention_dropout, 
                            add_bias_kv=True, 
                        ) \
                            for i in multimodal_translation_config.image_feat_dim
                    ]
                )
            elif self.multimodal_attention_type == "external_multimodal_transformer":
                self.external_multimodal_transformer_layers = multimodal_translation_config.external_multimodal_transformer_layers
                self.multimodal_transformer = nn.ModuleList(
                    [
                        ExternalMultimodalTransformerEncoder(
                            layer_config=TransformerLayerConfig(
                                embed_dim=embed_dim, kdim=i, vdim=i, 
                                nhead=i // 64, dim_feedforward=i * 4, 
                                dropout=multimodal_translation_config.SA_attention_dropout, 
                                batch_first=False, 
                            ),
                            num_layers=self.external_multimodal_transformer_layers,
                        ) \
                            for i in multimodal_translation_config.image_feat_dim
                    ]
                )
            elif self.multimodal_attention_type == "wav2vec2_multimodal":
                pass
            else:
                raise NotImplementedError()
            self.gate_denses = nn.ModuleList(
                [
                    Linear(2 * embed_dim, embed_dim) for i in multimodal_translation_config.image_feat_dim
                    # Linear(embed_dim + i, embed_dim) for i in multimodal_translation_config.image_feat_dim
                ]
            )
            self.image_dropout_module = FairseqDropout(
                multimodal_translation_config.SA_image_dropout, module_name=self.__class__.__name__
            )
            self.text_dropout_module = FairseqDropout(
                multimodal_translation_config.SA_text_dropout, module_name=self.__class__.__name__
            )
            self.image_pre_norm_module = nn.Identity()
            if multimodal_translation_config.image_pre_norm:
                self.image_pre_norm_module = nn.LayerNorm(multimodal_translation_config.image_feat_dim, 1e-5, True)
            self.is_fusion_top = multimodal_translation_config.is_fusion_top
            self.modality_dropout = multimodal_translation_config.modality_dropout
            self.audio_dropout = multimodal_translation_config.audio_dropout
            self.multimodal_extractor_type = getattr(multimodal_translation_config, "multimodal_extractor_type", None)
            if self.multimodal_extractor_type is not None and self.multimodal_extractor_type == "q_former":
                self.q_former = QFormerModel(
                    num_queries=getattr(multimodal_translation_config, "num_queries", 32),
                    num_query_layers=getattr(multimodal_translation_config, "num_query_layers", 4),
                    num_multimodal_layers=getattr(multimodal_translation_config, "num_multimodal_layers", 2),
                    self_attention_first=getattr(multimodal_translation_config, "self_attention_first", False),
                    layer_config=TransformerLayerConfig(
                        embed_dim=768, kdim=768, vdim=768, 
                        nhead=768 // 64, dim_feedforward=768 * 4, 
                        dropout=multimodal_translation_config.SA_attention_dropout, 
                        batch_first=True,
                    ),
                )
                logger.info(f"loaded QFormer")
                logger.info(f"num_query_layers={self.q_former.num_query_layers}, num_multimodal_layers={self.q_former.num_multimodal_layers}")
        self.mhubert_flag = False
        self.wav2vec2_flag = False
        self.proj_768_to_512 = Linear(768, 512)
        self.proj_1024_to_512 = Linear(1024, 512)
        self.proj_1024_to_768 = Linear(1024, 768)
        self.wav2vec2_adaptor = Conv1dAdaptor(
            in_dim=1024,
            out_dim=768,
            n_layers=3,
            kernel_size=3,
            stride=2,
            layerdrop=0.0,
            layernorm=True,
            proj=False,
        )
        self.num_updates = None
        self.freezing_updates = getattr(args, "freezing_updates", None)
        # mhubert
        if getattr(args, "mhubert_ckpt_path", None) is not None:
            [self.mhubert], cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.mhubert_ckpt_path])
            self.mhubert.feature_extractor.requires_grad = False
            # self.mhubert.requires_grad = False
            self.mhubert_flag = True
            logger.info("mhubert_flag = True")
            logger.info(f"loaded {args.mhubert_ckpt_path}")
        # wav2vec2
        elif getattr(args, "wav2vec2_model_dir", None) is not None:
            self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(
                args.wav2vec2_model_dir, 
                # gradient_checkpointing=True,
            )
            self.wav2vec2.freeze_feature_extractor()
            self.wav2vec2_flag = True
            logger.info("wav2vec2_flag = True")
            logger.info(f"loaded {args.wav2vec2_model_dir}")
            if getattr(self, "multimodal_attention_type", "") == "wav2vec2_multimodal":
                self.num_cross_attention_layers = getattr(multimodal_translation_config, "num_cross_attention_layers", 1)
                self.wav2vec2_multimodal = Wav2Vec2WithMultiModal(
                    wav2vec2=self.wav2vec2,
                    num_cross_attention_layers=self.num_cross_attention_layers,
                    wav2vec2_embed_dim=768,
                    m2_dim=768,
                    dropout=0.1,
                    batch_first=True,
                )
                logger.info(f"load Wav2Vec2WithMultiModal, num_cross_attention_layers = {self.num_cross_attention_layers}")
        assert (not self.mhubert_flag and not self.wav2vec2_flag) \
            or (self.mhubert_flag ^ self.wav2vec2_flag), \
                "only load one of speech encoders! "
        logger.info(f"multimodal_translation_flag = {self.multimodal_translation_flag}")
        logger.info(f"is_fusion_top = {self.is_fusion_top}")

    # def set_num_updates(self, num_updates):
    #     super().set_num_updates(num_updates)
    #     self.num_updates = num_updates
    #     if self.mhubert_flag:
    #         unfreeze_module(self.mhubert, self.num_updates, self.freezing_updates)
    #         self.mhubert.feature_extractor.requires_grad = False
    #     if self.wav2vec2_flag:
    #         unfreeze_module(self.wav2vec2, self.num_updates, self.freezing_updates)
    #         self.wav2vec2.feature_extractor.requires_grad = False
    
    def build_visual_extractor(self, *args, **kwargs):
        # keys = ["visual_extractor_type", "visual_extractor_path", "image_feat_dim"]
        # visual_extractor_type_choices: [None, "", "vit", "detr", "resnet", "resnet+encoder"]
        if kwargs["visual_extractor_type"] == "vit_timm":
            model_dir = kwargs["visual_extractor_path"]
            model_name = os.path.split(model_dir)[-1]
            model_path = os.path.join(model_dir, "pytorch_model.bin")
            model = timm.create_model(
                model_name,
                pretrained=False,
            )
            timm.models.load_checkpoint(model, model_path)
            self.vit = model
            logger.info("loaded vit_timm")
        elif kwargs["visual_extractor_type"] == "vit_openai":
            clip_model = transformers.CLIPModel.from_pretrained(kwargs["visual_extractor_path"])
            self.vit = clip_model.vision_model
        elif kwargs["visual_extractor_type"] == "vit_huggingface":
            self.vit = transformers.ViTForImageClassification.from_pretrained(kwargs["visual_extractor_path"])

    def forward_visual_extractor(
        self, 
        img_tensor, 
    ):
        if self.load_visual_extractor_type == "vit_timm":
            visual_feat = self.vit.forward_features(img_tensor)
            return [visual_feat]
        elif self.load_visual_extractor_type in ("vit_openai", "vit_huggingface"):
            outputs = self.vit(
                pixel_values=img_tensor, output_hidden_states=True, return_dict=True
            )
            if isinstance(outputs['hidden_states'], tuple):
                outputs['hidden_states'] = [_ for _ in outputs['hidden_states']]
            return outputs['hidden_states']
        raise NotImplemented
    
    def forward_wav2vec2_multimodal(
        self, 
        src_tokens, 
        src_lengths, 
        src_audio_path, 
        img_path, 
        img_tensor, 
        imgs_list=[],
        img_masks_list=[],
        tgt_speaker=None, 
        return_all_hiddens=False, 
    ):
        # img_feat_list = None
        # if self.multimodal_translation_flag:
        #     if self.training:
        #         modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
        #         if modality_drop_prob < self.modality_dropout:
        #             if audio_drop_prob < self.audio_dropout:
        #                 img_feat_list = self.forward_visual_extractor(img_tensor)
        #             else:
        #                 img_feat_list = None
        #         else:
        #             img_feat_list = self.forward_visual_extractor(img_tensor)
        #     else:
        #         img_feat_list = self.forward_visual_extractor(img_tensor)
        # if img_feat_list is not None:
        #     # img_feat_list = img_feat_list[-self.num_cross_attention_layers:]
        #     img_feat_list = [img_feat_list[-1]] * self.num_cross_attention_layers
        img_feat_list = self.forward_visual_extractor(img_tensor)
        if self.training:
            modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
            if modality_drop_prob < self.modality_dropout:
                if audio_drop_prob < self.audio_dropout:
                    pass
                else:
                    img_feat_list[-1] = torch.zeros_like(img_feat_list[-1], device=img_feat_list[-1].device, requires_grad=False)
        # img_feat_list = [img_feat_list[-1]] * self.num_cross_attention_layers
        img_feat_list = expand(img_feat_list[-1], self.num_cross_attention_layers)
        padding_mask = lengths_to_padding_mask(src_lengths)
        attention_mask = (~padding_mask).int()
        output = self.wav2vec2_multimodal(
            input_values=src_tokens, 
            attention_mask=attention_mask, 
            m2=img_feat_list, m2_mask=None,
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=True, 
        )
        encoder_padding_mask = forward_padding_mask(
            output["hidden_states"][-1], 
            padding_mask, 
        )
        output["hidden_states"] = [hidden_state.transpose(0, 1) for hidden_state in output["hidden_states"]]
        # output["hidden_states"] = [self.proj_1024_to_768(hidden_state.transpose(0, 1)) for hidden_state in output["hidden_states"]]
        # output["hidden_states"] = [self.wav2vec2_adaptor(hidden_state.transpose(0, 1)) for hidden_state in output["hidden_states"]]
        out = {
            "encoder_out": [output["hidden_states"][-1]], 
            "encoder_padding_mask": [encoder_padding_mask], 
            # "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [], 
            "encoder_states": output["hidden_states"], 
            "encoder_embedding": [], 
        }
        if self.spk_emb_proj:
            x = out["encoder_out"][0]
            seq_len, bsz, _ = x.size()
            tgt_speaker_emb = tgt_speaker.view(1, bsz, -1).expand(seq_len, bsz, -1)
            x = self.spk_emb_proj(torch.cat([x, tgt_speaker_emb], dim=2))
            out["encoder_out"][0] = x
        return out

    def forward(
        self, 
        src_tokens, 
        src_lengths, 
        src_audio_path, 
        img_path, 
        img_tensor, 
        imgs_list=[],
        img_masks_list=[],
        tgt_speaker=None, 
        return_all_hiddens=False, 
        **kwargs
    ):
        if self.only_img:
            out = {
                "encoder_out": [None], 
                "encoder_padding_mask": [None], 
                "encoder_states": [], 
                "encoder_embedding": [], 
            }
        if getattr(self, "multimodal_attention_type", "") == "wav2vec2_multimodal" and not self.only_img:
            return self.forward_wav2vec2_multimodal(
                src_tokens, src_lengths, 
                src_audio_path, img_path,
                img_tensor, imgs_list, img_masks_list,
                tgt_speaker, return_all_hiddens
            )
        if self.multimodal_translation_flag:
            xs = []
            idx = 0
        if self.mhubert_flag and not self.only_img:
            # self.mhubert.eval()
            encoder_padding_mask = lengths_to_padding_mask(src_lengths)
            if isinstance(self.mhubert, HubertModel):
                features, encoder_padding_mask = self.mhubert.extract_features(
                    src_tokens, 
                    encoder_padding_mask, 
                    mask=True if self.training else False, 
                    output_layer=None, 
                    # output_layer=10, 
                )
            elif isinstance(self.mhubert, Wav2Vec2Model):
                res = self.mhubert.extract_features(
                    src_tokens, 
                    encoder_padding_mask, 
                    mask=True if self.training else False, 
                    layer=None, 
                )
                features, encoder_padding_mask = res["x"], res["padding_mask"]
            # features = self.proj_768_to_512(features)
            # features = self.proj_1024_to_512(features)
            # logger.info(f"666 {features.shape} {encoder_padding_mask.shape}") [B, T, C]
            # features, encoder_padding_mask = self.wav2vec2_adaptor(features.transpose(0, 1), encoder_padding_mask)
            # features = features.transpose(0, 1)
            out = {
                "encoder_out": [features.transpose(0, 1)], 
                # "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [], 
                "encoder_padding_mask": [encoder_padding_mask], 
                "encoder_states": [features.transpose(0, 1)] * 12, 
                "encoder_embedding": [], 
            }
        elif self.wav2vec2_flag and not self.only_img:
            padding_mask = lengths_to_padding_mask(src_lengths)
            attention_mask = (~padding_mask).int()
            output = self.wav2vec2(
                input_values=src_tokens, 
                attention_mask=attention_mask, 
                output_attentions=True, 
                output_hidden_states=True, 
                return_dict=True, 
            )
            encoder_padding_mask = forward_padding_mask(
                output["hidden_states"][-1], 
                padding_mask, 
            )
            output["hidden_states"] = [hidden_state.transpose(0, 1) for hidden_state in output["hidden_states"]]
            # output["hidden_states"] = [self.proj_1024_to_768(hidden_state.transpose(0, 1)) for hidden_state in output["hidden_states"]]
            # output["hidden_states"] = [self.wav2vec2_adaptor(hidden_state.transpose(0, 1)) for hidden_state in output["hidden_states"]]
            out = {
                "encoder_out": [output["hidden_states"][-1]], 
                "encoder_padding_mask": [encoder_padding_mask], 
                # "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [], 
                "encoder_states": output["hidden_states"], 
                "encoder_embedding": [], 
            }
        elif not self.only_img:
            out = super().forward(src_tokens, src_lengths, return_all_hiddens)
        if self.spk_emb_proj and not self.only_img:
            x = out["encoder_out"][0]
            seq_len, bsz, _ = x.size()
            tgt_speaker_emb = tgt_speaker.view(1, bsz, -1).expand(seq_len, bsz, -1)
            x = self.spk_emb_proj(torch.cat([x, tgt_speaker_emb], dim=2))
            out["encoder_out"][0] = x
        if self.multimodal_translation_flag and self.is_fusion_top and \
            imgs_list:
            # logger.warning("enter multimodal translation! ")
            img_feat_list = None
            if self.load_visual_extractor_type is not None and self.load_visual_extractor_type != "":
                # custom your visual encoder
                assert len(imgs_list) == 1, "now only support one type of image faeatures! "
                img_feat_list = self.forward_visual_extractor(img_tensor)
                if self.multimodal_extractor_type is not None and self.multimodal_extractor_type == "q_former":
                    if not self.only_img:
                        img_feat_list[-1] = self.q_former(
                            m1=out["encoder_out"][0].transpose(0, 1),
                            m2=img_feat_list[-1],
                            m1_key_padding_mask=out["encoder_padding_mask"][0],
                            m2_key_padding_mask=None,
                        )
                    else:
                        img_feat_list[-1] = self.q_former(
                            m1=None,
                            m2=img_feat_list[-1],
                            m1_key_padding_mask=None,
                            m2_key_padding_mask=None,
                        )
                imgs_list[0] = img_feat_list[-1]
            # if self.training: 
            if self.training and not self.only_img:
                modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
                if modality_drop_prob < self.modality_dropout:
                    if audio_drop_prob < self.audio_dropout:
                        out["encoder_out"][0] = torch.zeros_like(out["encoder_out"][0], device=imgs_list[i].device, requires_grad=False)
                        # out["encoder_out"][0] = 0 * out["encoder_out"][0]
                        # out["encoder_states"] = [out["encoder_out"][0] for i in range(len(out["encoder_states"]))]
                    else:
                        for i in range(len(imgs_list)):
                            imgs_list[i] = torch.zeros_like(imgs_list[i], device=imgs_list[i].device, requires_grad=False)
                            # imgs_list[i] = 0 * imgs_list[i]
                            # img_masks_list[i] = torch.full(
                            #     (imgs_list[i].shape[0], imgs_list[i].shape[1]), 
                            #     fill_value=True, dtype=torch.bool, 
                            #     device=imgs_list[i].device,
                            #     requires_grad=False,
                            # )
            for img, img_mask in zip(imgs_list, img_masks_list):
                img = img.transpose(0, 1)
                if self.only_img:
                    out["encoder_out"] = [img]
                    out["encoder_padding_mask"] = [torch.full(
                        size=(img.shape[1], img.shape[0]),
                        fill_value=False, dtype=torch.bool,
                        device=img.device,
                    )]
                    out["encoder_states"] = out["encoder_out"]
                    break
                if self.multimodal_attention_type in ("selective_attention", "multimodal_attention"):
                    res, text_mask = self.fuse_img_feat(
                        out["encoder_out"][0], idx, img, img_mask, 
                        text_mask=out["encoder_padding_mask"][0], 
                    )
                    # out["encoder_states"] = [res for i in range(len(out["encoder_states"]))]
                    out["encoder_padding_mask"][0] = text_mask
                elif self.multimodal_attention_type in ("external_multimodal_transformer"):
                    speech_feat, img_feat = None, None
                    if img_feat_list is not None:
                        if self.load_visual_extractor_type is not None and self.load_visual_extractor_type in ("vit_openai", "vit_huggingface"):
                            # logger.info(f"666 enter multimodal transformer")
                            img_feat = [i.transpose(0, 1) for i in img_feat_list[-self.external_multimodal_transformer_layers:]]
                        else:
                            # img_feat = [img] * self.external_multimodal_transformer_layers
                            img_feat = expand(img, self.external_multimodal_transformer_layers)
                    elif not isinstance(img, list) and not isinstance(img, tuple):
                        # img_feat = [img] * self.external_multimodal_transformer_layers
                        img_feat = expand(img, self.external_multimodal_transformer_layers)
                    # if out["encoder_states"] is None or len(out["encoder_states"]) == 0:
                    if out["encoder_states"] is None:
                        # speech_feat = [out["encoder_out"][0]] * self.external_multimodal_transformer_layers
                        speech_feat = expand(out["encoder_out"][0], self.external_multimodal_transformer_layers)
                    else:
                        speech_feat = out["encoder_states"][-self.external_multimodal_transformer_layers:]
                    res = self.fuse_speech_img(
                        idx=idx, 
                        speech_feat=speech_feat, img_feat=img_feat, 
                        speech_padding_mask=out["encoder_padding_mask"][0], 
                        img_padding_mask=img_mask, 
                    )
                else:
                    raise NotImplementedError()
                xs.append(res)
                idx += 1
            if not self.only_img:
                out["encoder_out"][0] = self.f(xs, fun='sum')

        return out
    
    def f(self, l, fun='sum'):
        if fun == 'avg':
            size = len(l)
            res = l[0]
            for i in l[1:]:
                res = res + i
            return res / size
        elif fun == 'sum':
            res = l[0]
            for i in l[1:]:
                res = res + i
            return res
    
    def fuse_speech_img(
        self, idx, 
        speech_feat, img_feat, 
        speech_padding_mask, img_padding_mask,
    ):
        if self.multimodal_attention_type == "external_multimodal_transformer":
            output = self.multimodal_transformer[idx](
                m1=speech_feat, m2=img_feat, 
                m1_mask=None, m2_mask=None, 
                m1_key_padding_mask=speech_padding_mask, 
                m2_key_padding_mask=img_padding_mask, 
            )
            return output
        else:
            raise NotImplementedError()


    def fuse_img_feat(self, text, idx, image, image_mask, text_mask):
        image = self.image_pre_norm_module(image)
        image = self.image_dropout_module(image)
        text = self.text_dropout_module(text)
        mask = text_mask
        if self.multimodal_attention_type == "selective_attention":
            output, _map = self.selective_attns[idx](
                query=text, key=image, value=image, key_padding_mask=image_mask
            )   # t, b, c
            # output, _map = self.selective_attns[idx](
            #     query=text, key=image, value=image, key_padding_mask=text_mask
            # )   # t, b, c
        elif self.multimodal_attention_type == "multimodal_attention":
            output, mask = self.multimodal_attns[idx](
                text=text, text_mask=text_mask, 
                img=image, img_mask=image_mask, 
                is_merge_text_img=self.is_merge_text_img, 
            )
        if self.use_selective_gate:
            merge = torch.cat([output, text], dim=-1)
            gate = torch.sigmoid(self.gate_denses[idx](merge))
            # self.recoder.record_gate(gate.cpu(), text_mask.cpu())
            # _map = _map[:,:,1:].softmax(dim=-1)
            # self.recoder.record_map(_map.cpu())
            res = (1 - gate) * text + gate * output
        else:
            # res = output
            res = text + output
        return res, mask, 


@register_model("mm_s2ut_transformer")
class MM_S2UTTransformerModel(S2UTTransformerModel):
    @classmethod
    def build_encoder(cls, args):
        encoder = MM_S2STransformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder
    
    def forward_encoder(
        self, 
        src_tokens, 
        src_lengths, 
        src_audio_path, 
        img_path, 
        img_tensor, 
        imgs_list,
        img_masks_list,
        speaker=None, 
        **kwargs
    ):
        return self.encoder(
            src_tokens, 
            src_lengths=src_lengths, 
            src_audio_path=src_audio_path, 
            img_path=img_path, 
            img_tensor=img_tensor, 
            imgs_list=imgs_list, 
            img_masks_list=img_masks_list, 
            tgt_speaker=speaker, 
            **kwargs
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_audio_path, 
        img_path, 
        img_tensor, 
        imgs_list=[],
        img_masks_list=[],
        tgt_speaker=None,
        return_all_hiddens=False,
        **kwargs
    ):
        encoder_out = self.forward_encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_audio_path=src_audio_path, 
            img_path=img_path, 
            img_tensor=img_tensor, 
            imgs_list=imgs_list, 
            img_masks_list=img_masks_list, 
            speaker=tgt_speaker,
            return_all_hiddens=return_all_hiddens,
            **kwargs
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
        )
        if return_all_hiddens:
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_states"]
            decoder_out[-1]["encoder_padding_mask"] = encoder_out["encoder_padding_mask"]
        return decoder_out


@register_model_architecture(
    model_name="mm_s2ut_transformer", arch_name="mm_s2ut_transformer"
)
def mm_s2ut_architecture_base(args):
    s2ut_architecture_base(args)
    # args.decoder_layers = getattr(args, "decoder_layers", 12)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    # args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
