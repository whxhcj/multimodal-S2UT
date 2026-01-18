import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from omegaconf import OmegaConf
import torch
from torch import Tensor, nn
import numpy as np
import copy

from fairseq import checkpoint_utils, utils
from fairseq.data.audio.data_cfg import get_config_from_yaml
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.hubert import HubertModel
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.models.speech_to_text import (
    XMTransformerModel, 
    base_architecture, 
    build_embedding, 
    Wav2VecEncoderWithAdaptor, 
    remove_weight_norm_from_model, 
)
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

logger = logging.getLogger(__name__)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@with_incremental_state
class SelectiveAttention(nn.Module):
    def __init__(
        self, 
        qdim, kdim, vdim, 
        attn_dim, intermediate_dim, 
        output_dim, num_heads=1, qkv_bias=True, attn_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim

        self.qkhead_dim = attn_dim // num_heads
        self.vhead_dim = intermediate_dim // num_heads               
        self.scale = self.qkhead_dim ** -0.5

        self.q_proj = Linear(qdim, attn_dim, bias=qkv_bias)
        self.k_proj = Linear(kdim, attn_dim, bias=qkv_bias)
        self.v_proj = Linear(vdim, intermediate_dim, bias=qkv_bias)   
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(intermediate_dim, output_dim)

        self.mask_as_text = False

    def forward(self, query, key, value, key_padding_mask=None):
        Tq, Bq, Cq = query.shape
        Tk, Bk, Ck = key.shape
        Tv, Bv, Cv = value.shape
        assert Bq == Bk == Bv
        assert Tk == Tv
        assert Cq == self.qdim
        assert Ck == self.kdim
        assert Cv == self.vdim
        bsz = Bq
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
       
        q *= self.scale
        
        q = q.contiguous().view(Tq, bsz * self.num_heads, self.qkhead_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.qkhead_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.vhead_dim).transpose(0, 1)
        # B * H, T, C//H

        attn = (q @ k.transpose(-2, -1)) 
        if key_padding_mask is not None:
            attn = attn.view(bsz, self.num_heads, Tq, Tk)
            if not self.mask_as_text:
                attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            else:
                attn = attn.transpose(1, 2) # (bsz, num_heads, Tq, Tk) -> (bsz, Tq, num_heads, Tk)
                attn = attn.view(bsz, Tq, self.num_heads * Tk)
                attn = attn.masked_fill(key_padding_mask.to(torch.bool), float("-inf"))
                attn = attn.view(bsz, Tq, self.num_heads, Tk)
                attn = attn.transpose(1, 2) # (bsz, Tq, num_heads, Tk) -> (bsz, num_heads, Tq, Tk)
            attn = attn.view(bsz * self.num_heads, Tq, Tk)

        attn = attn.softmax(dim=-1)
        attn_after_drop = self.attn_drop(attn)

        x = (attn_after_drop @ v)
        assert list(x.size()) == [bsz * self.num_heads, Tq, self.vhead_dim]
        x = x.transpose(0, 1).contiguous().view(Tq, bsz, self.intermediate_dim)
        x = self.proj(x)
        return x, attn
    

class Wav2VecEncoderWithAdaptorForMultiModal(Wav2VecEncoderWithAdaptor):
    def __init__(self, args):
        super().__init__(args)
        self.build_multimodal_fusion(args)
        logger.info(f"multimodal_translation_flag = {self.multimodal_translation_flag}")
        logger.info(f"is_fusion_top = {self.is_fusion_top}")
    
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

    def fuse_img_feat(self, text, idx, image, image_mask, text_mask):
        image = self.image_pre_norm_module(image)
        image = self.image_dropout_module(image)
        text = self.text_dropout_module(text)
        output, _map = self.selective_attns[idx](
            query=text, key=image, value=image, key_padding_mask=image_mask
        )   # t, b, c
        merge = torch.cat([output, text], dim=-1)
        gate = torch.sigmoid(self.gate_denses[idx](merge))
        # self.recoder.record_gate(gate.cpu(), text_mask.cpu())
        # _map = _map[:,:,1:].softmax(dim=-1)
        # self.recoder.record_map(_map.cpu())
        res = (1 - gate) * text + gate * output
        return res
    
    def build_multimodal_fusion(self, args):
        # image MMT
        self.multimodal_translation_flag = False
        if getattr(args, "multimodal_translation_config_yaml", None) is not None:
            multimodal_translation_config = OmegaConf.load(Path(args.multimodal_translation_config_yaml))
            self.multimodal_translation_flag = True
        if self.multimodal_translation_flag:
            embed_dim = args.decoder_embed_dim
            encoder_embed_dim = args.decoder_embed_dim
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
            self.gate_denses = nn.ModuleList(
                [
                    nn.Linear(2 * encoder_embed_dim, encoder_embed_dim) \
                        for i in multimodal_translation_config.image_feat_dim
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

    def forward(
        self,
        src_tokens,
        src_lengths,
        **kwargs,
    ):
        if self.multimodal_translation_flag:
            xs = []
            idx = 0
        encoder_out = super().forward(
            src_tokens=src_tokens, src_lengths=src_lengths, **kwargs
        )
        # logger.info(f'777 {"imgs_list" in kwargs}')
        if self.multimodal_translation_flag and self.is_fusion_top:
            # logger.info(f"666 {kwargs.keys()}")
            imgs_list = kwargs["imgs_list"]
            img_masks_list = kwargs["img_masks_list"]
            # modality dropout
            if self.training:
                modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
                if modality_drop_prob < self.modality_dropout:
                    if audio_drop_prob < self.audio_dropout:
                        encoder_out["encoder_out"][0] = 0 * encoder_out["encoder_out"][0]
                        encoder_out["encoder_padding_mask"] = torch.full_like(
                            encoder_out["encoder_padding_mask"],
                            fill_value=False, 
                            dtype=torch.bool, 
                        )
                    else:
                        for i in range(len(imgs_list)):
                            imgs_list[i] = 0 * imgs_list[i]
            for img, img_mask in zip(imgs_list, img_masks_list):
                img = img.transpose(0, 1)
                xs.append(
                    self.fuse_img_feat(
                        encoder_out["encoder_out"][0], idx, img, 
                        img_mask, 
                        text_mask=encoder_out["encoder_padding_mask"], 
                        # text_mask=~encoder_out["encoder_padding_mask"][0], 
                    )
                )
                # logger.info(f'666 {~encoder_out["encoder_padding_mask"][0]}')
                idx += 1
            encoder_out["encoder_out"][0] = self.f(xs, fun='sum')
        return encoder_out


@register_model("mm_xm_transformer")
class MM_XMTransformerModel(XMTransformerModel):
    @classmethod
    def build_encoder(cls, args):
        _args = copy.deepcopy(args)
        if not args.adaptor_proj and not args.encoder_proj:  # V0 arch
            if args.w2v_path:
                state = checkpoint_utils.load_checkpoint_to_cpu(args.w2v_path)
                if state.get("cfg") is not None:
                    encoder_embed_dim = state["cfg"]._content["model"][
                        "encoder_embed_dim"
                    ]
                elif state.get("args") is not None:
                    encoder_embed_dim = state["args"].encoder_embed_dim
                else:
                    raise ValueError(f"Invalid config in {args.w2v_path}")
                _args.decoder_embed_dim = encoder_embed_dim
                del state
            else:
                _args.decoder_embed_dim = args.encoder_embed_dim

        encoder = Wav2VecEncoderWithAdaptorForMultiModal(_args)
        encoder = cls.maybe_load_pretrained(
            encoder, getattr(args, "load_pretrained_encoder_from", None)
        )
        if args.remove_weight_norm:
            # remove the wn for EMA usage
            logger.warning("Removing weight norm from wav2vec encoder")
            remove_weight_norm_from_model(encoder)

        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        # base_architecture(args)
        mm_xm_transformer_architecture_base(args)
        if getattr(args, "load_pretrained_decoder_from", None) is not None:
            ckpt = torch.load(getattr(args, "load_pretrained_decoder_from", None))
            decoder_args_dict = cls.get_decoder_args_from_checkpoint(ckpt["cfg"])
            args = cls.override_decoder_args(args, decoder_args_dict)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        base_model = cls(encoder, decoder)

        # set up multitask decoders
        base_model.multitask_decoders = {}
        for i, (task_name, task_obj) in enumerate(task.multitask_tasks.items()):
            # dummy auxiliary decoder
            if task_obj.args.get_loss_weight(0) == 0:
                continue

            task_decoder = cls.build_multitask_decoder(
                args, task_obj.args, task_obj.target_dictionary, args.decoder_embed_dim
            )

            setattr(base_model, f"{task_name}_decoder", task_decoder)
            decoder_model_cls = (
                FairseqEncoderModel
                if task_obj.args.decoder_type == "ctc"
                else FairseqLanguageModel
            )
            base_model.multitask_decoders[task_name] = decoder_model_cls(
                getattr(base_model, f"{task_name}_decoder")
            )
        return base_model

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens=False,
        **kwargs,
    ):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths, **kwargs
        )
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        if return_all_hiddens:
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_out"]
            # NOTE: from the top layer
            decoder_out[-1]["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ]
        return decoder_out
    

@register_model_architecture(
    model_name="mm_xm_transformer", arch_name="mm_xm_transformer"
)
def mm_xm_transformer_architecture_base(args):
    base_architecture(args)
