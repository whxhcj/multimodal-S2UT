import torch
from torch import nn, Tensor
from torch.nn import (
    MultiheadAttention, 
    Dropout, 
    LayerNorm,
)
from torch.nn.modules.transformer import _get_activation_fn
import torch.nn.functional as F
import transformers
from pathlib import Path
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.incremental_decoding_utils import with_incremental_state
from typing import Any, Dict, List, Optional, Tuple, Union, Callable


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def get_length(t: Union[List, torch.Tensor]):
    if isinstance(t, list) or isinstance(t, tuple):
        return len(t)
    elif isinstance(t, torch.Tensor):
        return t.shape[0]
    else:
        raise NotImplementedError("")


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

        # self.mask_as_text = True
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
                # attn = attn.transpose(1, 2) # (bsz, num_heads, Tq, Tk) -> (bsz, Tq, num_heads, Tk)
                # attn = attn.view(bsz, Tq, self.num_heads * Tk)
                # key_padding_mask = key_padding_mask.to(torch.bool)
                # key_padding_mask = key_padding_mask.reshape(bsz, Tq, 1)
                # key_padding_mask = key_padding_mask.expand(-1, -1, self.num_heads * Tk)
                # attn = attn.masked_fill(key_padding_mask, float("-inf"))
                # attn = attn.view(bsz, Tq, self.num_heads, Tk)
                # attn = attn.transpose(1, 2) # (bsz, Tq, num_heads, Tk) -> (bsz, num_heads, Tq, Tk)
                attn = attn.view(bsz * self.num_heads, Tq, Tk)
                key_padding_mask = key_padding_mask.to(torch.bool)
                key_padding_mask = key_padding_mask.view(bsz, 1, Tq, 1)
                key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1)
                key_padding_mask = key_padding_mask.reshape(bsz * self.num_heads, Tq, 1)
                attn = attn.masked_fill(key_padding_mask, float("-inf"))
                # attn = attn.view(bsz, self.num_heads, Tq, Tk)
            attn = attn.view(bsz * self.num_heads, Tq, Tk)

        attn = attn.softmax(dim=-1)
        attn_after_drop = self.attn_drop(attn)

        x = (attn_after_drop @ v)
        assert list(x.size()) == [bsz * self.num_heads, Tq, self.vhead_dim]
        x = x.transpose(0, 1).contiguous().view(Tq, bsz, self.intermediate_dim)
        x = self.proj(x)
        return x, attn


class MultimodalAttention(nn.MultiheadAttention):
    def __init__(
        self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
        kdim=None, vdim=None, batch_first=False, device=None, dtype=None, 
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
            kdim=kdim, vdim=vdim, batch_first=batch_first, device=device, dtype=dtype, 
        )

    def merge_text_image(self, text, text_mask, img):
        # text = [t1, b, d1]
        # image(vit_timm) = [t2=577, b, d2=768]
        # text_padding_mask = [b, t3] t1 = floor(t3 / 4)
        # text_attention_mask = [b, t1]
        # text_attention_mask = [False, False, ……, True, True]
        # img_attention_mask = [b, t2]
        text_t, text_b, text_d = text.shape
        img_t, img_b, img_d = img.shape
        assert text_b == img_b and text_d == img_d
        text_img = torch.cat([text, img], dim=0) # [t1 + t2, b, d1]
        img_mask = torch.full((img_b, img_t), False, device=text_mask.device)
        text_img_mask = torch.cat([text_mask, img_mask], dim=1)
        return text_img, text_img_mask, 

    def forward(
        self, text, text_mask, img, img_mask, 
        is_merge_text_img, 
        need_weights: bool = True,
        average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # text_mask = [b, t1]
        # img_mask = [b, t2]
        # text_img_mask = [b, t1, t2]
        if is_merge_text_img:
            text, text_mask = self.merge_text_image(text, text_mask, img)
        # attn_mask = text_mask[:, :, None].repeat(self.num_heads, 1, img.shape[0])
        attn_mask = None
        query = text
        key, value = img, img
        key_padding_mask = img_mask
        attn_output, attn_output_weights = super().forward(
            query=query, key=key, value=value, key_padding_mask=key_padding_mask,
            need_weights=need_weights, attn_mask=attn_mask,
            average_attn_weights=average_attn_weights, 
        )
        # logger.info(f"666 text = {text.shape} img = {img.shape} text_mask = {text_mask.shape}")
        return attn_output, text_mask, 


@dataclass
class TransformerLayerConfig(FairseqDataclass):
    d_model: int = field(default=768)
    embed_dim: int = field(default=768)
    kdim: int = field(default=768)
    vdim: int = field(default=768)
    nhead: int = field(default=12)
    dim_feedforward: int = field(default=3072)
    dropout: float = field(default=0.1)
    activation: Any = field(default=F.gelu)
    layer_norm_eps: float = field(default=1e-5)
    batch_first: bool = field(default=False)
    norm_first: bool = field(default=False)
    device: Any = field(default=None)
    dtype: Any = field(default=None)


class MultimodalTransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(
        self, embed_dim: int, kdim: int, vdim: int, 
        nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5, batch_first: bool = False, 
        norm_first: bool = False,
        self_attention_first: bool = True,
        device=None, dtype=None, 
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, 
            batch_first=batch_first, **factory_kwargs
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=batch_first,
            kdim=kdim, vdim=vdim, 
            **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.self_attention_first = self_attention_first

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self, 
        tgt: Tensor, memory: Tensor, 
        tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            if self.self_attention_first:
                x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            else:
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            if self.self_attention_first:
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            else:
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class ExternalMultimodalTransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(
        self, 
        layer_config: TransformerLayerConfig = TransformerLayerConfig(),
        num_layers: int = 6, 
        norm=None, 
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        decoder_layer = MultimodalTransformerDecoderLayer(
            embed_dim=layer_config.embed_dim, 
            kdim=layer_config.kdim,
            vdim=layer_config.vdim,
            nhead=layer_config.nhead, 
            dim_feedforward=layer_config.dim_feedforward,
            dropout=layer_config.dropout,
            activation=layer_config.activation, 
            layer_norm_eps=layer_config.layer_norm_eps, 
            batch_first=layer_config.batch_first, 
            norm_first=layer_config.norm_first, 
            device=layer_config.device, 
            dtype=layer_config.dtype, 
        )
        self.layers = nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.layer_norm1 = nn.LayerNorm(layer_config.embed_dim, eps=layer_config.layer_norm_eps)

    def forward(
        self, m1: List[Tensor], m2: List[Tensor], 
        m1_mask: Optional[Tensor] = None,
        m2_mask: Optional[Tensor] = None, 
        m1_key_padding_mask: Optional[Tensor] = None,
        m2_key_padding_mask: Optional[Tensor] = None, 
    ) -> Tensor:
        """
        if batch_first:
            m1: [[b, t_m1, d_m1]]
            m2: [[b, t_m2, d_m2]]
            last_m1: [b, t_m1, d_m1]
            m1_mask: [t_m1, t_m1] or [num_heads_m1, t_m1, t_m1]
            m2_mask: [t_m1, t_m2] or [num_heads_m1, t_m2, t_m1]
            m1_key_padding_mask: [b, t_m1]
            m2_key_padding_mask: [b, t_m2]
        else:
            m1: [[t_m1, b, d_m1]]
            m2: [[t_m2, b, d_m2]]
            last_m1: [t_m1, b, d_m1]
            m1_mask: [t_m1, t_m1] or [num_heads_m1, t_m1, t_m1]
            m2_mask: [t_m1, t_m2] or [num_heads_m1, t_m2, t_m1]
            m1_key_padding_mask: [b, t_m1]
            m2_key_padding_mask: [b, t_m2]      
        """
        assert self.num_layers == get_length(m1) == get_length(m2)
        last_m1 = None
        for i in range(self.num_layers):
            output = m1[i]
            if last_m1 is not None:
                output = self.layer_norm1(output + last_m1)
            output = self.layers[i](
                tgt=output, memory=m2[i], 
                tgt_mask=m1_mask, memory_mask=m2_mask,
                tgt_key_padding_mask=m1_key_padding_mask,
                memory_key_padding_mask=m2_key_padding_mask, 
            )
            last_m1 = output
        if self.norm is not None:
            output = self.norm(output)
        return output


class BridgeTowerTransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(
        self, 
        layer_config: TransformerLayerConfig = TransformerLayerConfig(),
        num_layers: int = 6, 
        norm=None, 
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        decoder_layer = MultimodalTransformerDecoderLayer(
            embed_dim=layer_config.embed_dim, 
            kdim=layer_config.kdim,
            vdim=layer_config.vdim,
            nhead=layer_config.nhead, 
            dim_feedforward=layer_config.dim_feedforward,
            dropout=layer_config.dropout,
            activation=layer_config.activation, 
            layer_norm_eps=layer_config.layer_norm_eps, 
            batch_first=layer_config.batch_first, 
            norm_first=layer_config.norm_first, 
            device=layer_config.device, 
            dtype=layer_config.dtype, 
        )
        self.layers_m1 = nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.layers_m2 = nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.layer_norm1 = nn.LayerNorm(layer_config.embed_dim, eps=layer_config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(layer_config.embed_dim, eps=layer_config.layer_norm_eps)

    def forward(
        self, m1: List[Tensor], m2: List[Tensor], 
        m1_mask: Optional[Tensor] = None,
        m2_mask: Optional[Tensor] = None, 
        m1_key_padding_mask: Optional[Tensor] = None,
        m2_key_padding_mask: Optional[Tensor] = None, 
    ):
        """
        if batch_first:
            m1: [[b, t_m1, d_m1]]
            m2: [[b, t_m2, d_m2]]
            last_m1: [b, t_m1, d_m1]
            m1_mask: [t_m1, t_m1] or [num_heads_m1, t_m1, t_m1]
            m2_mask: [t_m1, t_m2] or [num_heads_m1, t_m2, t_m1]
            m1_key_padding_mask: [b, t_m1]
            m2_key_padding_mask: [b, t_m2]
        else:
            m1: [[t_m1, b, d_m1]]
            m2: [[t_m2, b, d_m2]]
            last_m1: [t_m1, b, d_m1]
            m1_mask: [t_m1, t_m1] or [num_heads_m1, t_m1, t_m1]
            m2_mask: [t_m1, t_m2] or [num_heads_m1, t_m2, t_m1]
            m1_key_padding_mask: [b, t_m1]
            m2_key_padding_mask: [b, t_m2]      
        """
        if isinstance(m1, torch.Tensor) and isinstance(m2, torch.Tensor):
            assert self.num_layers == m1.shape[0] == m2.shape[0]
        else:
            assert self.num_layers == len(m1) == len(m2)
        last_f1, last_f2 = None, None
        for i in range(self.num_layers):
            last_m1 = m1[i]
            last_m2 = m2[i]
            if last_f1 is not None:
                last_f1 = self.layer_norm1(last_m1 + last_f1)
            else:
                last_f1 = self.layer_norm1(last_m1)
            if last_f2 is not None:
                last_f2 = self.layer_norm2(last_m2 + last_f2)
            else:
                last_f2 = self.layer_norm2(last_m2)
            last_f1 = self.layers_m1[i](
                tgt=last_f1, memory=last_f2, 
                tgt_mask=m1_mask, memory_mask=m2_mask,
                tgt_key_padding_mask=m1_key_padding_mask,
                memory_key_padding_mask=m2_key_padding_mask, 
            )
            last_f2 = self.layers_m2[i](
                tgt=last_f2, memory=last_f1, 
                tgt_mask=m2_mask, memory_mask=m1_mask,
                tgt_key_padding_mask=m2_key_padding_mask,
                memory_key_padding_mask=m1_key_padding_mask, 
            )
        if self.norm is not None:
            last_f1 = self.norm(last_f1)
            last_f2 = self.norm(last_f2)
        return last_f1, last_f2,


class Wav2Vec2WithMultiModal(nn.Module):
    _HIDDEN_STATES_START_POSITION: int = 2

    def __init__(
        self,
        wav2vec2: Union[str, Path, transformers.Wav2Vec2ForPreTraining, transformers.Wav2Vec2Model, transformers.Wav2Vec2ForCTC],
        num_cross_attention_layers: int = 2,
        wav2vec2_embed_dim: int = 768,
        m2_dim: int = 768,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        if isinstance(wav2vec2, str) or isinstance(wav2vec2, Path):
            self.wav2vec2 = transformers.Wav2Vec2ForCTC.from_pretrained(
                wav2vec2, 
                # gradient_checkpointing=True,
            )
            self.wav2vec2.freeze_feature_extractor()
        else:
            self.wav2vec2 = wav2vec2
        assert isinstance(self.wav2vec2, transformers.Wav2Vec2ForPreTraining) \
            or isinstance(self.wav2vec2, transformers.Wav2Vec2ForCTC), \
            "Not Implemented"
        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_wav2vec2_layers = len(self.wav2vec2.wav2vec2.encoder.layers)
        self.batch_first = batch_first
        self.cross_attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=wav2vec2_embed_dim,
                    num_heads=wav2vec2_embed_dim // 64, 
                    dropout=dropout,
                    kdim=m2_dim,
                    vdim=m2_dim,
                    batch_first=self.batch_first,
                ) for _ in range(self.num_cross_attention_layers)
            ]
        )
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(self.num_cross_attention_layers)]
        )
        self.layer_norm_layers = nn.ModuleList(
            [
                nn.LayerNorm(
                    self.wav2vec2.config.hidden_size, 
                    eps=self.wav2vec2.config.layer_norm_eps
                ) for _ in range(self.num_cross_attention_layers)
            ]
        )

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        m2: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        m2_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if isinstance(self.wav2vec2, transformers.Wav2Vec2ForCTC):
            return self.forward_Wav2Vec2ForCTC(
                wav2vec2=self.wav2vec2,
                input_values=input_values,
                attention_mask=attention_mask,
                m2=m2, m2_mask=m2_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
        else:
            raise NotImplementedError()
    
    def forward_Wav2Vec2EncoderLayer(
        self, 
        layer: transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer,
        idx: int, 
        hidden_states, 
        attention_mask=None, 
        m2: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        m2_mask: Optional[torch.Tensor] = None,
        output_attentions=False,
    ):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = layer.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = layer.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = layer.layer_norm(hidden_states)
        idx = idx + self.num_cross_attention_layers - self.num_wav2vec2_layers
        if (m2 is not None and len(m2) > 0) and idx >= 0:
            # print(idx)
            # m2 = m2.transpose(0, 1)
            attn_residual, _ = self.cross_attention_layers[idx](
                # query=hidden_states.transpose(0, 1), 
                query=hidden_states,
                key=m2[idx], value=m2[idx], 
                # key_padding_mask=m2_mask[idx],
                key_padding_mask=m2_mask,
            )
            # attn_residual = attn_residual.transpose(0, 1)
            attn_residual = self.dropout_layers[idx](attn_residual)
            hidden_states = attn_residual + hidden_states
            hidden_states = self.layer_norm_layers[idx](hidden_states)
        hidden_states = hidden_states + layer.feed_forward(hidden_states)
        hidden_states = layer.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
        
    def forward_Wav2Vec2Encoder(
        self,
        encoder: transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        m2: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        m2_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )
        position_embeddings = encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = encoder.layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)
        deepspeed_zero3_is_enabled = transformers.deepspeed.is_deepspeed_zero3_enabled()
        # for layer in encoder.layers:
        for i in range(len(encoder.layers)):
            layer = encoder.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])
            skip_the_layer = True if self.training and (dropout_probability < encoder.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if encoder.gradient_checkpointing and encoder.training:
                # if True:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)
                        return custom_forward
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        # create_custom_forward(layer),
                        create_custom_forward(self.forward_Wav2Vec2EncoderLayer),
                        layer, i,
                        hidden_states,
                        attention_mask,
                        m2, m2_mask,
                    )
                else:
                    layer_outputs = self.forward_Wav2Vec2EncoderLayer(
                        layer, i,
                        hidden_states, 
                        attention_mask=attention_mask, 
                        m2=m2, m2_mask=m2_mask,
                        output_attentions=output_attentions,
                    )
                    # print(i)
                    # layer_outputs = layer(
                    #     hidden_states, 
                    #     attention_mask=attention_mask, 
                    #     output_attentions=output_attentions,
                    # )
                hidden_states = layer_outputs[0]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def forward_Wav2Vec2Model(
        self,
        wav2vec2:transformers.Wav2Vec2Model,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        m2: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        m2_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else wav2vec2.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else wav2vec2.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else wav2vec2.config.use_return_dict
        extract_features = wav2vec2.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = wav2vec2._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
        hidden_states, extract_features = wav2vec2.feature_projection(extract_features)
        hidden_states = wav2vec2._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )
        encoder_outputs = self.forward_Wav2Vec2Encoder(
            wav2vec2.encoder,
            hidden_states,
            attention_mask=attention_mask,
            m2=m2, m2_mask=m2_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # encoder_outputs = wav2vec2.encoder(
        #     hidden_states,
        #     attention_mask=attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        hidden_states = encoder_outputs[0]
        if wav2vec2.adapter is not None:
            hidden_states = wav2vec2.adapter(hidden_states)
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]
        return transformers.modeling_outputs.Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
    def forward_Wav2Vec2ForCTC(
        self,
        wav2vec2: transformers.Wav2Vec2ForCTC,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        m2: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        m2_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.wav2vec2.config.use_return_dict
        outputs = self.forward_Wav2Vec2Model(
            wav2vec2.wav2vec2,
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            m2=m2, m2_mask=m2_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # outputs = wav2vec2.wav2vec2(
        #     input_values,
        #     attention_mask=attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        hidden_states = outputs[0]
        hidden_states = wav2vec2.dropout(hidden_states)
        logits = wav2vec2.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if labels.max() >= wav2vec2.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {wav2vec2.config.vocab_size}")
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=wav2vec2.config.pad_token_id,
                    reduction=wav2vec2.config.ctc_loss_reduction,
                    zero_infinity=wav2vec2.config.ctc_zero_infinity,
                )
        if not return_dict:
            output = (logits,) + outputs[self._HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
        return transformers.modeling_outputs.CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    

class QFormerModel(nn.Module):
    def __init__(
        self, 
        num_queries: int = 32,
        layer_config: TransformerLayerConfig = TransformerLayerConfig(),
        num_query_layers: int = 4,
        num_multimodal_layers: int = 2,
        self_attention_first: bool = False,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.query_embedding = nn.Parameter(torch.zeros(1, num_queries, layer_config.embed_dim))
        self.num_query_layers = num_query_layers
        self.num_multimodal_layers = num_multimodal_layers
        self.norm = norm

        query_transformer_layer = MultimodalTransformerDecoderLayer(
            embed_dim=layer_config.embed_dim, 
            kdim=layer_config.kdim,
            vdim=layer_config.vdim,
            nhead=layer_config.nhead, 
            dim_feedforward=layer_config.dim_feedforward,
            dropout=layer_config.dropout,
            activation=layer_config.activation, 
            layer_norm_eps=layer_config.layer_norm_eps, 
            batch_first=layer_config.batch_first, 
            norm_first=layer_config.norm_first, 
            self_attention_first=self_attention_first,
            device=layer_config.device, 
            dtype=layer_config.dtype, 
        )
        self.query_transformer_layers = nn.modules.transformer._get_clones(query_transformer_layer, self.num_query_layers)

        multimodal_transformer_layer = MultimodalTransformerDecoderLayer(
            embed_dim=layer_config.embed_dim, 
            kdim=layer_config.kdim,
            vdim=layer_config.vdim,
            nhead=layer_config.nhead, 
            dim_feedforward=layer_config.dim_feedforward,
            dropout=layer_config.dropout,
            activation=layer_config.activation, 
            layer_norm_eps=layer_config.layer_norm_eps, 
            batch_first=layer_config.batch_first, 
            norm_first=layer_config.norm_first, 
            self_attention_first=self_attention_first,
            device=layer_config.device, 
            dtype=layer_config.dtype, 
        )
        self.multimodal_transformer_layers = nn.modules.transformer._get_clones(multimodal_transformer_layer, self.num_multimodal_layers)

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        m1: Optional[Tensor] = None,
        m2: Optional[Tensor] = None,
        m1_key_padding_mask: Optional[Tensor] = None,
        m2_key_padding_mask: Optional[Tensor] = None, 
    ) -> Tensor:
        """
        if batch_first:
            m1: [[b, t_m1, d_m1]]
            m2: [[b, t_m2, d_m2]]
            last_m1: [b, t_m1, d_m1]
            m1_mask: [t_m1, t_m1] or [num_heads_m1, t_m1, t_m1]
            m2_mask: [t_m1, t_m2] or [num_heads_m1, t_m2, t_m1]
            m1_key_padding_mask: [b, t_m1]
            m2_key_padding_mask: [b, t_m2]
        else:
            m1: [[t_m1, b, d_m1]]
            m2: [[t_m2, b, d_m2]]
            last_m1: [t_m1, b, d_m1]
            m1_mask: [t_m1, t_m1] or [num_heads_m1, t_m1, t_m1]
            m2_mask: [t_m1, t_m2] or [num_heads_m1, t_m2, t_m1]
            m1_key_padding_mask: [b, t_m1]
            m2_key_padding_mask: [b, t_m2]      
        """
        if m1 is not None:
            output = self.query_embedding.expand(m1.shape[0], -1, -1)
        elif m2 is not None:
            output = self.query_embedding.expand(m2.shape[0], -1, -1)
        else:
            raise ValueError("all of inputs m1 and m2 are none!")
        for i in range(self.num_query_layers):
            output = self.query_transformer_layers[i](
                tgt=output, memory=m1,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=m1_key_padding_mask,
            )
        for i in range(self.num_multimodal_layers):
            output = self.multimodal_transformer_layers[i](
                tgt=output, memory=m2,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=m2_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output            
