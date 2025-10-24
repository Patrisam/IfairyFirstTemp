# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.utils import add_prefix, make_layers

iFairyConfig = None


logger = logging.getLogger(__name__)

class iFairyMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.up_proj_real = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.up_proj_imag = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj_imag", prefix),
        )
        self.gate_proj_real = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj_real", prefix),
        )
        self.gate_proj_imag = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj_imag", prefix),
        )
        self.down_proj_real = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj_real", prefix),
        )
        self.down_proj_imag = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj_imag", prefix),
        )

        self.mlp_sub_norm = iFairyRMSNorm(hidden_size, eps=eps)

        if hidden_act == "relu2":
            self.act_fn = self._iFairy_ReLU2()
        else:   
            raise ValueError(f"Unsupported activation: {hidden_act}. Only relu2 is supported for now.")

    def forward(self, x_real, x_imag):
        up_x_real = self.up_proj_real(x_real) + self.up_proj_imag(x_imag)
        up_x_imag = self.up_proj_imag(x_real) - self.up_proj_real(x_imag)

        gate_x_real = self.gate_proj_real(x_real) + self.gate_proj_imag(x_imag)
        gate_x_imag = self.gate_proj_imag(x_real) - self.gate_proj_real(x_imag)

        act_x_real, act_x_imag = self.act_fn(gate_x_real, gate_x_imag)
        up_act_x_real = (act_x_real * up_x_real + act_x_imag * up_x_imag)
        up_act_x_imag = (act_x_real * up_x_imag - act_x_imag * up_x_real)

        ln_up_act_x_real, ln_up_act_x_imag = self.mlp_sub_norm(up_act_x_real, up_act_x_imag)

        down_x_real = self.down_proj_real(ln_up_act_x_real) + self.down_proj_imag(ln_up_act_x_imag)
        down_x_imag = self.down_proj_imag(ln_up_act_x_real) - self.down_proj_real(ln_up_act_x_imag)
        return down_x_real, down_x_imag

    def _iFairy_ReLU2(
        self, 
        x_real: torch.Tensor, 
        x_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dead_mask = torch.logical_and(x_real < 0, x_imag < 0)
        x_real[dead_mask] = 0
        x_imag[dead_mask] = 0

        x_real = torch.pow(x_real, 2)
        x_imag = torch.pow(x_imag, 2)
        return x_real, x_imag

class ComplexNetRotaryEmbedding(nn.Module):
    def __init__(
        self,
        rope_theta: float,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.base = rope_theta
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_seq_len_cached = max_position_embeddings

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self):
        head_dim = self.hidden_size // self.num_attention_heads
        # 用 float32 防止整型参与幂运算
        t = torch.arange(0, head_dim, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (t / head_dim))  # [D]
        return inv_freq

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, hidden_states_type: torch.dtype):
        batch_size = position_ids.shape[0]
        position_ids = position_ids[:, None, :].to(torch.float32)

        if self.inv_freq.dim() == 1:
            self.inv_freq = (
                self.inv_freq[None, :, None]
                .expand(batch_size, -1, 1)
                .to(position_ids.device)
            )

        if position_ids.shape[0] > self.max_seq_len_cached:
            print("Truncate position_ids within max_seq_len_cached.")
            position_ids = position_ids[: self.max_seq_len_cached]

        theta = (self.inv_freq.to(position_ids.dtype) @ position_ids).transpose(1, 2)
        cos_emb = torch.cos(theta).to(hidden_states_type)
        sin_emb = torch.sin(theta).to(hidden_states_type)
        return cos_emb, sin_emb

def _apply_rotary_pos_emb(
    q_real: torch.Tensor,
    q_imag: torch.Tensor,
    k_real: torch.Tensor,
    k_imag: torch.Tensor,
    cos_emb: torch.Tensor,
    sin_emb: torch.Tensor,
) -> tuple:

    def _apply_rotation(
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        cos_emb: torch.Tensor,
        sin_emb: torch.Tensor,
    ) -> torch.Tensor:
        cos_emb = cos_emb.unsqueeze(1)
        sin_emb = sin_emb.unsqueeze(1)

        rotated_x_real = x_real * cos_emb - x_imag * sin_emb
        rotated_x_imag = x_real * sin_emb + x_imag * cos_emb

        return rotated_x_real, rotated_x_imag

    rotated_q_real, rotated_q_imag = _apply_rotation(q_real, q_imag, cos_emb, sin_emb)
    rotated_k_real, rotated_k_imag = _apply_rotation(k_real, k_imag, cos_emb, sin_emb)

    return rotated_q_real, rotated_q_imag, rotated_k_real, rotated_k_imag

class iFairyAttention(nn.Module):
    
    def __init__(
        self,
        config: iFairyConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        attn_dropout_p: float = 0.0,
        quant_config: Optional["QuantizationConfig"] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
 
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = head_dim if head_dim is not None else hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_proj_real = QKVParallelLinear(
            hidden_size, 
            self.head_dim, 
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config, 
            prefix=add_prefix("qkv_proj_real", prefix)
        )
        self.qkv_proj_imag = QKVParallelLinear(
            hidden_size, 
            self.head_dim, 
            self.total_num_heads, 
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config, 
            prefix=add_prefix("qkv_proj_imag", prefix)
        )
        self.o_proj_real = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False, 
            quant_config=quant_config,
            prefix=add_prefix("o_proj_real", prefix)
        )
        self.o_proj_imag = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj_imag", prefix)
        )

        self.rotary_emb = ComplexNetRotaryEmbedding(
            rope_theta=rope_theta,
            hidden_size=self.hidden_size,
            num_attention_heads=self.total_num_heads,   # 注意用 total_num_heads
            max_position_embeddings=max_position_embeddings,
        )
        
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,        # [B, S]
        hidden_states: torch.Tensor,    # [B, S, hidden_size]
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

    # 1) QKV（实/虚）投影并切分
        qkv_real, _ = self.qkv_proj_real(hidden_states)  # [B, S, q_size + 2*kv_size]（按你的split）
        qkv_imag, _ = self.qkv_proj_imag(hidden_states)

        q_real, k_real, v_real = qkv_real.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_imag, k_imag, v_imag = qkv_imag.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2) reshape 成 [B, S, H, D] / [B, S, kvH, D]
        def to_heads(x, n_h):
            return x.view(B, S, n_h, self.head_dim)

        q_real = to_heads(q_real, self.num_heads)
        q_imag = to_heads(q_imag, self.num_heads)
        k_real = to_heads(k_real, self.num_kv_heads)
        k_imag = to_heads(k_imag, self.num_kv_heads)
        v_real = to_heads(v_real, self.num_kv_heads)
        v_imag = to_heads(v_imag, self.num_kv_heads)

    # 3) 计算 cos/sin 并对 Q/K 施加旋转
        cos_emb, sin_emb = self.rotary_emb(positions, hidden_states.dtype)  # [B,S,D]
    # _apply_rotary_pos_emb 会把 cos/sin 扩到 [B,1,S,D]，自动广播到 head 维
        q_r, q_i, k_r, k_i = _apply_rotary_pos_emb(
            q_real=q_r, q_imag=q_i, k_real=k_r, k_imag=k_i,
            cos_emb=cos_emb, sin_emb=sin_emb
        )
        
        attn_out_real, attn_out_imag = self.attn(q_r, q_i, k_r, k_i, v_r, v_i, forward_batch)
        attn_out_real = attn_out_real.reshape(B, S, self.num_heads * self.head_dim)
        attn_out_imag = attn_out_imag.reshape(B, S, self.num_heads * self.head_dim)
        o_real, _ = self.o_proj_real(attn_out_real)
        o_imag, _ = self.o_proj_imag(attn_out_imag)
        return o_real,o_imag
    
class iFairyDecoderLayer(nn.Module):
    def __init__(
        self,
        config: iFairyNetConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config =config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        
        self.self_attn = iFairyAttention(
            config= self.config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = iFairyMLP(
            config = self.config,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states_real: torch.Tensor,
        hidden_states_imag:torch.Tensor,
        forward_batch: ForwardBatch,
        residual_real: Optional[torch.Tensor],
        residual_imag:Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual_real is None:
            residual_real = hidden_states_real
            hidden_states_real = self.input_layernorm(hidden_states_real)
        else:
            hidden_states_real, residual_real = self.input_layernorm(hidden_states_real, residual_real)
        if residual_imag is None:
            residual_imag = hidden_states_imag
            hidden_states_imag = self.input_layernorm(hidden_states_imag)
        else:
            hidden_states_imag, residual_imag = self.input_layernorm(hidden_states_imag, residual_imag)
        hidden_states_real = self.self_attn(
            positions=positions,
            hidden_states_real=hidden_states_real,
            forward_batch=forward_batch,
        )
        hidden_states_imag = self.self_attn(
            positions=positions,
            hidden_states_imag=hidden_states_imag,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    
class iFairyModel(nn.Module):
    def __init__(
        self,
        config: iFairyConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = iFairyDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb
        else:
            return self.get_input_embeddings()(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states_real = self.embed_tokens(input_ids)
                hidden_states_imag = self.embed_tokens(input_ids)
            else:
                hidden_states_real = input_embeds
                hidden_states_imag = input_embeds
            residual_real = None
            residual_imag = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states_real = pp_proxy_tensors["hidden_states_real"]
            hidden_states_imag = pp_proxy_tensors["hidden_states_imag"]
            residual_real = pp_proxy_tensors["residual_real"]
            residual_imag = pp_proxy_tensors["residual_imag"]

        aux_hidden_states_real = []
        aux_hidden_states_imag = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states_real.append(
                    hidden_states_real + residual_real if residual_real is not None else hidden_states_real
                )
            layer = self.layers[i]
            hidden_states_real, residual_real = layer(
                positions,
                hidden_states_real,
                forward_batch,
                residual_real,
            )
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states_imag.append(
                    hidden_states_imag + residual_imag if residual_imag is not None else hidden_states_imag
                )
            layer = self.layers[i]
            hidden_states_imag, residual_imag = layer(
                positions,
                hidden_states_imag,
                forward_batch,
                residual_imag,
            )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states_real": hidden_states_real,
                    "hidden_states_imag": hidden_states_imag,
                    "residual_real": residual_real,
                    "residual_imag": residual_imag,
                }
            )
        else:
            if hidden_states_real.shape[0] != 0:
                if residual_real is None:
                    hidden_states_real = self.norm(hidden_states_real)
                else:
                    hidden_states_real, _ = self.norm(hidden_states_real, residual_real)
            if hidden_states_imag.shape[0] != 0:
                if residual_imag is None:
                    hidden_states_imag = self.norm(hidden_states_imag)
                else:
                    hidden_states_imag, _ = self.norm(hidden_states_imag, residual_imag)

        if len(aux_hidden_states_real and aux_hidden_states_imag) == 0:
            return hidden_states_real,hidden_states_imag

        return hidden_states_real, hidden_states_imag, aux_hidden_states_real, aux_hidden_states_imag

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_real_scale") and hasattr(layer_self_attn.attn, "k_imag_scale"):
                # 这里确保 k_real_scale 和 k_imag_scale 都存在
                layer_self_attn.attn.k_real_scale = scaling_factor
                layer_self_attn.attn.v_real_scale = scaling_factor
                layer_self_attn.attn.k_imag_scale = scaling_factor
                layer_self_attn.attn.v_imag_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling factor attributes (k_real_scale or k_imag_scale)!"
                )
    
class iFairyForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj_real.",
        ".gate_proj_imag.",
        ".down_proj_real.",
        ".down_proj_imag.",
        ".up_proj_real.",
        ".up_proj_imag.",
        ".q_proj_real.",
        ".q_proj_imag.",
        ".k_proj_real.",
        ".k_proj_imag.",
        ".v_proj_real.",
        ".v_proj_imag.",
        ".o_proj_real.",
        ".o_proj_imag.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj_real": ("qkv_proj_real", 0),
        "k_proj_real": ("qkv_proj_real", 1),
        "v_proj_real": ("qkv_proj_real", 2),
        "q_proj_imag": ("qkv_proj_imag", 0),
        "k_proj_imag": ("qkv_proj_imag", 1),
        "v_proj_imag": ("qkv_proj_imag", 2),
    }

    def __init__(
        self,
        config: BitNetConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = BitNetModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            else:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # For EAGLE3 support
        self.capture_aux_hidden_states_real = False
        self.capture_aux_hidden_states_imag = False

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states_real = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        hidden_states_imag = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states_real = None
        aux_hidden_states_imag = None
        if self.capture_aux_hidden_states_real and self.capture_aux_hidden_states_imag:
            aux_hidden_states_real = hidden_states_real
            aux_hidden_states_imag = hidden_states_imag

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states_real,
                    hidden_states_imag,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states_real,
                    aux_hidden_states_imag,
                )
            else:
                return self.pooler(hidden_states_real,hidden_states_imag, forward_batch)
        else:
            return hidden_states_real,hidden_states_imag

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            hidden_states_real, hidden_states_imag = forward_batch.hidden_states
            residual_real, residual_imag = forward_batch.residual

            hidden_states_real, hidden_states_imag, residual_real, residual_imag = layer(
                positions,
                hidden_states_real, hidden_states_imag,
                forward_batch,
                residual_real, residual_imag,
            )
            forward_batch.hidden_states = (hidden_states_real, hidden_states_imag)
            forward_batch.residual = (residual_real, residual_imag)

        if end == self.model.config.num_hidden_layers:
            # norm for real and imaginary parts
            hidden_states_real, hidden_states_imag = forward_batch.hidden_states
            residual_real, residual_imag = forward_batch.residual

            # Normalize real and imaginary parts
            hidden_states_real, hidden_states_imag = self.model.norm(
                hidden_states_real, hidden_states_imag, residual_real, residual_imag
            )

            forward_batch.hidden_states = (hidden_states_real, hidden_states_imag)

    # logits process for real and imaginary parts
            result = self.logits_processor(
                input_ids,
                hidden_states_real, hidden_states_imag,
                self.lm_head,
                forward_batch
            )
        else:
            result = None
            
        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj_real", "q_proj_real", "q"),
            ("qkv_proj_real", "k_proj_real", "k"),
            ("qkv_proj_real", "v_proj_real", "v"),
            ("qkv_proj_imag", "q_proj_imag", "q"),
            ("qkv_proj_imag", "k_proj_imag", "k"),
            ("qkv_proj_imag", "v_proj_imag", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:   #  iii
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)  # iii
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]



EntryClass = iFairyForCausalLM