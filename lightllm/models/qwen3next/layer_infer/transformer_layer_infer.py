import os
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from typing import Tuple
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import Qwen3NextTransformerLayerWeight
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from functools import partial
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen3next.mem_manager import Qwen3NextMemoryManager
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor
from typing_extensions import override
from einops import rearrange
from lightllm.models.qwen3next.triton_kernel.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.models.qwen3next.triton_kernel.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
from lightllm.models.qwen3next.triton_kernel.fla.ops.chunk import chunk_gated_delta_rule
from lightllm.models.qwen3next.triton_kernel.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule
from lightllm.distributed import all_reduce


logger = init_logger(__name__)


class Qwen3NextTransformerLayerInfer(Qwen3MOETransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.is_linear = (layer_num + 1) % network_config["full_attention_interval"] != 0
        if self.is_linear:
            self.linear_attn_infer = Qwen3NextGatedDeltaNetInfer(network_config, layer_num, self.tp_world_size_)

        return

    @override
    def rmsnorm(self, input, weight, out: torch.Tensor):
        # Zero-Centered RMSNorm TODO trion op
        input_float32 = self.alloc_tensor(input.shape, torch.float32, device=input.device)
        input_float32.copy_(input)
        input_float32 = input_float32 * torch.rsqrt(input_float32.pow(2).mean(-1, keepdim=True) + self.eps_)
        input_float32 = input_float32 * (1.0 + weight.float())
        out.copy_(input_float32.to(input.dtype))
        return out

    @override
    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ) -> torch.Tensor:
        # TODO fuse it
        input = input.view(-1, self.tp_o_head_num_, self.head_dim_)
        input = input * layer_weight._gate
        layer_weight._gate = None
        input = input.reshape(-1, self.tp_o_head_num_ * self.head_dim_)
        o_tensor = layer_weight.o_proj.mm(input)
        return o_tensor

    def context_attention_forward(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        q, cache_kv = self._get_qkv(input, infer_state, layer_weight)
        input = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return o

    def context_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        if self.is_linear:
            o = self.linear_attn_infer._linear_attn(input1, infer_state, layer_weight, is_prefill=True, infer_cls=self)
        else:
            layer_weight._gate = torch.sigmoid(layer_weight.o_gate_proj.mm(input1)).view(
                -1, self.tp_o_head_num_, self.head_dim_
            )
            o = self.context_attention_forward(input1, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_attention_forward(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        q, cache_kv = self._get_qkv(input, infer_state, layer_weight)
        input = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return o

    def token_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        if self.is_linear:
            o = self.linear_attn_infer._linear_attn(input1, infer_state, layer_weight, is_prefill=False, infer_cls=self)
        else:
            layer_weight._gate = torch.sigmoid(layer_weight.o_gate_proj.mm(input1)).view(
                -1, self.tp_o_head_num_, self.head_dim_
            )
            o = self.token_attention_forward(input1, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings


class Qwen3NextGatedDeltaNetInfer:
    def __init__(self, network_config, layer_idx, tp_world_size_):
        self.network_config_ = network_config
        self.layer_idx_ = layer_idx
        self.tp_world_size_ = tp_world_size_
        self.num_v_heads = self.network_config_["linear_num_value_heads"]
        self.num_k_heads = self.network_config_["linear_num_key_heads"]
        self.head_k_dim = self.network_config_["linear_key_head_dim"]
        self.head_v_dim = self.network_config_["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_dim = self.network_config_["linear_conv_kernel_dim"]
        self.activation = self.network_config_["hidden_act"]
        self.tp_qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) // self.tp_world_size_
        self.tp_ba_dim = (self.num_v_heads * 2) // self.tp_world_size_
        self.tp_num_k_heads = self.num_k_heads // self.tp_world_size_
        self.tp_num_v_heads = self.num_v_heads // self.tp_world_size_
        self.tp_key_dim = self.key_dim // self.tp_world_size_
        self.tp_value_dim = self.value_dim // self.tp_world_size_
        assert self.num_v_heads % self.num_k_heads == 0, "num_v_heads must be divisible by num_k_heads"
        self.num_v_heads_per_k_head = self.num_v_heads // self.num_k_heads

    def _fix_query_key_value_ba_ordering(self, mixed_qkvzba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        mixed_qkvz, mixed_ba = torch.split(mixed_qkvzba, [self.tp_qkvz_dim, self.tp_ba_dim], dim=-1)

        mixed_qkvz = mixed_qkvz.view(
            -1,
            self.tp_num_k_heads,
            self.head_k_dim + self.head_k_dim + (self.head_v_dim + self.head_v_dim) * self.num_v_heads_per_k_head,
        )
        mixed_ba = mixed_ba.view(-1, self.tp_num_k_heads, 2 * self.num_v_heads_per_k_head)

        qkvz_split_list = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads_per_k_head * self.head_v_dim),
            (self.num_v_heads_per_k_head * self.head_v_dim),
        ]
        (query, key, value, z) = torch.split(mixed_qkvz, qkvz_split_list, dim=2)
        (b, a) = torch.split(mixed_ba, [self.num_v_heads_per_k_head, self.num_v_heads_per_k_head], dim=2)

        query = query.reshape(-1, self.tp_num_k_heads * self.head_k_dim)
        key = key.reshape(-1, self.tp_num_k_heads * self.head_k_dim)
        value = value.reshape(-1, self.tp_num_v_heads * self.head_v_dim)
        z = z.reshape(-1, self.tp_num_v_heads, self.head_v_dim)
        b = b.reshape(-1, self.tp_num_v_heads)
        a = a.reshape(-1, self.tp_num_v_heads)

        return query, key, value, z, b, a

    def _rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim],
            dim=-1,
        )
        query, key = map(lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim), (query, key))
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query, key, value

    def _linear_attn(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
        is_prefill: bool,
        infer_cls: Qwen3NextTransformerLayerInfer,
    ):
        assert layer_weight.is_linear, "layer_weight must be linear"
        assert isinstance(infer_state.mem_manager, Qwen3NextMemoryManager)
        input = input.view(-1, infer_cls.embed_dim_)

        # Get conv_states and ssm_states buffer
        conv_states, ssm_states = infer_state.mem_manager.get_mamba_state_buffer(self.layer_idx_)

        # Project input to qkvzba
        mixed_qkvzba = layer_weight.linear_in_proj.mm(
            input
        )  # tgt: [batch_size, (self.key_dim * 2 + self.value_dim * 2) + (self.num_v_heads * 2)]
        q, k, v, z, b, a = self._fix_query_key_value_ba_ordering(mixed_qkvzba)
        mixed_qkv = torch.cat([q, k, v], dim=-1)  # tgt: [batch_size, tp_qkv_dim]

        # Convolution: different paths for prefill and decode
        if is_prefill:
            # Prefill: use causal_conv1d_fn for full sequence processing
            mixed_qkv = mixed_qkv.transpose(0, 1)  # [tp_qkv_dim, seq_len]
            out_tensor = infer_cls.alloc_tensor(mixed_qkv.shape, mixed_qkv.dtype, device=mixed_qkv.device)
            causal_conv1d_fn(
                mixed_qkv,
                layer_weight.linear_conv1d.weight.transpose(0, 1),
                layer_weight.linear_conv1d.bias,
                conv_states.transpose(1, 2),
                infer_state.b1_cu_q_seq_len,
                out=out_tensor,
                cache_indices=infer_state.b_req_idx,
                activation=self.activation,  # 添加 activation 参数
            )
            mixed_qkv = out_tensor.transpose(0, 1)  # [seq_len, tp_qkv_dim]
        else:
            # Decode: use causal_conv1d_update for single token update
            # Need to transpose conv_states to match expected format: (..., dim, state_len)
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states.transpose(1, 2),
                layer_weight.linear_conv1d.weight.transpose(0, 1),
                layer_weight.linear_conv1d.bias,
                self.activation,
                conv_state_indices=infer_state.b_req_idx,
                validate_data=True,
            )

        # Rearrange mixed_qkv to query, key, value
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv)

        # Compute beta and g
        beta = b.sigmoid()
        g = fused_gdn_gating(layer_weight.linear_A_log.weight, a, layer_weight.linear_dt_bias.weight)
        g, beta = map(lambda x: rearrange(x, "l d -> 1 l d"), (g, beta))

        # Recurrent attention: different paths for prefill and decode
        if is_prefill:
            # Prefill: use chunk_gated_delta_rule
            # Get initial state and clear it for new requests (no prompt cache support yet)
            initial_state = ssm_states[infer_state.b_req_idx].contiguous()
            initial_state[...] = 0  # Clear initial state for all requests
            (core_attn_out, last_recurrent_state,) = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=infer_state.b1_cu_q_seq_len,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            # Update SSM state with final state
            ssm_states[infer_state.b_req_idx, ...] = last_recurrent_state.to(ssm_states.dtype)
        else:
            # Decode: use fused_recurrent_gated_delta_rule for single token
            batch_size = input.shape[0]
            cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=input.device)
            (core_attn_out, last_recurrent_state,) = fused_recurrent_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=ssm_states,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=infer_state.b_req_idx,
                use_qk_l2norm_in_kernel=True,
            )

        # Gated RMSNorm and output projection
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        norm_out = infer_cls.alloc_tensor(core_attn_out.shape, core_attn_out.dtype, device=core_attn_out.device)
        gated_rmsnorm_forward(
            core_attn_out,
            layer_weight.linear_norm.weight,
            layer_weight.linear_norm.bias,
            infer_cls.eps_,
            z,
            out=norm_out,
        )
        core_attn_out = norm_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")

        output = layer_weight.linear_out_proj.mm(core_attn_out)
        if infer_cls.tp_world_size_ > 1:
            all_reduce(output, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        return output
