from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.functional as F
import triton
from functools import partial

from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama_quik.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeightQuik
from lightllm.utils.infer_utils import mark_cost_time


class LlamaTransformerLayerInferQuik(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]

        self.inter_dim_ = network_config["intermediate_size"]
        self._bind_func()
        return

    def _get_qkv(
        self,
        input: torch.Tensor,
        cache_kv: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerWeightQuik,
    ) -> torch.Tensor:
        q = layer_weight.q_proj(input.view(-1, self.embed_dim_))
        if layer_weight.cat_kv_:
            cache_kv = layer_weight.kv_proj(input.view(-1, self.embed_dim_)).view(-1, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        else:
            cache_k = layer_weight.k_proj(input.view(-1, self.embed_dim_)).view(-1, self.tp_k_head_num_, self.head_dim_)
            cache_v = layer_weight.v_proj(input.view(-1, self.embed_dim_)).view(-1, self.tp_v_head_num_, self.head_dim_)
            cache_kv = torch.cat([cache_k, cache_v], dim=1)
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuik
    ) -> torch.Tensor:
        return layer_weight.o_proj(input.view(-1, self.embed_dim_))

    def _ffn(
        self,
        input,
        infer_state: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerWeightQuik,
    ) -> torch.Tensor:
        if not layer_weight.cat_gate_up_:
            gate_out = layer_weight.gate_proj(input.view(-1, self.embed_dim_))
            up_out = layer_weight.up_proj(input.view(-1, self.embed_dim_))
            torch.nn.functional.silu(gate_out, inplace=True)
            gate_out.mul_(up_out)
            input = None
            ffn2_out = layer_weight.down_proj(gate_out)
            gate_out, up_out = None, None
        else:
            gate_up_out = layer_weight.gate_up_proj(input.view(-1, self.embed_dim_)).view(-1, self.inter_dim_ * 2 // self.world_size_)
            # gate_out, up_out = torch.split(gate_up_out, split_size_or_sections=1, dim=1)
            ffn1_out = silu_and_mul_fwd(gate_up_out)
            input = None
            gate_up_out = None
            ffn2_out = layer_weight.down_proj(ffn1_out)
            ffn1_out = None

        return ffn2_out

    @mark_cost_time(
        "trans context flash forward time cost"
    )  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, cache_kv, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    @mark_cost_time(
        "trans context ffn forward time cost"
    )  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return

    # this impl dont to use @mark_cost_time
    def _token_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, cache_kv, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    # this impl dont to use @mark_cost_time
    def _token_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return