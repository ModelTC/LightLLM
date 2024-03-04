from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.functional as F
import triton
from functools import partial

from lightllm.models.llama_awquant.layer_weights.transformer_layer_weight import (
    LlamaTransformerLayerActivationWeightQuantized,
)
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.common.basemodel import TransformerLayerInferActivationWeightQuantTpl
from lightllm.common.basemodel.cuda_kernel.ppl_awquant import (
    matmul_i8_i32_ppl,
    skiprmsnorm_ppl,
    channel_token_dequant_i32_fp16_ppl,
)
from lightllm.common.basemodel.cuda_kernel.ppl_awquant import dynamic_channelwise_quant_fp16_i8_ppl, gatesilu_i32_i8_ppl
from lightllm.utils.infer_utils import mark_cost_time


class LlamaTransformerLayerInferAWquant(TransformerLayerInferActivationWeightQuantTpl):
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

    def _bind_func(self):
        self._bind_norm()
        self._bind_matmul()
        self._bind_silu()
        LlamaTransformerLayerInfer._bind_attention(self)
        return

    def _bind_norm(self):
        if "ppl_w8a8" in self.mode:
            self._awquant_att_norm = partial(LlamaTransformerLayerInferAWquant._awquant_att_norm_ppl_int8, self)
            self._awquant_ffn_norm = partial(LlamaTransformerLayerInferAWquant._awquant_ffn_norm_ppl_int8, self)
        else:
            raise Exception(f"error mode {self.mode}")
        return

    def _bind_matmul(self):
        if "ppl_w8a8" in self.mode:
            self._awquant_matmul_for_qkv = partial(
                LlamaTransformerLayerInferAWquant._awquant_matmul_ppl_int8_quant_dequant, self
            )
            self._awquant_matmul_for_o = partial(
                LlamaTransformerLayerInferAWquant._awquant_matmul_ppl_int8_quant_dequant, self
            )
            self._awquant_matmul_for_ffn_up = partial(
                LlamaTransformerLayerInferAWquant._awquant_matmul_ppl_int8_quant, self
            )
            self._awquant_matmul_for_ffn_down = partial(
                LlamaTransformerLayerInferAWquant._awquant_matmul_ppl_int8_quant_dequant, self
            )
            if self.tp_rank_ == 0 and self.layer_num_ == 0:
                print("model use ppl_w8a8 kernel")
        else:
            raise Exception(f"error mode {self.mode}")
        return

    def _bind_silu(self):
        if "ppl_w8a8" in self.mode:
            func = partial(LlamaTransformerLayerInferAWquant._awquant_silu_ppl_int8, self)
            self._awquant_silu = func
        else:
            raise Exception(f"error mode {self.mode}")
        return

    def _get_qkv(
        self,
        input,
        cache_kv,
        token_scale,
        infer_state: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerActivationWeightQuantized,
    ) -> torch.Tensor:
        q = self._awquant_matmul_for_qkv(
            input.view(-1, self.embed_dim_),
            quant_weight_params=layer_weight.q_weight_,
            is_prefill=infer_state.is_prefill,
            token_scale=token_scale,
        )
        cache_kv = self._awquant_matmul_for_qkv(
            input.view(-1, self.embed_dim_),
            quant_weight_params=layer_weight.kv_weight_,
            is_prefill=infer_state.is_prefill,
            token_scale=token_scale,
        ).view(-1, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        out = self._awquant_matmul_for_qkv(
            input.view(-1, self.embed_dim_),
            quant_weight_params=layer_weight.v_weight_,
            is_prefill=infer_state.is_prefill,
            token_scale=token_scale,
        )
        cache_kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :] = out.view(
            -1, self.tp_v_head_num_, self.head_dim_
        )
        return q, cache_kv

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerActivationWeightQuantized
    ) -> torch.Tensor:
        o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        return o_tensor

    def _ffn(
        self,
        input,
        token_scale,
        infer_state: LlamaInferStateInfo,
        layer_weight: LlamaTransformerLayerActivationWeightQuantized,
    ) -> torch.Tensor:
        gate_out = self._awquant_matmul_for_ffn_up(
            input.view(-1, self.embed_dim_),
            layer_weight.gate_proj,
            is_prefill=infer_state.is_prefill,
        )
        up_out = self._awquant_matmul_for_ffn_up(
            input.view(-1, self.embed_dim_),
            layer_weight.up_proj,
            is_prefill=infer_state.is_prefill,
        )
        input = None
        _, gate_proj_scale = layer_weight.gate_proj
        _, up_proj_scale = layer_weight.up_proj
        ffn1_out, ffn1_out_scale = self._awquant_silu(gate_out, up_out, gate_proj_scale, up_proj_scale, token_scale)
        gate_out, up_out = None, None
        ffn2_out = self._awquant_matmul_for_ffn_down(
            ffn1_out, layer_weight.down_proj, is_prefill=infer_state.is_prefill, token_scale=ffn1_out_scale
        )
        ffn1_out = None

        return ffn2_out

    @mark_cost_time(
        "trans context flash forward time cost"
    )  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1, token_scale, skip_out = self._awquant_att_norm(input_embding, infer_state, layer_weight)
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, cache_kv, token_scale, infer_state, layer_weight)
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
        input1, token_scale, skip_out = self._awquant_ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, token_scale, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return

    # this impl dont to use @mark_cost_time
    def _token_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1, token_scale, skip_out = self._awquant_att_norm(input_embding, infer_state, layer_weight)
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, cache_kv, token_scale, infer_state, layer_weight)
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
        input1, token_scale, skip_out = self._awquant_ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, token_scale, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return

    def _awquant_matmul_ppl_int8_quant_dequant(
        self, input, quant_weight_params, is_prefill, token_scale=None, out=None, bias=None, has_act=False
    ):
        if input.dtype == torch.float16:
            input, token_scale = dynamic_channelwise_quant_fp16_i8_ppl(input.transpose(0, 1))
        assert has_act is False
        if is_prefill:
            qweight, qscale = quant_weight_params
            out = matmul_i8_i32_ppl(input, qweight)
        else:
            qweight, qscale = quant_weight_params
            out = matmul_i8_i32_ppl(input, qweight)
        out = channel_token_dequant_i32_fp16_ppl(out, token_scale, qscale)
        if bias is None:
            return out
        else:
            out.add_(bias)
            return out

    def _awquant_matmul_ppl_int8_quant(
        self, input, quant_weight_params, is_prefill, out=None, bias=None, has_act=False
    ):
        assert has_act is False
        if is_prefill:
            qweight, qscale = quant_weight_params
            out = matmul_i8_i32_ppl(input, qweight)
        else:
            qweight, qscale = quant_weight_params
            out = matmul_i8_i32_ppl(input, qweight)
        if bias is None:
            return out
        else:
            out.add_(bias)
            return out

    def _awquant_att_norm_ppl_int8(self, input, infer_state: LlamaInferStateInfo, layer_weight):
        if getattr(infer_state, "skip", None) is None:
            infer_state.skip = torch.zeros_like(input)
        return skiprmsnorm_ppl(input, layer_weight.att_norm_weight_, skip=infer_state.skip)

    def _awquant_ffn_norm_ppl_int8(self, input, infer_state: LlamaInferStateInfo, layer_weight):
        return skiprmsnorm_ppl(input, layer_weight.ffn_norm_weight_, skip=infer_state.skip)

    def _awquant_silu_ppl_int8(self, x, y, x_scale, y_scale, token_scale):
        return gatesilu_i32_i8_ppl(x, y, x_scale, y_scale, token_scale)
