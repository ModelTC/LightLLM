import os
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from typing import Tuple
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from functools import partial
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.distributed.communication_op import all_gather_into_tensor

logger = init_logger(__name__)


class Qwen3MOETransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        self.n_routed_experts = network_config["num_experts"]
        self.is_moe = (
            network_config["num_experts"] > 0
            and layer_num not in network_config["mlp_only_layers"]
            and (layer_num + 1) % network_config["decoder_sparse_step"] == 0
        )
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.norm_topk_prob = network_config["norm_topk_prob"]
        super().__init__(layer_num, network_config, mode)
        self.head_dim_ = network_config["head_dim"]
        self.tp_k_head_num_ = max(self.tp_k_head_num_, 1)
        self.tp_v_head_num_ = max(self.tp_v_head_num_, 1)
        return

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_ffn(self):
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Qwen3MOETransformerLayerInfer._moe_ffn_edp, self)
            else:
                self._ffn = partial(Qwen3MOETransformerLayerInfer._moe_ffn, self)
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = self._pre_cache_kv(infer_state=infer_state, layer_weight=layer_weight)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        rmsnorm_forward(
            q.view(-1, self.head_dim_),
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _tpsp_get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input[0 : len(infer_state.position_cos), :]

        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = self._pre_cache_kv(infer_state=infer_state, layer_weight=layer_weight)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rmsnorm_forward(
            q.view(-1, self.head_dim_),
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _moe_ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape
        router_logits = layer_weight.moe_gate.mm(hidden_states)
        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
        )
        return hidden_states.view(num_tokens, hidden_dim)

    def _moe_ffn_edp(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input
        token_num, hidden_dim = hidden_states.shape

        router_logits = layer_weight.moe_gate.mm(hidden_states)
        ep_output = layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            is_prefill=infer_state.is_prefill,
        )

        ep_output = ep_output.view(token_num, hidden_dim)
        return ep_output

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_token_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._token_attention_kernel(_0_q, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)
        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 dispatch
        (
            _0_recv_x,
            _0_masked_m,
            _0_topk_idx,
            _0_topk_weight,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.low_latency_dispatch(_0_input1, _0_router_logits)
        infer_state.hook = _0_hook

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._token_attention_kernel(_1_q, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)
        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        # 1 dispatch
        (
            _1_recv_x,
            _1_masked_m,
            _1_topk_idx,
            _1_topk_weight,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.low_latency_dispatch(_1_input1, _1_router_logits)
        infer_state1.hook = _1_hook

        # moe calu
        expected_m = triton.cdiv(
            input_embdings.shape[0] * get_global_world_size() * self.num_experts_per_tok, self.n_routed_experts
        )
        _0_moe_out = layer_weight.experts.masked_group_gemm(_0_recv_x, _0_masked_m, input_embdings.dtype, expected_m)

        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 combine
        _0_ffn_out, _0_hook = layer_weight.experts.low_latency_combine(
            _0_moe_out, _0_topk_idx, _0_topk_weight, _0_handle
        )

        infer_state.hook = _0_hook

        # to do moe caclue
        _1_moe_out = layer_weight.experts.masked_group_gemm(_1_recv_x, _1_masked_m, input_embdings1.dtype, expected_m)

        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))
            infer_state.hook = None

        # 1 combine
        _1_ffn_out, _1_hook = layer_weight.experts.low_latency_combine(
            _1_moe_out, _1_topk_idx, _1_topk_weight, _1_handle
        )

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1

    def overlap_tpsp_context_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_context_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._context_attention_kernel(_0_q, _0_cache_kv, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)

        # wait last 1 combine
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_topk_weight, _0_topk_idx, _0_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _0_input1, _0_router_logits
        )
        from deep_ep import Buffer

        _0_overlap_event = Buffer.capture()

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._context_attention_kernel(_1_q, _1_cache_kv, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)

        # 0 dispatch execute
        (
            _0_recv_x,
            _0_recv_topk_idx,
            _0_recv_topk_weight,
            _0_num_recv_tokens_per_expert_list,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.dispatch(_0_qinput_tensor, _0_topk_idx, _0_topk_weight, overlap_event=_0_overlap_event)
        infer_state.hook = _0_hook

        # wait 0 dispatch
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_topk_weight, _1_topk_idx, _1_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _1_input1, _1_router_logits
        )

        _1_overlap_event = Buffer.capture()

        # 0 moe calu
        _0_moe_out = layer_weight.experts.prefilled_group_gemm(
            _0_num_recv_tokens_per_expert_list, _0_recv_x, _0_recv_topk_idx, _0_recv_topk_weight
        )

        # 1 dispatch execute
        (
            _1_recv_x,
            _1_recv_topk_idx,
            _1_recv_topk_weight,
            _1_num_recv_tokens_per_expert_list,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.dispatch(_1_qinput_tensor, _1_topk_idx, _1_topk_weight, overlap_event=_1_overlap_event)
        infer_state1.hook = _1_hook

        # wait 1 dispatch
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_combine_event = Buffer.capture()
        # 0 combine execute
        _0_ffn_out, _0_hook = layer_weight.experts.combine(_0_moe_out, _0_handle, _0_combine_event)
        infer_state.hook = _0_hook

        # 1 moe calc
        _1_moe_out = layer_weight.experts.prefilled_group_gemm(
            _1_num_recv_tokens_per_expert_list, _1_recv_x, _1_recv_topk_idx, _1_recv_topk_weight
        )

        # wait 0 combine
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_combine_event = Buffer.capture()

        input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))

        # 1 combine execute
        _1_ffn_out, _1_hook = layer_weight.experts.combine(_1_moe_out, _1_handle, _1_combine_event)

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1
