import os
import torch
import torch.distributed as dist
import triton
from typing import Tuple
from functools import partial

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.glm4_moe.layer_weights.transformer_layer_weight import Glm4MoeTransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen3.triton_kernel.qk_norm import qk_rmsnorm_forward
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor
from lightllm.utils.dist_utils import get_global_world_size


class Glm4MoeTransformerLayerInfer(LlamaTransformerLayerInfer):
    """
    GLM-4.7 MoE Transformer Layer Inference.

    Key features:
    - QK normalization (RMSNorm on Q and K projections)
    - Partial rotary embeddings (0.5 factor)
    - MoE FFN with 160 routed + 1 shared expert, top-8 routing
    - Sigmoid gating with e_score_correction_bias
    - routed_scaling_factor = 2.5 applied to expert outputs
    - First 3 layers use dense FFN (first_k_dense_replace=3)
    """

    def __init__(self, layer_num, network_config):
        # Parse MoE config before calling super().__init__
        self.n_routed_experts = network_config.get("n_routed_experts", 160)
        self.n_shared_experts = network_config.get("n_shared_experts", 1)
        first_k_dense_replace = network_config.get("first_k_dense_replace", 3)

        self.is_moe = self.n_routed_experts is not None and layer_num >= first_k_dense_replace

        # MoE routing parameters
        self.num_experts_per_tok = network_config.get("num_experts_per_tok", 8)
        self.norm_topk_prob = network_config.get("norm_topk_prob", True)
        self.n_group = network_config.get("n_group", None)  # GLM-4.7 may not use grouped routing
        self.topk_group = network_config.get("topk_group", None)
        self.routed_scaling_factor = network_config.get("routed_scaling_factor", 2.5)

        # Partial rotary factor
        self.partial_rotary_factor = network_config.get("partial_rotary_factor", 0.5)

        super().__init__(layer_num, network_config)

        # Override head_dim for GLM-4.7
        self.head_dim_ = network_config.get("head_dim", 128)
        # Ensure at least 1 KV head per TP rank for GQA
        self.tp_k_head_num_ = max(self.tp_k_head_num_, 1)
        self.tp_v_head_num_ = max(self.tp_v_head_num_, 1)
        return

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_ffn(self):
        """Bind FFN function based on layer type (MoE vs dense) and parallelism mode."""
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Glm4MoeTransformerLayerInfer._moe_ffn_ep, self)
                self._tpsp_ffn = self._tpsp_ffn_ep
            else:
                self._ffn = partial(Glm4MoeTransformerLayerInfer._moe_ffn, self)
                self._tpsp_ffn = self._tpsp_ffn_tp
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)
            self._tpsp_ffn = self._tpsp_ffn_tp
        return

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Glm4MoeTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V projections with QK normalization and partial RoPE.

        GLM-4.7 specific:
        - Applies RMSNorm to Q and K projections (QK norm)
        - Applies partial rotary embeddings (0.5 factor = half of head dim)
        """
        input = input.view(-1, self.embed_dim_)

        # Compute Q and KV projections
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input)

        # Apply QK normalization (RMSNorm on Q and K)
        qk_rmsnorm_forward(
            q,
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
        )
        qk_rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        )

        # Reshape cache_kv for attention
        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        # Apply partial rotary embeddings (0.5 factor)
        # GLM-4.7 uses partial_rotary_factor=0.5, applying RoPE to half of head dimensions
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            partial_rotary_factor=self.partial_rotary_factor,
        )

        return q, cache_kv

    def _tpsp_get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Glm4MoeTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TPSP mode QKV computation with all-gather for sequence parallelism."""
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input[0 : len(infer_state.input_ids), :]

        input = input.view(-1, self.embed_dim_)

        # Compute Q and KV projections
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input)

        # Apply QK normalization
        qk_rmsnorm_forward(
            q,
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
        )
        qk_rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        )

        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        # Apply partial rotary embeddings (0.5 factor for GLM-4.7)
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            partial_rotary_factor=self.partial_rotary_factor,
        )

        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)

        return q, cache_kv

    def _moe_ffn(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Glm4MoeTransformerLayerWeight,
    ) -> torch.Tensor:
        """
        MoE FFN forward pass for TP mode.

        GLM-4.7 uses:
        - Sigmoid gating with e_score_correction_bias
        - Top-8 routing from 160 experts
        - routed_scaling_factor = 2.5 applied to routed output
        - Shared expert output added to routed output
        """
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        # Compute shared expert output if not fused and shared experts exist
        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        # Compute router logits
        router_logits = layer_weight.moe_gate.mm(hidden_states)

        # Run MoE experts (handles routing, dispatch, compute, combine internally)
        # The experts.experts() method modifies hidden_states in-place
        # Note: routed_scaling_factor is already applied inside experts() to topk_weights
        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group is not None and self.n_group > 0,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
        )

        # Add shared expert output if computed separately
        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            hidden_states.add_(shared_output)

        return hidden_states.view(num_tokens, hidden_dim)

    def _moe_ffn_ep(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Glm4MoeTransformerLayerWeight,
    ) -> torch.Tensor:
        """MoE FFN forward pass for EP (Expert Parallelism) mode."""
        hidden_states = input
        token_num, hidden_dim = hidden_states.shape

        # Compute shared expert output if exists
        if self.n_shared_experts is not None:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        # Compute router logits
        router_logits = layer_weight.moe_gate.mm(hidden_states)

        # Run MoE experts in EP mode
        # Note: routed_scaling_factor is already applied inside experts() to topk_weights
        ep_output = layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group is not None and self.n_group > 0,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
            is_prefill=infer_state.is_prefill,
        )

        # Add shared expert output if exists
        if self.n_shared_experts is not None:
            ep_output.add_(shared_output)

        return ep_output.view(token_num, hidden_dim)

    def _tpsp_ffn(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Glm4MoeTransformerLayerWeight,
    ):
        """Placeholder - will be bound to actual implementation."""
        raise Exception("_tpsp_ffn needs to be bound to real implementation")

    def _tpsp_ffn_tp(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Glm4MoeTransformerLayerWeight,
    ) -> torch.Tensor:
        """TPSP FFN for TP mode with all-gather/reduce-scatter for sequence parallelism."""
        input = input.view(-1, self.embed_dim_)

        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input

        ffn_out = self._ffn(input=input, infer_state=infer_state, layer_weight=layer_weight)

        if self.tp_world_size_ > 1:
            sp_token_num = ffn_out.shape[0] // self.tp_world_size_
            reduce_o_tensor = self.alloc_tensor(
                (sp_token_num, self.embed_dim_), dtype=ffn_out.dtype, device=ffn_out.device
            )
            reduce_scatter_tensor(
                reduce_o_tensor, ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False
            )
            ffn_out = reduce_o_tensor

        return ffn_out

    def _tpsp_ffn_ep(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Glm4MoeTransformerLayerWeight,
    ) -> torch.Tensor:
        """TPSP FFN for EP mode."""
        input = input.view(-1, self.embed_dim_)
        ffn_out = self._ffn(input=input, infer_state=infer_state, layer_weight=layer_weight)
        return ffn_out

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: Glm4MoeTransformerLayerWeight,
    ):
        """Overlapped TPSP token forward for decode phase with MoE."""
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

        # 0 shared expert
        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            _0_shared_output = LlamaTransformerLayerInfer._ffn(self, _0_input1, infer_state, layer_weight)

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

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)

        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        # 1 shared expert
        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            _1_shared_output = LlamaTransformerLayerInfer._ffn(self, _1_input1, infer_state1, layer_weight)

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

        # moe compute
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

        # moe compute for batch 1
        _1_moe_out = layer_weight.experts.masked_group_gemm(_1_recv_x, _1_masked_m, input_embdings1.dtype, expected_m)

        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
                _0_ffn_out.add_(_0_shared_output)
            input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))
            infer_state.hook = None

        # 1 combine
        _1_ffn_out, _1_hook = layer_weight.experts.low_latency_combine(
            _1_moe_out, _1_topk_idx, _1_topk_weight, _1_handle
        )

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
                _1_ffn_out.add_(_1_shared_output)
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
        layer_weight: Glm4MoeTransformerLayerWeight,
    ):
        """Overlapped TPSP context forward for prefill phase with MoE."""
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

        # 0 shared expert
        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            _0_shared_output = LlamaTransformerLayerInfer._ffn(self, _0_input1, infer_state, layer_weight)

        # 1 shared expert
        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            _1_shared_output = LlamaTransformerLayerInfer._ffn(self, _1_input1, infer_state1, layer_weight)

        # 0 moe compute
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

        # 1 moe compute
        _1_moe_out = layer_weight.experts.prefilled_group_gemm(
            _1_num_recv_tokens_per_expert_list, _1_recv_x, _1_recv_topk_idx, _1_recv_topk_weight
        )

        # wait 0 combine
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_combine_event = Buffer.capture()

        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            _0_ffn_out.add_(_0_shared_output)
        input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))

        # 1 combine execute
        _1_ffn_out, _1_hook = layer_weight.experts.combine(_1_moe_out, _1_handle, _1_combine_event)

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
                _1_ffn_out.add_(_1_shared_output)
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1
