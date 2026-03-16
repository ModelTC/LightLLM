import os
import torch

import torch.distributed as dist
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextTransformerLayerWeight,
)
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_template import TransformerLayerInferTpl
from lightllm.utils.log_utils import init_logger
from lightllm.models.qwen3next.mem_manager import Qwen3NextHybridMemManager
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from typing import Tuple
from lightllm.models.qwen3next.triton_kernel.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.models.qwen3next.triton_kernel.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
from lightllm.models.qwen3next.triton_kernel.fla.ops import chunk_gated_delta_rule
from lightllm.models.qwen3next.triton_kernel.fla.ops import fused_recurrent_gated_delta_rule
from lightllm.models.qwen3next.triton_kernel.gdn_decode_mtp import (
    copy_conv_states,
    copy_ssm_states,
    copy_states_fused,
)
from lightllm.distributed import all_reduce
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
from functools import partial

logger = init_logger(__name__)


class Qwen3NextFullAttentionBaseLayerInfer(LlamaTransformerLayerInfer):
    """
    Base class for Qwen3Next full attention layers.
    Contains shared logic for both standard full attention and MTP layers.
    """

    def __init__(self, layer_num, network_config):
        self.partial_rotary_factor = network_config.get("partial_rotary_factor", 1.0)
        self.n_routed_experts = network_config.get("num_experts", 0)
        self.is_moe = (
            network_config.get("num_experts", 0) > 0
            and layer_num not in network_config.get("mlp_only_layers", [])
            and (layer_num + 1) % network_config.get("decoder_sparse_step", 1) == 0
        )
        self.num_experts_per_tok = network_config.get("num_experts_per_tok", 1)
        self.norm_topk_prob = network_config.get("norm_topk_prob", False)

        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config.get(
            "head_dim", network_config["hidden_size"] // network_config["num_attention_heads"]
        )
        return

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_ffn(self):
        """Bind FFN implementation based on MoE configuration."""
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._moe_ffn_edp, self)
            else:
                self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._moe_ffn, self)
        else:
            self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._ffn, self)
        return

    def _compute_shared_expert(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        input = input.view(-1, self.embed_dim_)
        shared_expert_out = super()._ffn(input, infer_state, layer_weight)
        gate = layer_weight.ffn_gate.mm(input).sigmoid_()
        shared_expert_out.mul_(gate)
        return shared_expert_out

    def _moe_ffn(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        """MoE FFN with tensor parallelism."""

        shared_expert_out = self._compute_shared_expert(input, infer_state, layer_weight)

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
        hidden_states = hidden_states.view(num_tokens, hidden_dim)
        hidden_states.add_(shared_expert_out)
        return hidden_states

    def _moe_ffn_edp(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        """MoE FFN with expert parallelism."""
        shared_expert_out = self._compute_shared_expert(input, infer_state, layer_weight)
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
        ep_output.add_(shared_expert_out)
        return ep_output

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        QKV projection with output gating, Q/K normalization, and partial rotary embedding.
        """
        input = input.view(-1, self.embed_dim_)
        qkv_out = layer_weight.qkv_proj.mm(input)
        q, cache_kv = qkv_out.split(
            [self.tp_q_head_num_ * self.head_dim_ * 2, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_],
            dim=-1,
        )
        o_gate = layer_weight._o_gate_proj.mm(input)
        # In-place sigmoid saves one allocation (gate_value is consumed once in _get_o)
        infer_state.gate_value = o_gate.sigmoid_()
        layer_weight.qk_norm_weight_(
            q,
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )
        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        return q, cache_kv

    def _get_o(
        self,
        input,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ) -> torch.Tensor:
        """Output projection with gating (in-place multiply to save one allocation)."""
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        input.mul_(infer_state.gate_value)
        infer_state.gate_value = None
        o_tensor = layer_weight.o_proj.mm(input)
        return o_tensor


class Qwen3NextFullAttentionTransformerLayerInfer(Qwen3NextFullAttentionBaseLayerInfer):
    """
    Full attention layer for Qwen3Next that uses the abstracted attention backend.
    Inherits from Qwen3NextFullAttentionBaseLayerInfer to get shared Qwen3Next logic.
    """

    pass


class Qwen3NextGatedDeltaNetTransformerLayerInfer(LlamaTransformerLayerInfer):
    """
    Linear attention (Gated Delta Networks) layer for Qwen3Next.
    """

    def __init__(self, layer_num, network_config):
        self.n_routed_experts = network_config.get("num_experts", 0)
        self.is_moe = (
            network_config.get("num_experts", 0) > 0
            and layer_num not in network_config.get("mlp_only_layers", [])
            and (layer_num + 1) % network_config.get("decoder_sparse_step", 1) == 0
        )
        super().__init__(layer_num, network_config)
        # MoE configuration
        self.num_experts_per_tok = network_config.get("num_experts_per_tok", 1)
        self.norm_topk_prob = network_config.get("norm_topk_prob", False)

        # Linear attention specific dimensions
        self.num_v_heads = network_config["linear_num_value_heads"]
        self.num_k_heads = network_config["linear_num_key_heads"]
        self.head_k_dim = network_config["linear_key_head_dim"]
        self.head_v_dim = network_config["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_dim = network_config["linear_conv_kernel_dim"]
        self.activation = network_config["hidden_act"]

        # Tensor parallelism dimensions
        self.tp_qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) // self.tp_world_size_
        self.tp_ba_dim = (self.num_v_heads * 2) // self.tp_world_size_
        self.tp_num_k_heads = self.num_k_heads // self.tp_world_size_
        self.tp_num_v_heads = self.num_v_heads // self.tp_world_size_
        self.tp_key_dim = self.key_dim // self.tp_world_size_
        self.tp_value_dim = self.value_dim // self.tp_world_size_

        # Template required dimensions (not used for GDN but required by interface)
        self.tp_q_head_num_ = self.tp_num_k_heads
        self.tp_k_head_num_ = self.tp_num_k_heads
        self.tp_v_head_num_ = self.tp_num_v_heads
        self.tp_o_head_num_ = self.tp_num_v_heads
        self.head_dim_ = self.head_v_dim

        assert self.num_v_heads % self.num_k_heads == 0, "num_v_heads must be divisible by num_k_heads"
        self.num_v_heads_per_k_head = self.num_v_heads // self.num_k_heads

        # MTP configuration
        self.mtp_step = get_env_start_args().mtp_step
        self.mtp_size = self.mtp_step + 1

        # SSM state dtype optimization
        ssm_dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        start_args = get_env_start_args()
        self.ssm_state_dtype = ssm_dtype_dict.get(start_args.mamba_ssm_data_type, torch.bfloat16)

        # Pre-compute whether dtype conversion is needed
        # GDN kernel output dtype is self.data_type
        # Conversion needed only if SSM state uses different dtype
        self.needs_ssm_dtype_conversion = get_llm_data_type() != self.ssm_state_dtype
        self._bind_func()
        return

    def _bind_func(self):
        """Bind layer-specific implementations"""
        self._bind_ffn()
        return

    def _bind_ffn(self):
        """Bind FFN implementation based on MoE configuration."""
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._moe_ffn_edp, self)
            else:
                self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._moe_ffn, self)
        else:
            self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._ffn, self)
        return

    def _compute_shared_expert(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        input = input.view(-1, self.embed_dim_)
        shared_expert_out = super()._ffn(input, infer_state, layer_weight)
        gate = layer_weight.ffn_gate.mm(input).sigmoid_()
        shared_expert_out.mul_(gate)
        return shared_expert_out

    def _moe_ffn(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        """MoE FFN with tensor parallelism."""

        shared_expert_out = self._compute_shared_expert(input, infer_state, layer_weight)

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
        hidden_states = hidden_states.view(num_tokens, hidden_dim)
        hidden_states.add_(shared_expert_out)
        return hidden_states

    def _moe_ffn_edp(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        """MoE FFN with expert parallelism."""
        shared_expert_out = self._compute_shared_expert(input, infer_state, layer_weight)
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
        ep_output.add_(shared_expert_out)
        return ep_output

    def _gdn_layer_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
        is_prefill: bool,
    ):
        """Unified forward for both prefill and decode in GDN layers."""
        # Attention + GDN processing
        input1 = layer_weight.att_norm_weight_(input=input_embdings, eps=self.eps_, alloc_func=self.alloc_tensor)
        gdn_out = self.gdn_forward(input1, infer_state, layer_weight, is_prefill=is_prefill)
        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)

        # FFN
        input_embdings.add_(gdn_out.view(-1, self.embed_dim_))
        gdn_out = None
        input1 = layer_weight.ffn_norm_weight_(input=input_embdings, eps=self.eps_, alloc_func=self.alloc_tensor)

        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def context_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        """Override context_forward to use GDN logic instead of standard attention flow."""
        return self._gdn_layer_forward(input_embdings, infer_state, layer_weight, is_prefill=True)

    def token_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        """Override token_forward to use GDN logic instead of standard attention flow."""
        return self._gdn_layer_forward(input_embdings, infer_state, layer_weight, is_prefill=False)

    def overlap_tpsp_token_forward(
        self,
        input_embdings,
        input_embdings1,
        infer_state: Qwen3NextInferStateInfo,
        infer_state1: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        """Microbatch overlap for decode: process two half-batches sequentially.
        Enables --enable_decode_microbatch_overlap for GDN layers."""
        input_embdings = self.token_forward(input_embdings, infer_state, layer_weight)
        input_embdings1 = self.token_forward(input_embdings1, infer_state1, layer_weight)
        return input_embdings, input_embdings1

    def overlap_tpsp_context_forward(
        self,
        input_embdings,
        input_embdings1,
        infer_state: Qwen3NextInferStateInfo,
        infer_state1: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        """Microbatch overlap for context: process two half-batches sequentially."""
        input_embdings = self.context_forward(input_embdings, infer_state, layer_weight)
        input_embdings1 = self.context_forward(input_embdings1, infer_state1, layer_weight)
        return input_embdings, input_embdings1

    # ==================== GDN Helper Methods ====================

    def _split_qkvzba(self, mixed_qkvzba, is_decode=False):
        qkv_dim = self.tp_key_dim * 2 + self.tp_value_dim
        z_end = qkv_dim + self.tp_value_dim
        b_end = z_end + self.tp_num_v_heads
        mixed_qkv = mixed_qkvzba[:, :qkv_dim]
        z = mixed_qkvzba[:, qkv_dim:z_end].view(-1, self.tp_num_v_heads, self.head_v_dim)
        b = mixed_qkvzba[:, z_end:b_end]
        a = mixed_qkvzba[:, b_end:]
        return mixed_qkv, z, b, a

    def _rearrange_mixed_qkv(self, mixed_qkv, decode=False):
        if mixed_qkv is None:
            return None, None, None
        if decode:
            query, key, value = torch.split(
                mixed_qkv,
                [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim],
                dim=-1,
            )
            batch_size = mixed_qkv.shape[0]
            query = query.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(batch_size, 1, self.tp_num_v_heads, self.head_v_dim)
            return query, key, value
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim],
                dim=-1,
            )
            seq_len = query.shape[0]
            query = query.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(1, seq_len, self.tp_num_v_heads, self.head_v_dim)
            return query, key, value

    def context_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        gdn_out = self.gdn_forward(input_embdings, infer_state, layer_weight, is_prefill=True)
        return gdn_out

    def token_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        gdn_out = self.gdn_forward(input_embdings, infer_state, layer_weight, is_prefill=False)
        return gdn_out

    def _gdn_prefill_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        """Prefill kernel for GDN forward pass."""
        # Conv1D processing
        mixed_qkv = mixed_qkv.transpose(0, 1)
        out_tensor = causal_conv1d_fn(
            mixed_qkv,
            layer_weight.linear_conv1d.mm_param.weight,
            bias=layer_weight.linear_conv1d.bias,
            query_start_loc=infer_state.b1_cu_q_seq_len,
            cache_indices=infer_state.b_buffer_idx,
            has_initial_state=infer_state.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation=self.activation,
        )
        mixed_qkv = out_tensor.transpose(0, 1)

        # Recurrent processing
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv)
        initial_state = ssm_states[infer_state.b_buffer_idx]
        # g and beta have shape (total_tokens, num_heads), need to unsqueeze to get (1, total_tokens, num_heads)
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=infer_state.b1_cu_q_seq_len,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        # Use pre-computed dtype conversion flag to avoid runtime check
        if self.needs_ssm_dtype_conversion:
            ssm_states[infer_state.b_buffer_idx] = last_recurrent_state.to(self.ssm_state_dtype, copy=False)
        else:
            ssm_states[infer_state.b_buffer_idx] = last_recurrent_state
        return core_attn_out

    def _gdn_decode_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        """Decode kernel for GDN forward pass (single-token, non-MTP mode).
        Uses fused gating: g/beta computed inline in the recurrent kernel."""
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer_weight.linear_conv1d.mm_param.weight,
            bias=layer_weight.linear_conv1d.bias,
            activation=self.activation,
            conv_state_indices=infer_state.b_buffer_idx,
        )

        # Recurrent processing with fused gating
        # FusedRecurrentFunction.forward calls .contiguous() on q/k/v/a/b internally
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv, decode=True)
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            initial_state=ssm_states,
            inplace_final_state=True,
            ssm_state_indices=infer_state.b_buffer_idx,
            use_qk_l2norm_in_kernel=True,
            A_log=layer_weight.linear_A_log.weight,
            dt_bias=layer_weight.linear_dt_bias.weight,
            a_raw=a,
            b_raw=b,
        )
        return core_attn_out

    def _gdn_decode_mtp_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        """
        Optimized decode kernel for GDN forward pass (MTP mode with multiple steps).

        Key optimizations:
        1. Uses pre-allocated work buffer to avoid per-step .contiguous() allocations
        2. Uses optimized flat Triton kernels for state copying
        3. Direct slice assignment for output instead of .copy_()

        Note: Sequential processing is required because each MTP step depends on
        the previous step's final state (both conv and SSM states).
        """
        total_tokens = mixed_qkv.shape[0]
        batch_size = total_tokens // self.mtp_size

        # Pre-allocate output tensor
        core_attn_out = torch.empty(
            (total_tokens, 1, self.tp_num_v_heads, self.head_v_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )

        # Pre-allocate work buffer for conv1d input (avoids per-step .contiguous())
        qkv_work_buffer = torch.empty(
            (batch_size, mixed_qkv.shape[-1]),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )

        # Process each MTP step sequentially (required due to state dependencies)
        for step_idx in range(self.mtp_size):
            cur_buffer_idx = infer_state.mtp_buffer_idx_list[step_idx]

            # ========== Conv1D processing ==========
            # Copy strided data to contiguous work buffer
            qkv_work_buffer.copy_(mixed_qkv[step_idx :: self.mtp_size])

            # causal_conv1d_update operates in-place on contiguous input
            causal_conv1d_update(
                qkv_work_buffer,
                conv_states,
                layer_weight.linear_conv1d.mm_param.weight,
                bias=layer_weight.linear_conv1d.bias,
                activation=self.activation,
                conv_state_indices=cur_buffer_idx,
            )

            # ========== Recurrent processing ==========
            query_i, key_i, value_i = self._rearrange_mixed_qkv(qkv_work_buffer, decode=True)
            g_i = g[step_idx :: self.mtp_size].unsqueeze(1)
            beta_i = beta[step_idx :: self.mtp_size].unsqueeze(1)

            core_attn_out_i, _ = fused_recurrent_gated_delta_rule(
                q=query_i,
                k=key_i,
                v=value_i,
                g=g_i,
                beta=beta_i,
                initial_state=ssm_states,
                inplace_final_state=True,
                ssm_state_indices=cur_buffer_idx,
                use_qk_l2norm_in_kernel=True,
            )

            # Direct slice assignment (no .copy_() needed)
            core_attn_out[step_idx :: self.mtp_size] = core_attn_out_i

            # ========== State propagation to next step ==========
            if step_idx < self.mtp_step:
                next_buffer_idx = infer_state.mtp_buffer_idx_list[step_idx + 1]
                if conv_states.is_contiguous() and ssm_states.is_contiguous():
                    copy_states_fused(conv_states, ssm_states, cur_buffer_idx, next_buffer_idx)
                else:
                    copy_conv_states(conv_states, cur_buffer_idx, next_buffer_idx)
                    copy_ssm_states(ssm_states, cur_buffer_idx, next_buffer_idx)

        return core_attn_out

    def gdn_forward(
        self,
        input: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
        is_prefill: bool,
    ):
        assert isinstance(infer_state.mem_manager, Qwen3NextHybridMemManager)

        # Common preprocessing
        input = input.view(-1, self.embed_dim_)
        conv_states, ssm_states = infer_state.mem_manager.get_mamba_cache(self.layer_num_)

        mixed_qkvzba = layer_weight.linear_in_proj.mm(input)
        # mixed_qkv is now returned pre-concatenated (no torch.cat needed)
        mixed_qkv, z, b, a = self._split_qkvzba(mixed_qkvzba, is_decode=not is_prefill)

        # Dispatch to appropriate kernel
        if is_prefill:
            # Prefill: compute g/beta upfront (chunk kernel doesn't support fused gating)
            g, beta = fused_gdn_gating(layer_weight.linear_A_log.weight, a, b, layer_weight.linear_dt_bias.weight)
            core_attn_out = self._gdn_prefill_kernel(
                mixed_qkv, conv_states, ssm_states, g, beta, infer_state, layer_weight
            )
        elif self.mtp_step == 0:
            # Decode (non-MTP): fuse gating into recurrent kernel to save 2 kernel launches
            core_attn_out = self._gdn_decode_kernel(mixed_qkv, conv_states, ssm_states, a, b, infer_state, layer_weight)
        else:
            # Decode (MTP): compute g/beta upfront (multiple recurrent calls per step)
            g, beta = fused_gdn_gating(layer_weight.linear_A_log.weight, a, b, layer_weight.linear_dt_bias.weight)
            core_attn_out = self._gdn_decode_mtp_kernel(
                mixed_qkv, conv_states, ssm_states, g, beta, infer_state, layer_weight
            )

        # Common postprocessing
        num_tokens = z.shape[0]  # batch (decode) or total_tokens (prefill/MTP)
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        norm_out = self.alloc_tensor(core_attn_out.shape, core_attn_out.dtype, device=core_attn_out.device)
        gated_rmsnorm_forward(
            core_attn_out,
            layer_weight.linear_norm.weight,
            None,  # RMSNormWeight has no bias
            self.eps_,
            z,
            out=norm_out,
        )
        # Merge head and value dims in a single view: (num_tokens * HV, V) → (num_tokens, HV * V)
        core_attn_out = norm_out.view(num_tokens, -1)

        output = layer_weight.linear_out_proj.mm(core_attn_out)
        # Note: all_reduce is handled by context_forward/token_forward callers
        return output
