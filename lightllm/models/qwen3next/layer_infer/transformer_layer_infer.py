import torch
import torch.distributed as dist
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextFullAttentionTransformerLayerWeight,
    Qwen3NextGatedDeltaNetTransformerLayerWeight,
)
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_template import TransformerLayerInferTpl
from lightllm.utils.log_utils import init_logger
from lightllm.models.qwen3next.mem_manager import Qwen3NextHybridMemManager
from lightllm.models.qwen3next.layer_infer.shared_expert_mixin import SharedExpertFFNMixin
from typing import Tuple
from typing_extensions import override
from einops import rearrange
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
from lightllm.models.qwen3next.triton_kernel.gemma_rmsnorm import gemma_rmsnorm_forward
from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
from functools import partial

logger = init_logger(__name__)


class GemmaRMSNormMixin:
    """
    Mixin providing Gemma-style RMSNorm implementations with buffer pooling.

    Requirements:
    - Class must have: eps_, buffer_pool, alloc_tensor()
    """

    def _gemma_norm_with_pool(self, input, norm_weight):
        """Apply Gemma RMSNorm with optional buffer pooling."""
        if self.buffer_pool:
            out = self.buffer_pool.get_buffer(input.shape, input.dtype, input.device)
        else:
            out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, norm_weight, self.eps_, out=out)
        return out


class Qwen3NextFullAttentionBaseLayerInfer(GemmaRMSNormMixin, SharedExpertFFNMixin, LlamaTransformerLayerInfer):
    """
    Base class for Qwen3Next full attention layers.
    Contains shared logic for both standard full attention and MTP layers.
    Inherits from SharedExpertFFNMixin for FFN logic, LlamaTransformerLayerInfer for attention.
    """

    def __init__(self, layer_num, network_config):
        # Store Qwen3Next specific configs before calling super().__init__
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
        # Override head_dim which may be different in Qwen3Next
        self.head_dim_ = network_config.get(
            "head_dim", network_config["hidden_size"] // network_config["num_attention_heads"]
        )
        self.buffer_pool = None  # Set by model during _init_infer_layer
        return

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_norm(self):
        """Use Gemma-style RMSNorm"""
        self._att_norm = partial(Qwen3NextFullAttentionBaseLayerInfer._att_norm_impl, self)
        self._ffn_norm = partial(Qwen3NextFullAttentionBaseLayerInfer._ffn_norm_impl, self)
        return

    def _att_norm_impl(
        self,
        input,
        _infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> torch.Tensor:
        return self._gemma_norm_with_pool(input, layer_weight.att_norm_weight_.weight)

    def _ffn_norm_impl(
        self,
        input,
        _infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> torch.Tensor:
        return self._gemma_norm_with_pool(input, layer_weight.ffn_norm_weight_.weight)

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        QKV projection with output gating, Q/K normalization, and partial rotary embedding.
        """
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        # Save gate value for output projection
        infer_state.gate_value = torch.sigmoid(layer_weight.o_gate_proj.mm(input))
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        # Q normalization
        gemma_rmsnorm_forward(
            q.view(-1, self.head_dim_),
            layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        # K normalization
        cache_kv[:, : self.tp_k_head_num_, :] = gemma_rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        # Rotary embedding with partial rotation support
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
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> torch.Tensor:
        """Output projection with gating."""
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        gated_input = input * infer_state.gate_value
        infer_state.gate_value = None
        o_tensor = layer_weight.o_proj.mm(gated_input)
        return o_tensor


class Qwen3NextFullAttentionTransformerLayerInfer(Qwen3NextFullAttentionBaseLayerInfer):
    """
    Full attention layer for Qwen3Next that uses the abstracted attention backend.
    Inherits from Qwen3NextFullAttentionBaseLayerInfer to get shared Qwen3Next logic.
    """

    pass


class Qwen3NextGatedDeltaNetTransformerLayerInfer(GemmaRMSNormMixin, SharedExpertFFNMixin, TransformerLayerInferTpl):
    """
    Linear attention (Gated Delta Networks) layer for Qwen3Next.
    Inherits from SharedExpertFFNMixin for FFN logic, TransformerLayerInferTpl for structure.
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.network_config_ = network_config

        # MoE configuration
        self.n_routed_experts = network_config.get("num_experts", 0)
        self.is_moe = (
            network_config.get("num_experts", 0) > 0
            and layer_num not in network_config.get("mlp_only_layers", [])
            and (layer_num + 1) % network_config.get("decoder_sparse_step", 1) == 0
        )
        self.num_experts_per_tok = network_config.get("num_experts_per_tok", 1)
        self.norm_topk_prob = network_config.get("norm_topk_prob", False)

        # Standard layer dimensions
        self.eps_ = network_config["rms_norm_eps"]
        self.embed_dim_ = network_config["hidden_size"]

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

        self.buffer_pool = None  # Set by model during _init_infer_layer

        self._bind_func()
        return

    def _bind_func(self):
        """Bind layer-specific implementations"""
        self._bind_norm()
        self._bind_ffn()
        return

    def _bind_norm(self):
        """Use Gemma-style RMSNorm"""
        self._att_norm = partial(Qwen3NextGatedDeltaNetTransformerLayerInfer._att_norm_impl, self)
        self._ffn_norm = partial(Qwen3NextGatedDeltaNetTransformerLayerInfer._ffn_norm_impl, self)
        return

    def _att_norm_impl(
        self,
        input,
        _infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        return self._gemma_norm_with_pool(input, layer_weight.att_norm_weight_.weight)

    def _ffn_norm_impl(
        self,
        input,
        _infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        return self._gemma_norm_with_pool(input, layer_weight.ffn_norm_weight_.weight)

    def _get_qkv(
        self,
        _input: torch.Tensor,
        _infer_state: Qwen3NextInferStateInfo,
        _layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Not used by GDN - QKV projection handled in gdn_forward.

        GDN uses a fused projection that includes z, b, a parameters
        in addition to q, k, v, so the standard template flow doesn't apply.
        This method exists to satisfy the template interface.
        """
        pass  # Implementation in gdn_forward

    def _tpsp_get_qkv(
        self,
        _input: torch.Tensor,
        _infer_state: Qwen3NextInferStateInfo,
        _layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TPSP mode not implemented for GDN layers."""
        pass  # No TPSP support planned

    def _get_o(
        self,
        _input,
        _infer_state: Qwen3NextInferStateInfo,
        _layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """
        Not used by GDN - output projection handled in gdn_forward.

        Output computation is fused with GDN recurrence in gdn_forward.
        """
        pass  # Implementation in gdn_forward

    def _tpsp_get_o(
        self,
        _input,
        _infer_state: Qwen3NextInferStateInfo,
        _layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """TPSP mode not implemented for GDN layers."""
        pass  # No TPSP support planned

    def _context_attention_kernel(
        self,
        _q: torch.Tensor,
        _kv: torch.Tensor,
        _infer_state: Qwen3NextInferStateInfo,
        _layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """Not used by GDN - attention computed in gdn_forward."""
        pass  # Implementation in gdn_forward

    def _token_attention_kernel(
        self,
        _q: torch.Tensor,
        _infer_state: Qwen3NextInferStateInfo,
        _layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """Not used by GDN - attention computed in gdn_forward."""
        pass  # Implementation in gdn_forward

    def _gdn_layer_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
        is_prefill: bool,
    ):
        """Unified forward for both prefill and decode in GDN layers."""
        # Attention + GDN processing
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        gdn_out = self.gdn_forward(input1, infer_state, layer_weight, is_prefill=is_prefill)
        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(gdn_out.view(-1, self.embed_dim_))
        gdn_out = None

        # FFN
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
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
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Override context_forward to use GDN logic instead of standard attention flow."""
        return self._gdn_layer_forward(input_embdings, infer_state, layer_weight, is_prefill=True)

    def token_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Override token_forward to use GDN logic instead of standard attention flow."""
        return self._gdn_layer_forward(input_embdings, infer_state, layer_weight, is_prefill=False)

    # ==================== GDN Helper Methods ====================

    def _fix_query_key_value_ba_ordering(self, mixed_qkvzba):
        """
        Derives `query`, `key`, `value`, `z`, `b`, `a` tensors from `mixed_qkvzba`.
        Returns qkv already concatenated to avoid allocation in gdn_forward.
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
        query, key, value, z = torch.split(mixed_qkvz, qkvz_split_list, dim=2)
        b, a = torch.split(mixed_ba, [self.num_v_heads_per_k_head, self.num_v_heads_per_k_head], dim=2)

        # Reshape qkv components
        query = query.reshape(-1, self.tp_num_k_heads * self.head_k_dim)
        key = key.reshape(-1, self.tp_num_k_heads * self.head_k_dim)
        value = value.reshape(-1, self.tp_num_v_heads * self.head_v_dim)

        # Concatenate qkv here instead of in gdn_forward (avoids extra allocation)
        mixed_qkv = torch.cat([query, key, value], dim=-1)

        z = z.reshape(-1, self.tp_num_v_heads, self.head_v_dim)
        b = b.reshape(-1, self.tp_num_v_heads)
        a = a.reshape(-1, self.tp_num_v_heads)

        return mixed_qkv, z, b, a

    def _rearrange_mixed_qkv(self, mixed_qkv, decode=False):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim],
            dim=-1,
        )
        if decode:
            batch_size = mixed_qkv.shape[0]
            query = query.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(batch_size, 1, self.tp_num_v_heads, self.head_v_dim)
        else:
            query, key = map(lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim), (query, key))
            value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query, key, value

    @override
    def context_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        gdn_out = self.gdn_forward(input_embdings, infer_state, layer_weight, is_prefill=True)
        return gdn_out

    @override
    def token_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
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
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Prefill kernel for GDN forward pass."""
        # Conv1D processing
        mixed_qkv = mixed_qkv.transpose(0, 1)
        out_tensor = causal_conv1d_fn(
            mixed_qkv,
            layer_weight.linear_conv1d.mm_param.weight,
            bias=layer_weight.linear_conv1d.mm_param.bias,
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
        g: torch.Tensor,
        beta: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Decode kernel for GDN forward pass (single-token, non-MTP mode)."""
        # Conv1D processing
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer_weight.linear_conv1d.mm_param.weight,
            bias=layer_weight.linear_conv1d.mm_param.bias,
            activation=self.activation,
            conv_state_indices=infer_state.b_buffer_idx,
        )

        # Recurrent processing
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv, decode=True)
        # g and beta have shape (batch, num_heads), need to unsqueeze to get (batch, 1, num_heads)
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g.unsqueeze(1),
            beta=beta.unsqueeze(1),
            initial_state=ssm_states,
            inplace_final_state=True,
            ssm_state_indices=infer_state.b_buffer_idx,
            use_qk_l2norm_in_kernel=True,
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
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
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
                bias=layer_weight.linear_conv1d.mm_param.bias,
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
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
        is_prefill: bool,
    ):
        assert isinstance(infer_state.mem_manager, Qwen3NextHybridMemManager)

        # Common preprocessing
        input = input.view(-1, self.embed_dim_)
        conv_states, ssm_states = infer_state.mem_manager.get_mamba_cache(self.layer_num_)

        mixed_qkvzba = layer_weight.linear_in_proj.mm(input)
        # mixed_qkv is now returned pre-concatenated (no torch.cat needed)
        mixed_qkv, z, b, a = self._fix_query_key_value_ba_ordering(mixed_qkvzba)

        # Compute g and beta for all modes
        g, beta = fused_gdn_gating(layer_weight.linear_A_log.weight, a, b, layer_weight.linear_dt_bias.weight)

        # Dispatch to appropriate kernel
        if is_prefill:
            core_attn_out = self._gdn_prefill_kernel(
                mixed_qkv, conv_states, ssm_states, g, beta, infer_state, layer_weight
            )
        elif self.mtp_step == 0:
            core_attn_out = self._gdn_decode_kernel(
                mixed_qkv, conv_states, ssm_states, g, beta, infer_state, layer_weight
            )
        else:
            core_attn_out = self._gdn_decode_mtp_kernel(
                mixed_qkv, conv_states, ssm_states, g, beta, infer_state, layer_weight
            )

        # Common postprocessing
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if self.buffer_pool:
            norm_out = self.buffer_pool.get_buffer(core_attn_out.shape, core_attn_out.dtype, core_attn_out.device)
        else:
            norm_out = self.alloc_tensor(core_attn_out.shape, core_attn_out.dtype, device=core_attn_out.device)
        gated_rmsnorm_forward(
            core_attn_out,
            layer_weight.linear_norm.weight,
            layer_weight.linear_norm.bias,
            self.eps_,
            z,
            out=norm_out,
        )
        core_attn_out = norm_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")

        output = layer_weight.linear_out_proj.mm(core_attn_out)
        # Note: all_reduce is handled by context_forward/token_forward callers
        return output
