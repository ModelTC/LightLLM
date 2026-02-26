import os
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
from lightllm.models.qwen3next.triton_kernel.gemma_rmsnorm import gemma_rmsnorm_forward
from lightllm.models.qwen3next.triton_kernel.fused_add_gemma_rmsnorm import fused_add_gemma_rmsnorm
from lightllm.models.qwen3next.triton_kernel.fused_split_copy import fused_split_copy_qkvzba, fused_split_copy_qkv
from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
from functools import partial

logger = init_logger(__name__)


class GemmaRMSNormMixin:
    """
    Mixin providing Gemma-style RMSNorm implementations.

    Requirements:
    - Class must have: eps_, alloc_tensor()
    """

    def _gemma_norm_with_pool(self, input, norm_weight):
        """Apply Gemma RMSNorm."""
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, norm_weight, self.eps_, out=out)
        return out


class Qwen3NextFullAttentionBaseLayerInfer(GemmaRMSNormMixin, LlamaTransformerLayerInfer):
    """
    Base class for Qwen3Next full attention layers.
    Contains shared logic for both standard full attention and MTP layers.
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

        # Pre-allocated decode buffers (mirrors GDN layer pattern)
        start_args = get_env_start_args()
        self._decode_buffers = {}
        self._graph_max_batch_size = start_args.graph_max_batch_size

        # Pre-compute dims for decode buffer pre-allocation
        self.shared_inter_size = network_config.get("shared_expert_intermediate_size", 0)
        self.tp_gate_up_dim = 2 * self.shared_inter_size // self.tp_world_size_ if self.shared_inter_size > 0 else 0
        self.tp_q_gate_dim = (self.tp_q_head_num_ + self.tp_o_head_num_) * self.head_dim_
        self.tp_kv_dim = (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_

        return

    def _get_decode_buffer(self, name, max_shape, dtype, device):
        """Get or create a pre-allocated buffer for the decode path."""
        key = (name, dtype, device if isinstance(device, str) else str(device))
        if key not in self._decode_buffers:
            self._decode_buffers[key] = torch.empty(max_shape, dtype=dtype, device=device)
        return self._decode_buffers[key]

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_norm(self):
        """Use Gemma-style RMSNorm"""
        self._att_norm = partial(Qwen3NextFullAttentionBaseLayerInfer._att_norm_impl, self)
        self._ffn_norm = partial(Qwen3NextFullAttentionBaseLayerInfer._ffn_norm_impl, self)
        return

    def _bind_ffn(self):
        """Bind FFN implementation based on MoE configuration."""
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._ffn_with_shared_expert_ep, self)
            else:
                self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._ffn_with_shared_expert_tp, self)
        else:
            self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._standard_ffn, self)
        return

    def _ffn_core(self, input, layer_weight, is_decode=False):
        """Core FFN computation: gate_up -> silu_and_mul -> down."""
        input = input.view(-1, self.embed_dim_)
        if is_decode and self.tp_gate_up_dim > 0:
            up_gate_buf = self._get_decode_buffer(
                "up_gate_out",
                (self._graph_max_batch_size, self.tp_gate_up_dim),
                input.dtype,
                input.device,
            )[: input.size(0)]
            up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input, out=up_gate_buf)
        else:
            up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input)
        inter_dim = up_gate_out.size(1) // 2
        if is_decode:
            ffn1_out = self._get_decode_buffer(
                "ffn1_out", (self._graph_max_batch_size, inter_dim), input.dtype, input.device
            )[: input.size(0)]
        else:
            ffn1_out = self.alloc_tensor((input.size(0), inter_dim), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn2_out = layer_weight.shared_expert_down_proj.mm(ffn1_out)
        return ffn2_out, input

    def _standard_ffn(self, input, infer_state, layer_weight):
        """Standard FFN using shared expert weights (non-MoE layers)."""
        # For dense models without shared experts, return zeros (no FFN computation)
        if not hasattr(layer_weight, "shared_expert_gate_up_proj") or layer_weight.shared_expert_gate_up_proj is None:
            return torch.zeros_like(input)
        ffn2_out, _ = self._ffn_core(input, layer_weight, is_decode=not infer_state.is_prefill)
        return ffn2_out

    def _compute_shared_expert(self, input, layer_weight, is_decode=False):
        """Compute shared expert FFN output with gating."""
        ffn2_out, input_view = self._ffn_core(input, layer_weight, is_decode=is_decode)
        # Dense models don't have shared_expert_gate
        if layer_weight.shared_expert_gate is not None:
            gate = layer_weight.shared_expert_gate.mm(input_view).sigmoid_()
            ffn2_out.mul_(gate)
        return ffn2_out, input_view

    def _ffn_with_shared_expert_tp(self, input, infer_state, layer_weight):
        """FFN with shared expert + MoE (tensor parallelism mode)."""
        shared_expert_out, input = self._compute_shared_expert(
            input, layer_weight, is_decode=not infer_state.is_prefill
        )
        moe_out = self._moe_ffn(input, infer_state, layer_weight)
        moe_out.add_(shared_expert_out)
        return moe_out

    def _ffn_with_shared_expert_ep(self, input, infer_state, layer_weight):
        """FFN with shared expert + MoE (expert parallelism mode)."""
        shared_expert_out, input = self._compute_shared_expert(
            input, layer_weight, is_decode=not infer_state.is_prefill
        )
        moe_out = self._moe_ffn_edp(input, infer_state, layer_weight)
        moe_out.add_(shared_expert_out)
        return moe_out

    def _moe_ffn(self, input, infer_state, layer_weight):
        """MoE FFN with tensor parallelism."""
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape
        if not infer_state.is_prefill:
            router_buf = self._get_decode_buffer(
                "router_logits",
                (self._graph_max_batch_size, self.n_routed_experts),
                hidden_states.dtype,
                hidden_states.device,
            )[:num_tokens]
            router_logits = layer_weight.moe_gate.mm(hidden_states, out=router_buf)
        else:
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

    def _moe_ffn_edp(self, input, infer_state, layer_weight):
        """MoE FFN with expert parallelism."""
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
        # Single fused GEMM for both Q and output gate projections
        if not infer_state.is_prefill:
            q_gate_buf = self._get_decode_buffer(
                "q_gate_out",
                (self._graph_max_batch_size, self.tp_q_gate_dim),
                input.dtype,
                input.device,
            )[: input.size(0)]
            q_gate = layer_weight.q_gate_proj.mm(input, out=q_gate_buf)
            kv_buf = self._get_decode_buffer(
                "kv_out",
                (self._graph_max_batch_size, self.tp_kv_dim),
                input.dtype,
                input.device,
            )[: input.size(0)]
            kv_out = layer_weight.kv_proj.mm(input, out=kv_buf)
        else:
            q_gate = layer_weight.q_gate_proj.mm(input)
            kv_out = layer_weight.kv_proj.mm(input)
        q_dim = self.tp_q_head_num_ * self.head_dim_
        q = q_gate[:, :q_dim].contiguous()
        # In-place sigmoid saves one allocation (gate_value is consumed once in _get_o)
        infer_state.gate_value = q_gate[:, q_dim:].sigmoid_()
        cache_kv = kv_out.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        # Q normalization (in-place via out=input)
        gemma_rmsnorm_forward(
            q.view(-1, self.head_dim_),
            layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        # K normalization
        k_input = cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1])
        if not infer_state.is_prefill:
            k_normed = self._get_decode_buffer(
                "k_norm_out",
                (self._graph_max_batch_size * self.tp_k_head_num_, cache_kv.shape[-1]),
                k_input.dtype,
                k_input.device,
            )[: k_input.shape[0]]
            gemma_rmsnorm_forward(k_input, layer_weight.k_norm_weight_.weight, eps=self.eps_, out=k_normed)
        else:
            k_normed = gemma_rmsnorm_forward(k_input, layer_weight.k_norm_weight_.weight, eps=self.eps_)
        cache_kv[:, : self.tp_k_head_num_, :] = k_normed.view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

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
        """Output projection with gating (in-place multiply to save one allocation)."""
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        input.mul_(infer_state.gate_value)
        infer_state.gate_value = None
        o_tensor = layer_weight.o_proj.mm(input)
        return o_tensor

    def token_forward(self, input_embdings, infer_state, layer_weight):
        """Override token_forward to use pre-allocated decode buffers and fused kernels."""
        max_tokens = self._graph_max_batch_size
        input1 = self._get_decode_buffer(
            "att_norm_out", (max_tokens, self.embed_dim_), input_embdings.dtype, input_embdings.device
        )[: input_embdings.shape[0]]
        gemma_rmsnorm_forward(input_embdings, layer_weight.att_norm_weight_.weight, self.eps_, out=input1)

        o = self.token_attention_forward(input1, infer_state, layer_weight)

        # Fused residual add + FFN norm: saves 1 kernel launch + 1 read of input_embdings
        input1 = self._get_decode_buffer(
            "att_norm_out", (max_tokens, self.embed_dim_), input_embdings.dtype, input_embdings.device
        )[: input_embdings.shape[0]]
        fused_add_gemma_rmsnorm(
            input_embdings,
            o.view(-1, self.embed_dim_),
            layer_weight.ffn_norm_weight_.weight,
            self.eps_,
            out=input1,
        )
        o = None

        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings


class Qwen3NextFullAttentionTransformerLayerInfer(Qwen3NextFullAttentionBaseLayerInfer):
    """
    Full attention layer for Qwen3Next that uses the abstracted attention backend.
    Inherits from Qwen3NextFullAttentionBaseLayerInfer to get shared Qwen3Next logic.
    """

    pass


class Qwen3NextGatedDeltaNetTransformerLayerInfer(GemmaRMSNormMixin, TransformerLayerInferTpl):
    """
    Linear attention (Gated Delta Networks) layer for Qwen3Next.
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
        self.shared_inter_size = network_config.get("shared_expert_intermediate_size", 0)

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

        # Pre-allocated decode buffers to avoid repeated allocation during CUDA graph replay.
        # Buffers are lazily allocated on first decode call, sized to graph_max_batch_size.
        self._decode_buffers = {}
        self._graph_max_batch_size = start_args.graph_max_batch_size

        # Pre-compute FFN dims for decode buffer pre-allocation
        self.tp_gate_up_dim = 2 * self.shared_inter_size // self.tp_world_size_ if self.shared_inter_size > 0 else 0

        self._bind_func()
        return

    def _get_decode_buffer(self, name, max_shape, dtype, device):
        """Get or create a pre-allocated buffer for the decode path.

        On first call, allocates a buffer at max_shape. On subsequent calls,
        returns the same buffer (caller should slice to actual batch size).
        """
        key = (name, dtype, device if isinstance(device, str) else str(device))
        if key not in self._decode_buffers:
            self._decode_buffers[key] = torch.empty(max_shape, dtype=dtype, device=device)
        return self._decode_buffers[key]

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

    def _bind_ffn(self):
        """Bind FFN implementation based on MoE configuration."""
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Qwen3NextGatedDeltaNetTransformerLayerInfer._ffn_with_shared_expert_ep, self)
            else:
                self._ffn = partial(Qwen3NextGatedDeltaNetTransformerLayerInfer._ffn_with_shared_expert_tp, self)
        else:
            self._ffn = partial(Qwen3NextGatedDeltaNetTransformerLayerInfer._standard_ffn, self)
        return

    def _ffn_core(self, input, layer_weight, is_decode=False):
        """Core FFN computation: gate_up -> silu_and_mul -> down."""
        input = input.view(-1, self.embed_dim_)
        if is_decode and self.tp_gate_up_dim > 0:
            up_gate_buf = self._get_decode_buffer(
                "up_gate_out",
                (self._graph_max_batch_size * self.mtp_size, self.tp_gate_up_dim),
                input.dtype,
                input.device,
            )[: input.size(0)]
            up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input, out=up_gate_buf)
        else:
            up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input)
        inter_dim = up_gate_out.size(1) // 2
        if is_decode:
            ffn1_out = self._get_decode_buffer(
                "ffn1_out", (self._graph_max_batch_size, inter_dim), input.dtype, input.device
            )[: input.size(0)]
        else:
            ffn1_out = self.alloc_tensor((input.size(0), inter_dim), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn2_out = layer_weight.shared_expert_down_proj.mm(ffn1_out)
        return ffn2_out, input

    def _standard_ffn(self, input, infer_state, layer_weight):
        """Standard FFN using shared expert weights (non-MoE layers)."""
        # For dense models without shared experts, return zeros (no FFN computation)
        if not hasattr(layer_weight, "shared_expert_gate_up_proj") or layer_weight.shared_expert_gate_up_proj is None:
            return torch.zeros_like(input)
        ffn2_out, _ = self._ffn_core(input, layer_weight, is_decode=not infer_state.is_prefill)
        return ffn2_out

    def _compute_shared_expert(self, input, layer_weight, is_decode=False):
        """Compute shared expert FFN output with gating."""
        ffn2_out, input_view = self._ffn_core(input, layer_weight, is_decode=is_decode)
        # Dense models don't have shared_expert_gate
        if layer_weight.shared_expert_gate is not None:
            gate = layer_weight.shared_expert_gate.mm(input_view).sigmoid_()
            ffn2_out.mul_(gate)
        return ffn2_out, input_view

    def _ffn_with_shared_expert_tp(self, input, infer_state, layer_weight):
        """FFN with shared expert + MoE (tensor parallelism mode)."""
        shared_expert_out, input = self._compute_shared_expert(
            input, layer_weight, is_decode=not infer_state.is_prefill
        )
        moe_out = self._moe_ffn(input, infer_state, layer_weight)
        moe_out.add_(shared_expert_out)
        return moe_out

    def _ffn_with_shared_expert_ep(self, input, infer_state, layer_weight):
        """FFN with shared expert + MoE (expert parallelism mode)."""
        shared_expert_out, input = self._compute_shared_expert(
            input, layer_weight, is_decode=not infer_state.is_prefill
        )
        moe_out = self._moe_ffn_edp(input, infer_state, layer_weight)
        moe_out.add_(shared_expert_out)
        return moe_out

    def _moe_ffn(self, input, infer_state, layer_weight):
        """MoE FFN with tensor parallelism."""
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape
        if not infer_state.is_prefill:
            router_buf = self._get_decode_buffer(
                "router_logits",
                (self._graph_max_batch_size * self.mtp_size, self.n_routed_experts),
                hidden_states.dtype,
                hidden_states.device,
            )[:num_tokens]
            router_logits = layer_weight.moe_gate.mm(hidden_states, out=router_buf)
        else:
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

    def _moe_ffn_edp(self, input, infer_state, layer_weight):
        """MoE FFN with expert parallelism."""
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
        if is_prefill:
            input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        else:
            # Decode: use pre-allocated buffer to avoid alloc_tensor overhead
            max_tokens = self._graph_max_batch_size * self.mtp_size
            input1 = self._get_decode_buffer(
                "att_norm_out", (max_tokens, self.embed_dim_), input_embdings.dtype, input_embdings.device
            )[: input_embdings.shape[0]]
            gemma_rmsnorm_forward(input_embdings, layer_weight.att_norm_weight_.weight, self.eps_, out=input1)

        gdn_out = self.gdn_forward(input1, infer_state, layer_weight, is_prefill=is_prefill)
        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)

        # FFN
        if is_prefill:
            input_embdings.add_(gdn_out.view(-1, self.embed_dim_))
            gdn_out = None
            input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        else:
            # Decode: fused residual add + FFN norm saves 1 kernel + 1 read of input_embdings
            input1 = self._get_decode_buffer(
                "att_norm_out", (max_tokens, self.embed_dim_), input_embdings.dtype, input_embdings.device
            )[: input_embdings.shape[0]]
            fused_add_gemma_rmsnorm(
                input_embdings,
                gdn_out.view(-1, self.embed_dim_),
                layer_weight.ffn_norm_weight_.weight,
                self.eps_,
                out=input1,
            )
            gdn_out = None

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

    def overlap_tpsp_token_forward(
        self,
        input_embdings,
        input_embdings1,
        infer_state: Qwen3NextInferStateInfo,
        infer_state1: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
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
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Microbatch overlap for context: process two half-batches sequentially."""
        input_embdings = self.context_forward(input_embdings, infer_state, layer_weight)
        input_embdings1 = self.context_forward(input_embdings1, infer_state1, layer_weight)
        return input_embdings, input_embdings1

    # ==================== GDN Helper Methods ====================

    def _fix_query_key_value_ba_ordering(self, mixed_qkvzba, is_decode=False):
        """
        Extract q, k, v, z, b, a from the MM output.

        After weight rearrangement at load time, the MM output is already in grouped layout:
        [all_q | all_k | all_v | all_z | all_b | all_a]
        so this is just simple slicing — no split+reshape+cat needed.

        Note:
        Decode fast-path fused split-copy kernels are intentionally avoided here.
        The explicit contiguous slicing path is slower but is more robust and
        matches the reference behavior used in vLLM.
        """
        qkv_dim = self.tp_key_dim * 2 + self.tp_value_dim
        z_end = qkv_dim + self.tp_value_dim
        b_end = z_end + self.tp_num_v_heads

        if is_decode:
            mixed_qkv = mixed_qkvzba[:, :qkv_dim].contiguous()
            z = mixed_qkvzba[:, qkv_dim:z_end].contiguous().view(-1, self.tp_num_v_heads, self.head_v_dim)
            b = mixed_qkvzba[:, z_end:b_end].contiguous()
            a = mixed_qkvzba[:, b_end:].contiguous()
        else:
            mixed_qkv = mixed_qkvzba[:, :qkv_dim]
            # .reshape() handles non-contiguous slices by copying when needed (unlike .view())
            z = mixed_qkvzba[:, qkv_dim:z_end].reshape(-1, self.tp_num_v_heads, self.head_v_dim)
            # b and a must be contiguous: fused_gdn_gating_kernel uses raw pointer arithmetic
            # (off = i_b * NUM_HEADS + head_off) that assumes contiguous layout.
            # Non-contiguous slices have stride[0]=total_dim, causing wrong reads for i_b > 0.
            b = mixed_qkvzba[:, z_end:b_end].contiguous()
            a = mixed_qkvzba[:, b_end:].contiguous()

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
            query = query.contiguous().view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            key = key.contiguous().view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            value = value.contiguous().view(batch_size, 1, self.tp_num_v_heads, self.head_v_dim)
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
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        gdn_out = self.gdn_forward(input_embdings, infer_state, layer_weight, is_prefill=True)
        return gdn_out

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
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Decode kernel for GDN forward pass (single-token, non-MTP mode).
        Uses fused gating: g/beta computed inline in the recurrent kernel."""
        # Conv1D processing — mixed_qkv is pre-copied to contiguous buffer
        # by _fix_query_key_value_ba_ordering (causal_conv1d_update requires contiguous input)
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
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
        is_prefill: bool,
    ):
        assert isinstance(infer_state.mem_manager, Qwen3NextHybridMemManager)

        # Common preprocessing
        input = input.view(-1, self.embed_dim_)
        conv_states, ssm_states = infer_state.mem_manager.get_mamba_cache(self.layer_num_)

        if not is_prefill:
            # Decode: pre-allocate GEMM output to avoid cache tensor manager overhead
            in_proj_out_dim = self.tp_qkvz_dim + self.tp_ba_dim
            in_proj_out = self._get_decode_buffer(
                "in_proj_out",
                (self._graph_max_batch_size * self.mtp_size, in_proj_out_dim),
                input.dtype,
                input.device,
            )[: input.shape[0]]
            mixed_qkvzba = layer_weight.linear_in_proj.mm(input, out=in_proj_out)
        else:
            mixed_qkvzba = layer_weight.linear_in_proj.mm(input)
        # mixed_qkv is now returned pre-concatenated (no torch.cat needed)
        mixed_qkv, z, b, a = self._fix_query_key_value_ba_ordering(mixed_qkvzba, is_decode=not is_prefill)

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
        if not is_prefill:
            # Decode: use pre-allocated buffer for norm output to avoid alloc_tensor
            max_decode_tokens = self._graph_max_batch_size * self.mtp_size
            flat_size = max_decode_tokens * self.tp_num_v_heads
            norm_out = self._get_decode_buffer(
                "gdn_norm_out",
                (flat_size, self.head_v_dim),
                core_attn_out.dtype,
                core_attn_out.device,
            )[: core_attn_out.shape[0]]
        else:
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
