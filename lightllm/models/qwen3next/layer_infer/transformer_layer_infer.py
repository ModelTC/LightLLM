import os
import torch
import torch.nn.functional as F
from torch.distributed import ReduceOp
from typing import Tuple
from typing_extensions import override
from functools import partial
from einops import rearrange

from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextFullAttentionTransformerLayerWeight,
    Qwen3NextGatedDeltaNetTransformerLayerWeight,
)
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_template import TransformerLayerInferTpl
from lightllm.utils.log_utils import init_logger
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.models.qwen3next.mem_manager import Qwen3NextHybridMemManager
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
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)

# Module-level constant for MoE mode
MOE_MODE = os.environ.get("MOE_MODE", "TP")


def is_moe_layer(layer_num: int, network_config: dict) -> bool:
    """Determine if a layer should use MoE based on network configuration."""
    return (
        network_config.get("num_experts", 0) > 0
        and layer_num not in network_config.get("mlp_only_layers", [])
        and (layer_num + 1) % network_config.get("decoder_sparse_step", 1) == 0
    )


class Qwen3NextFFNMixin:
    """
    Mixin providing shared FFN implementations for Qwen3Next layers.

    Both full attention and GDN layers use identical FFN logic (standard FFN,
    shared expert + MoE with TP or EP modes). This mixin eliminates duplication.

    Requires the using class to have:
    - embed_dim_: int
    - num_experts_per_tok: int
    - norm_topk_prob: bool
    - alloc_tensor(): method
    """

    def _standard_ffn(self, input, infer_state, layer_weight) -> torch.Tensor:
        """Standard FFN using shared expert weights (for non-MoE layers)."""
        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn2_out = layer_weight.shared_expert_down_proj.mm(ffn1_out)
        return ffn2_out

    def _ffn_with_shared_expert_tp(self, input, infer_state, layer_weight) -> torch.Tensor:
        """FFN with shared expert + MoE (tensor parallelism mode)."""
        input = input.view(-1, self.embed_dim_)

        # Shared expert
        up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn2_out = layer_weight.shared_expert_down_proj.mm(ffn1_out)
        shared_expert_out = F.sigmoid(layer_weight.shared_expert_gate.mm(input)) * ffn2_out

        # MoE
        moe_out = self._moe_ffn(input, infer_state, layer_weight)

        return shared_expert_out + moe_out

    def _ffn_with_shared_expert_ep(self, input, infer_state, layer_weight) -> torch.Tensor:
        """FFN with shared expert + MoE (expert parallelism mode)."""
        input = input.view(-1, self.embed_dim_)

        # Shared expert
        up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn2_out = layer_weight.shared_expert_down_proj.mm(ffn1_out)
        shared_expert_out = F.sigmoid(layer_weight.shared_expert_gate.mm(input)) * ffn2_out

        # MoE (EP mode)
        moe_out = self._moe_ffn_edp(input, infer_state, layer_weight)

        return shared_expert_out + moe_out

    def _moe_ffn(self, input, infer_state, layer_weight) -> torch.Tensor:
        """MoE FFN with tensor parallelism."""
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

    def _moe_ffn_edp(self, input, infer_state, layer_weight) -> torch.Tensor:
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

        return ep_output.view(token_num, hidden_dim)


class Qwen3NextFullAttentionTransformerLayerInfer(Qwen3NextFFNMixin, LlamaTransformerLayerInfer):
    """
    Full attention layer for Qwen3Next.
    Inherits from LlamaTransformerLayerInfer to get standard attention via abstraction.
    """

    def __init__(self, layer_num, network_config):
        # Store Qwen3Next specific configs before calling super().__init__
        self.partial_rotary_factor = network_config.get("partial_rotary_factor", 1.0)
        self.n_routed_experts = network_config.get("num_experts", 0)
        self.is_moe = is_moe_layer(layer_num, network_config)
        self.num_experts_per_tok = network_config.get("num_experts_per_tok", 1)
        self.norm_topk_prob = network_config.get("norm_topk_prob", False)

        super().__init__(layer_num, network_config)
        # Override head_dim which may be different in Qwen3Next
        self.head_dim_ = network_config.get(
            "head_dim", network_config["hidden_size"] // network_config["num_attention_heads"]
        )

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()

    def _bind_norm(self):
        """Use Gemma-style RMSNorm."""
        self._att_norm = partial(Qwen3NextFullAttentionTransformerLayerInfer._att_norm_impl, self)
        self._ffn_norm = partial(Qwen3NextFullAttentionTransformerLayerInfer._ffn_norm_impl, self)

    def _att_norm_impl(
        self,
        input,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.att_norm_weight_.weight, self.eps_, out=out)
        return out

    def _ffn_norm_impl(
        self,
        input,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.ffn_norm_weight_.weight, self.eps_, out=out)
        return out

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """QKV projection with output gating, Q/K normalization, and partial rotary embedding."""
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

    def _bind_ffn(self):
        """Bind FFN implementation (MoE or shared expert + MoE)."""
        if self.is_moe:
            if MOE_MODE == "EP":
                self._ffn = partial(Qwen3NextFFNMixin._ffn_with_shared_expert_ep, self)
            else:
                self._ffn = partial(Qwen3NextFFNMixin._ffn_with_shared_expert_tp, self)
        else:
            self._ffn = partial(Qwen3NextFFNMixin._standard_ffn, self)


class Qwen3NextGatedDeltaNetTransformerLayerInfer(Qwen3NextFFNMixin, TransformerLayerInferTpl):
    """
    Linear attention (Gated Delta Networks) layer for Qwen3Next.
    Inherits from TransformerLayerInferTpl and overrides attention methods with custom GDN logic.
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.network_config_ = network_config

        # MoE configuration
        self.n_routed_experts = network_config.get("num_experts", 0)
        self.is_moe = is_moe_layer(layer_num, network_config)
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

        self._bind_func()

    def _bind_func(self):
        """Bind layer-specific implementations."""
        self._bind_norm()
        self._bind_ffn()

    def _bind_norm(self):
        """Use Gemma-style RMSNorm."""
        self._att_norm = partial(Qwen3NextGatedDeltaNetTransformerLayerInfer._att_norm_impl, self)
        self._ffn_norm = partial(Qwen3NextGatedDeltaNetTransformerLayerInfer._ffn_norm_impl, self)

    def _bind_ffn(self):
        """Bind FFN implementation (MoE or standard)."""
        if self.is_moe:
            if MOE_MODE == "EP":
                self._ffn = partial(Qwen3NextFFNMixin._ffn_with_shared_expert_ep, self)
            else:
                self._ffn = partial(Qwen3NextFFNMixin._ffn_with_shared_expert_tp, self)
        else:
            self._ffn = partial(Qwen3NextFFNMixin._standard_ffn, self)

    def _att_norm_impl(
        self,
        input,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """Linear attention normalization."""
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.att_norm_weight_.weight, self.eps_, out=out)
        return out

    def _ffn_norm_impl(
        self,
        input,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """FFN normalization."""
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.ffn_norm_weight_.weight, self.eps_, out=out)
        return out

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Not used by GDN - implemented in gdn_forward."""
        raise NotImplementedError("GDN uses gdn_forward instead of _get_qkv")

    def _tpsp_get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Not implemented for GDN."""
        raise NotImplementedError("TPSP mode not implemented for GDN layers")

    def _get_o(
        self,
        input,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """Not used by GDN - output computed in gdn_forward."""
        raise NotImplementedError("GDN uses gdn_forward instead of _get_o")

    def _tpsp_get_o(
        self,
        input,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """Not implemented for GDN."""
        raise NotImplementedError("TPSP mode not implemented for GDN layers")

    def _context_attention_kernel(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """Not used by GDN."""
        raise NotImplementedError("GDN uses gdn_forward instead of _context_attention_kernel")

    def _token_attention_kernel(
        self,
        q: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ) -> torch.Tensor:
        """Not used by GDN."""
        raise NotImplementedError("GDN uses gdn_forward instead of _token_attention_kernel")

    def context_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Override context_forward to use GDN logic instead of standard attention flow."""
        # Attention + GDN processing
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        gdn_out = self.gdn_forward(input1, infer_state, layer_weight, is_prefill=True)
        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(gdn_out.view(-1, self.embed_dim_))
        gdn_out = None

        # FFN
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetTransformerLayerWeight,
    ):
        """Override token_forward to use GDN logic instead of standard attention flow."""
        # Attention + GDN processing
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        gdn_out = self.gdn_forward(input1, infer_state, layer_weight, is_prefill=False)
        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(gdn_out.view(-1, self.embed_dim_))
        gdn_out = None

        # FFN
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

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
        (query, key, value, z) = torch.split(mixed_qkvz, qkvz_split_list, dim=2)
        (b, a) = torch.split(mixed_ba, [self.num_v_heads_per_k_head, self.num_v_heads_per_k_head], dim=2)

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
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
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
        # Only convert dtype if necessary to avoid overhead
        if last_recurrent_state.dtype != ssm_states.dtype:
            ssm_states[infer_state.b_buffer_idx] = last_recurrent_state.to(ssm_states.dtype, copy=False)
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
        # g and beta have shape (1, batch, num_heads), need to squeeze and unsqueeze to get (batch, 1, num_heads)
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g.squeeze(0).unsqueeze(1),
            beta=beta.squeeze(0).unsqueeze(1),
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

        g_squeezed = g.squeeze(0)  # [total_tokens, num_heads]
        beta_squeezed = beta.squeeze(0)

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
            g_i = g_squeezed[step_idx :: self.mtp_size].unsqueeze(1)
            beta_i = beta_squeezed[step_idx :: self.mtp_size].unsqueeze(1)

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
        norm_out = self.alloc_tensor(core_attn_out.shape, core_attn_out.dtype)
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
