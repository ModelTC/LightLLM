# lightllm/models/qwen3next/layer_infer/shared_expert_mixin.py
import torch.nn.functional as F
from functools import partial
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
import os


class SharedExpertFFNMixin:
    """
    Mixin providing shared expert + MoE FFN implementations.

    Used by both full attention and GDN layers in Qwen3Next.

    Requirements:
    - Class must have: embed_dim_, tp_world_size_, alloc_tensor()
    - Class must have MoE config: is_moe, n_routed_experts, num_experts_per_tok, norm_topk_prob
    """

    def _bind_ffn(self):
        """Bind FFN implementation based on MoE configuration."""
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(SharedExpertFFNMixin._ffn_with_shared_expert_ep, self)
            else:
                self._ffn = partial(SharedExpertFFNMixin._ffn_with_shared_expert_tp, self)
        else:
            self._ffn = partial(SharedExpertFFNMixin._standard_ffn, self)
        return

    def _ffn_core(self, input, layer_weight):
        """Core FFN computation: gate_up -> silu_and_mul -> down."""
        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn2_out = layer_weight.shared_expert_down_proj.mm(ffn1_out)
        return ffn2_out, input

    def _standard_ffn(self, input, infer_state, layer_weight):
        """Standard FFN using shared expert weights (non-MoE layers)."""
        ffn2_out, _ = self._ffn_core(input, layer_weight)
        return ffn2_out

    def _compute_shared_expert(self, input, layer_weight):
        """Compute shared expert FFN output with gating."""
        ffn2_out, input_view = self._ffn_core(input, layer_weight)
        return F.sigmoid(layer_weight.shared_expert_gate.mm(input_view)) * ffn2_out, input_view

    def _ffn_with_shared_expert_tp(self, input, infer_state, layer_weight):
        """FFN with shared expert + MoE (tensor parallelism mode)."""
        shared_expert_out, input = self._compute_shared_expert(input, layer_weight)
        moe_out = self._moe_ffn(input, infer_state, layer_weight)
        return shared_expert_out + moe_out

    def _ffn_with_shared_expert_ep(self, input, infer_state, layer_weight):
        """FFN with shared expert + MoE (expert parallelism mode)."""
        shared_expert_out, input = self._compute_shared_expert(input, layer_weight)
        moe_out = self._moe_ffn_edp(input, infer_state, layer_weight)
        return shared_expert_out + moe_out

    def _moe_ffn(self, input, infer_state, layer_weight):
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
