from lightllm.common.basemodel.moe_route_info_manager import get_moe_capture_callback
import torch

import torch.distributed as dist
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextTransformerLayerWeight,
)
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.attention.base_att import AttControl
from typing import Tuple
from lightllm.models.qwen3next.triton_kernel.shared_expert_gate import sigmoid_mul_
from lightllm.distributed import all_reduce
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.utils.envs_utils import get_env_start_args
from functools import partial
from lightllm.models.qwen3next.config import is_linear_attention_layer

logger = init_logger(__name__)


class Qwen3NextTransformerLayerInfer(LlamaTransformerLayerInfer):
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
        self.is_linear_attention_layer = is_linear_attention_layer(network_config, layer_num)
        return

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_ffn(self):
        if self.is_moe:
            enable_ep_moe = get_env_start_args().enable_ep_moe
            if enable_ep_moe:
                self._ffn = self._ffn_ep_impl
            else:
                self._ffn = self._ffn_tp_impl
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)
        return

    def _ffn_tp_impl(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        ffn2_out = self._moe_ffn_tp(input=input, infer_state=infer_state, layer_weight=layer_weight)
        return self._tpsp_reduce(input=ffn2_out, infer_state=infer_state)

    def _ffn_ep_impl(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ) -> torch.Tensor:
        # ep 本身就是一种 sp 兼容，所以不需要再进行 allgather 和 reduce
        input = input.view(-1, self.embed_dim_)
        return self._moe_ffn_edp(input=input, infer_state=infer_state, layer_weight=layer_weight)

    def _compute_shared_expert(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        input = input.view(-1, self.embed_dim_)
        shared_expert_out = LlamaTransformerLayerInfer._ffn_tp(self, input, infer_state, layer_weight)
        gate = layer_weight.shared_expert_gate.mm(input)
        sigmoid_mul_(shared_expert_out, gate)
        return shared_expert_out

    def _moe_ffn_tp(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape
        router_logits = layer_weight.moe_gate.mm(hidden_states)
        shared_expert_gate = layer_weight.shared_expert_gate.mm(hidden_states)
        if getattr(self, "bound_moe", None) is not None:
            self.bound_moe.run(
                hidden_states,
                router_logits,
                capture=get_moe_capture_callback(infer_state, self.layer_num_),
                shared_gate=shared_expert_gate,
            )
        else:
            layer_weight.experts.experts(
                hidden_states,
                router_logits=router_logits,
                top_k=self.num_experts_per_tok,
                renormalize=self.norm_topk_prob,
                use_grouped_topk=False,
                topk_group=None,
                num_expert_group=None,
                infer_state=infer_state,
                shared_expert_gate=shared_expert_gate,
            )
        hidden_states = hidden_states.view(num_tokens, hidden_dim)
        return hidden_states

    def _moe_ffn_edp(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        shared_expert_out = self._compute_shared_expert(input, infer_state, layer_weight)
        hidden_states = input
        token_num, hidden_dim = hidden_states.shape
        router_logits = layer_weight.moe_gate.mm(hidden_states)
        if getattr(self, "bound_ep", None) is not None:
            ep_output = self.bound_ep.run(
                hidden_states,
                router_logits,
                is_prefill=infer_state.is_prefill,
                capture=get_moe_capture_callback(infer_state, self.layer_num_),
            )
        else:
            ep_output = layer_weight.experts.experts(
                hidden_states,
                router_logits=router_logits,
                top_k=self.num_experts_per_tok,
                renormalize=self.norm_topk_prob,
                use_grouped_topk=False,
                topk_group=None,
                num_expert_group=None,
                is_prefill=infer_state.is_prefill,
                infer_state=infer_state,
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
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        qkv_gate_out = layer_weight.qkvo_gate_proj.mm(input)
        qkv_out, o_gate = qkv_gate_out.split(
            [
                self.tp_q_head_num_ * self.head_dim_ + (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_,
                self.tp_q_head_num_ * self.head_dim_,
            ],
            dim=-1,
        )
        q, cache_kv = qkv_out.split(
            [self.tp_q_head_num_ * self.head_dim_, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_],
            dim=-1,
        )
        infer_state.gate_logics_value = o_gate
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
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv

    def _get_o(
        self,
        input,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ) -> torch.Tensor:
        """Output projection with gating (in-place multiply to save one allocation)."""
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        sigmoid_mul_(input, infer_state.gate_logics_value)
        infer_state.gate_logics_value = None
        o_tensor = layer_weight.o_proj.mm(input)
        o_tensor = self._tpsp_reduce(input=o_tensor, infer_state=infer_state)
        return o_tensor

    def _linear_prefill_cuda_graph_wrapper(
        self,
        mixed_qkvzba: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bound = getattr(self, "bound_prefill_attention", None)
        if bound is not None:
            heads, dim = bound.spec.v_heads, bound.spec.value_dim
        else:
            backend = infer_state.prefill_att_state1.backend
            heads, dim = backend.tp_num_v_heads, backend.head_v_dim

        shape = (mixed_qkvzba.shape[0], heads, dim)
        spec = (shape, mixed_qkvzba.dtype, mixed_qkvzba.device)

        def attention(tensors, batch):
            output, gate = self._gdn_attention(tensors[0], batch, layer_weight, "prefill")
            return output.view(shape), gate

        # GDN preparation can synchronize with the host, so declare its outputs.
        return infer_state.prefill_program.call(attention, (mixed_qkvzba.contiguous(),), infer_state, (spec, spec))

    def context_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        # full attention layer
        if not self.is_linear_attention_layer:
            return super().context_attention_forward(input_embdings, infer_state, layer_weight)

        mixed_qkvzba = self._linear_in_proj(input_embdings, layer_weight)

        if torch.cuda.is_current_stream_capturing():
            core_attn_out, z = self._linear_prefill_cuda_graph_wrapper(mixed_qkvzba, infer_state, layer_weight)
        else:
            core_attn_out, z = self._gdn_attention(mixed_qkvzba, infer_state, layer_weight, "prefill")

        gdn_out = self._linear_post(core_attn_out, z, layer_weight)

        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return gdn_out

    def token_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        if not self.is_linear_attention_layer:
            return super().token_attention_forward(input_embdings, infer_state, layer_weight)

        mixed_qkvzba = self._linear_in_proj(input_embdings, layer_weight)
        core_attn_out, z = self._gdn_attention(mixed_qkvzba, infer_state, layer_weight, "decode")
        gdn_out = self._linear_post(core_attn_out, z, layer_weight)

        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return gdn_out

    def _gdn_attention(self, inputs, batch, weight, phase):
        bound = getattr(self, f"bound_{phase}_attention", None)
        if bound is not None:
            return bound.run(inputs, batch.execution_metadata[bound.metadata_key])
        state = getattr(batch, f"{phase}_att_state1")
        return getattr(state, f"{phase}_att")(
            q=None,
            k=None,
            v=None,
            att_control=AttControl(
                **{
                    f"linear_att_{phase}": True,
                    f"linear_att_{phase}_dict": {
                        "mixed_qkvzba": inputs,
                        "layer_weight": weight,
                        "layer_num": self.layer_num_,
                    },
                }
            ),
            alloc_func=self.alloc_tensor,
        )

    def _linear_in_proj(
        self,
        input: torch.Tensor,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        return layer_weight.linear_in_proj.mm(input)

    def _linear_post(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ) -> torch.Tensor:
        num_tokens = z.shape[0]
        core_attn_out = core_attn_out.view(-1, core_attn_out.shape[-1])
        z = z.contiguous().view(-1, z.shape[-1])
        norm_out = layer_weight.linear_norm(core_attn_out, z, self.eps_)
        core_attn_out = norm_out.view(num_tokens, -1)
        output = layer_weight.linear_out_proj.mm(core_attn_out)
        return output
