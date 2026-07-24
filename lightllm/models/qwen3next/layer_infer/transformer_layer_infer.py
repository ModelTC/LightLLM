import torch

import torch.distributed as dist
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextTransformerLayerWeight,
)
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
from lightllm.utils.log_utils import init_logger
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor
from lightllm.common.kv_cache_mem_manager import Qwen3NextMemManager
from lightllm.common.basemodel.attention.base_att import AttControl
from typing import Tuple
from lightllm.models.qwen3next.triton_kernel.shared_expert_gate import sigmoid_mul_
from lightllm.distributed import all_reduce
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.utils.envs_utils import get_env_start_args
from functools import partial

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
        num_full_attention_layers = network_config["full_attention_interval"]
        self.is_linear_attention_layer = (layer_num + 1) % num_full_attention_layers != 0
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
                self.tp_q_head_num_ * self.head_dim_ * 2 + (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_,
                self.tp_q_head_num_ * self.head_dim_,
            ],
            dim=-1,
        )
        q, cache_kv = qkv_out.split(
            [self.tp_q_head_num_ * self.head_dim_ * 2, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_],
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

        backend = infer_state.prefill_att_state1.backend

        mixed_qkvzba = mixed_qkvzba.contiguous()
        _mixed_qkvzba = tensor_to_no_ref_tensor(mixed_qkvzba)
        pre_capture_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
        pre_capture_graph.__exit__(None, None, None)

        # _gdn_prefill_kernel returns the pre-projection value stream. Its
        # logical size is num_tokens * local value heads * value head dim.
        # We avoid a dry-run because FlashQLA may do host-side syncs while
        # preparing varlen chunk metadata, which is illegal during capture.
        num_tokens = mixed_qkvzba.shape[0]
        o_shape = (num_tokens, backend.tp_num_v_heads, backend.head_v_dim)
        o_dtype = mixed_qkvzba.dtype
        o_device = mixed_qkvzba.device
        z_shape = o_shape

        infer_state.prefill_cuda_graph_create_graph_obj()
        infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
        o = torch.empty(o_shape, dtype=o_dtype, device=o_device)
        _o = tensor_to_no_ref_tensor(o)
        z = torch.empty(z_shape, dtype=o_dtype, device=o_device)
        _z = tensor_to_no_ref_tensor(z)

        def gdn_prefill_func(new_infer_state: "Qwen3NextInferStateInfo"):
            # 与 Llama/FA3 一致：通过 new_infer_state.prefill_att_state1 进入，
            # 使用 runtime 上 init_att_state() 更新过的 buffer idx，而不是
            # capture 时闭包住的 graph prefill_att_state1。
            tmp_o, tmp_z = new_infer_state.prefill_att_state1.prefill_att(
                q=None,
                k=None,
                v=None,
                att_control=AttControl(linear_att_prefill=True, linear_att_prefill_dict={"mixed_qkvzba": _mixed_qkvzba, "layer_weight": layer_weight, "layer_num": self.layer_num_}),
                alloc_func=self.alloc_tensor,
            )
            tmp_o = tmp_o.view(_o.shape)
            _o.copy_(tmp_o)
            _z.copy_(tmp_z)
            return

        infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=gdn_prefill_func, after_graph=pre_capture_graph)
        return o, z

    def context_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        # full attention layer
        if not self.is_linear_attention_layer:
            return super().context_attention_forward(input_embdings, infer_state, layer_weight)

        assert isinstance(infer_state.mem_manager, Qwen3NextMemManager)
        mixed_qkvzba = self._linear_in_proj(input_embdings, layer_weight)

        if torch.cuda.is_current_stream_capturing():
            core_attn_out, z = self._linear_prefill_cuda_graph_wrapper(mixed_qkvzba, infer_state, layer_weight)
        else:
            core_attn_out, z = infer_state.prefill_att_state1.prefill_att(
                q=None, k=None, v=None,
                att_control=AttControl(linear_att_prefill=True, linear_att_prefill_dict={"mixed_qkvzba": mixed_qkvzba, "layer_weight": layer_weight, "layer_num": self.layer_num_}),
                alloc_func=self.alloc_tensor,
            )

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

        assert isinstance(infer_state.mem_manager, Qwen3NextMemManager)
        mixed_qkvzba = self._linear_in_proj(input_embdings, layer_weight)
        core_attn_out, z = infer_state.decode_att_state1.decode_att(
            q=None, k=None, v=None,
            att_control=AttControl(linear_att_decode=True, linear_att_decode_dict={"mixed_qkvzba": mixed_qkvzba, "layer_weight": layer_weight, "layer_num": self.layer_num_}),
            alloc_func=self.alloc_tensor,
        )
        gdn_out = self._linear_post(core_attn_out, z, layer_weight)

        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return gdn_out

    def _linear_in_proj(
        self, input: torch.Tensor, layer_weight: Qwen3NextTransformerLayerWeight,
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
