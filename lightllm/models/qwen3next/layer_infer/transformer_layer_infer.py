import torch

import torch.distributed as dist
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextTransformerLayerWeight,
)
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
from lightllm.utils.log_utils import init_logger
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor
from lightllm.common.attention.backends.gdn import GDNLayerConfig, GDNLayerInputs
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
        # Per-rank GDN dimensions for splitting the fused input projection. Set them on every layer:
        # subclasses such as SenseNova Loop re-derive which layers are linear from layer_types.
        if "linear_num_value_heads" in network_config:
            self.tp_linear_num_v_heads = network_config["linear_num_value_heads"] // self.tp_world_size_
            self.linear_head_v_dim = network_config["linear_value_head_dim"]
            self.tp_linear_key_dim = (
                network_config["linear_num_key_heads"] * network_config["linear_key_head_dim"] // self.tp_world_size_
            )
        return

    def _gdn_layer_inputs(self, mixed_qkvzba, infer_state, layer_weight) -> GDNLayerInputs:
        """Split the fused projection and gather the layer's parameters and GDN state views."""
        qkv_dim = self.tp_linear_key_dim * 2 + self.tp_linear_num_v_heads * self.linear_head_v_dim
        z_end = qkv_dim + self.tp_linear_num_v_heads * self.linear_head_v_dim
        b_end = z_end + self.tp_linear_num_v_heads
        state = infer_state.att_state_manager.get_state(
            model_role=infer_state.att_state_role, state_type_filter="gdn", logical_layer_id=self.layer_num_
        )
        conv_states, ssm_states = state.get_layer_view(logical_layer_id=self.layer_num_)
        return GDNLayerInputs(
            mixed_qkv=mixed_qkvzba[:, :qkv_dim],
            z=mixed_qkvzba[:, qkv_dim:z_end].view(-1, self.tp_linear_num_v_heads, self.linear_head_v_dim),
            b=mixed_qkvzba[:, z_end:b_end],
            a=mixed_qkvzba[:, b_end:],
            conv_weight=layer_weight.linear_conv1d.mm_param.weight,
            conv_bias=layer_weight.linear_conv1d.bias,
            A_log=layer_weight.linear_A_log.weight,
            dt_bias=layer_weight.linear_dt_bias.weight,
            conv_states=conv_states,
            ssm_states=ssm_states,
            replay_state=state if self.linear_replay else None,
            layer_num=self.layer_num_,
        )

    def _gdn_attention(self, inputs: GDNLayerInputs, infer_state, *, prefill: bool) -> torch.Tensor:
        z = inputs.z
        run = self.attn.prefill if prefill else self.attn.decode
        return run(
            infer_state.att_step_args[self.attn.group],
            inputs,
            out=self.alloc_tensor(
                (z.shape[0], self.tp_linear_num_v_heads, self.linear_head_v_dim), dtype=z.dtype, device=z.device
            ),
            alloc_func=self.alloc_tensor,
        )

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
            is_prefill=infer_state.is_prefill,
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
        mixed_qkvzba = mixed_qkvzba.contiguous()
        _mixed_qkvzba = tensor_to_no_ref_tensor(mixed_qkvzba)
        pre_capture_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
        pre_capture_graph.__exit__(None, None, None)

        # _gdn_prefill_kernel returns the pre-projection value stream. Its
        # logical size is num_tokens * local value heads * value head dim.
        # We avoid a dry-run because FlashQLA may do host-side syncs while
        # preparing varlen chunk metadata, which is illegal during capture.
        num_tokens = mixed_qkvzba.shape[0]
        o_shape = (num_tokens, self.tp_linear_num_v_heads, self.linear_head_v_dim)
        o_dtype = mixed_qkvzba.dtype
        o_device = mixed_qkvzba.device

        infer_state.prefill_cuda_graph_create_graph_obj()
        infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
        o = torch.empty(o_shape, dtype=o_dtype, device=o_device)
        _o = tensor_to_no_ref_tensor(o)

        def gdn_prefill_func(new_infer_state: "Qwen3NextInferStateInfo"):
            # 通过 new_infer_state.att_step_args 读取 runtime 上 init_att_state() 为本步构造的
            # step args（含 padding 请求的 buffer idx），而不是 capture 时的 step args。
            inputs = self._gdn_layer_inputs(_mixed_qkvzba, new_infer_state, layer_weight)
            tmp_o = self._gdn_attention(inputs, new_infer_state, prefill=True)
            _o.copy_(tmp_o.view(_o.shape))
            return

        infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=gdn_prefill_func, after_graph=pre_capture_graph)
        # The gate is a slice of the captured projection, which replays refill.
        z = mixed_qkvzba[:, 2 * self.tp_linear_key_dim + self.tp_linear_num_v_heads * self.linear_head_v_dim :]
        z = z[:, : self.tp_linear_num_v_heads * self.linear_head_v_dim]
        return o, z.view(-1, self.tp_linear_num_v_heads, self.linear_head_v_dim)

    def init_attention(self, model) -> None:
        if not self.is_linear_attention_layer:
            return super().init_attention(model)
        config, tp_world_size = self.network_config_, self.tp_world_size_
        layer_config = GDNLayerConfig(
            num_k_heads=config["linear_num_key_heads"] // tp_world_size,
            num_v_heads=config["linear_num_value_heads"] // tp_world_size,
            head_k_dim=config["linear_key_head_dim"],
            head_v_dim=config["linear_value_head_dim"],
            conv_kernel_dim=config["linear_conv_kernel_dim"],
            activation=config["hidden_act"],
            dtype=model.data_type,
            state_mtp_step=model.args.mtp_step,
            replay=model.args.ssm_state_mode == "replay",
        )
        # Replay state takes decode history through GDNLayerInputs.replay_state.
        self.linear_replay = layer_config.replay
        self.attn = model.att_groups.create_attention(layer_config, layer_config)

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
            inputs = self._gdn_layer_inputs(mixed_qkvzba, infer_state, layer_weight)
            z = inputs.z
            core_attn_out = self._gdn_attention(inputs, infer_state, prefill=True)

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
        inputs = self._gdn_layer_inputs(mixed_qkvzba, infer_state, layer_weight)
        z = inputs.z
        core_attn_out = self._gdn_attention(inputs, infer_state, prefill=False)
        gdn_out = self._linear_post(core_attn_out, z, layer_weight)

        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return gdn_out

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
