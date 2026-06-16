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
from typing import Tuple
from lightllm.models.qwen3next.triton_kernel.causal_conv1d import causal_conv1d_fn
from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
from lightllm.models.qwen3next.triton_kernel.gdn_decode_pack import conv_pack_gdn_decode_inputs
from lightllm.models.qwen3next.triton_kernel.shared_expert_gate import add_shared_expert_gate_, sigmoid_mul_
from lightllm.models.qwen3next.triton_kernel.fla.ops import chunk_gated_delta_rule
from lightllm.models.qwen3next.triton_kernel.fla.ops import fused_recurrent_gated_delta_rule
from lightllm.distributed import all_reduce
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
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
        if self.is_linear_attention_layer:
            self._init_linear_layer_metadata(layer_num, network_config)
        return

    def _init_linear_layer_metadata(self, layer_num, network_config):

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

        assert self.num_v_heads % self.num_k_heads == 0, "num_v_heads must be divisible by num_k_heads"
        self.num_v_heads_per_k_head = self.num_v_heads // self.num_k_heads

        # SSM state dtype optimization
        ssm_dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        start_args = get_env_start_args()
        self.ssm_state_dtype = ssm_dtype_dict.get(start_args.linear_att_ssm_data_type, torch.bfloat16)

        # Pre-compute whether dtype conversion is needed
        # GDN kernel output dtype is self.data_type
        # Conversion needed only if SSM state uses different dtype
        self.needs_ssm_dtype_conversion = get_llm_data_type() != self.ssm_state_dtype
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
        gate = layer_weight.ffn_gate.mm(input)
        return shared_expert_out, gate

    def _moe_ffn_tp(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):

        shared_expert_out, gate = self._compute_shared_expert(input, infer_state, layer_weight)

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
            shared_expert_out=shared_expert_out,
            shared_expert_gate=gate,
        )
        hidden_states = hidden_states.view(num_tokens, hidden_dim)
        return hidden_states

    def _moe_ffn_edp(
        self, input: torch.Tensor, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        shared_expert_out, gate = self._compute_shared_expert(input, infer_state, layer_weight)
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
        add_shared_expert_gate_(ep_output, shared_expert_out, gate)
        return ep_output

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        qkvo_gate_proj = getattr(layer_weight, "qkvo_gate_proj", None)
        if qkvo_gate_proj is None:
            qkv_out = layer_weight.qkv_proj.mm(input)
            o_gate = layer_weight._o_gate_proj.mm(input)
        else:
            qkv_gate_out = qkvo_gate_proj.mm(input)
            qkv_out, o_gate = qkv_gate_out.split(
                [
                    self.tp_q_head_num_ * self.head_dim_ * 2
                    + (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_,
                    self.tp_q_head_num_ * self.head_dim_,
                ],
                dim=-1,
            )
        q, cache_kv = qkv_out.split(
            [self.tp_q_head_num_ * self.head_dim_ * 2, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_],
            dim=-1,
        )
        infer_state.gate_value = o_gate
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
        o_tensor = self._get_o_local(input=input, infer_state=infer_state, layer_weight=layer_weight)
        o_tensor = self._tpsp_reduce(input=o_tensor, infer_state=infer_state)
        return o_tensor

    def _get_o_local(
        self,
        input,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ) -> torch.Tensor:
        """Output projection with gating (in-place multiply to save one allocation)."""
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        sigmoid_mul_(input, infer_state.gate_value)
        infer_state.gate_value = None
        o_tensor = layer_weight.o_proj.mm(input)
        return o_tensor

    # ==================== GDN Helper Methods ====================

    def context_attention_forward(
        self,
        input_embdings,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        # full attention layer
        if not self.is_linear_attention_layer:
            return super().context_attention_forward(input_embdings, infer_state, layer_weight)

        gdn_out = self.gdn_forward(input_embdings, infer_state, layer_weight, is_prefill=True)
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
        gdn_out = self.gdn_forward(input_embdings, infer_state, layer_weight, is_prefill=False)
        if self.tp_world_size_ > 1:
            all_reduce(gdn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return gdn_out

    def gdn_forward(
        self,
        input: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
        is_prefill: bool,
    ):
        assert isinstance(infer_state.mem_manager, Qwen3NextMemManager)

        input = input.view(-1, self.embed_dim_)
        mixed_qkvzba = layer_weight.linear_in_proj.mm(input)

        if is_prefill:
            core_attn_out, z = self._gdn_prefill_wrapper_run(mixed_qkvzba, infer_state, layer_weight)
        else:
            mixed_qkv, z, b, a = self._split_qkvzba(mixed_qkvzba)
            conv_states, ssm_states = infer_state.req_manager.get_mamba_cache(self.layer_num_)
            core_attn_out, z = self._gdn_decode_kernel(
                mixed_qkv,
                z,
                conv_states,
                ssm_states,
                a,
                b,
                infer_state,
                layer_weight,
            )

        num_tokens = z.shape[0]
        core_attn_out = core_attn_out.view(-1, core_attn_out.shape[-1])
        z = z.contiguous().view(-1, z.shape[-1])
        norm_out = layer_weight.linear_norm(core_attn_out, z, self.eps_)
        core_attn_out = norm_out.view(num_tokens, -1)
        output = layer_weight.linear_out_proj.mm(core_attn_out)
        return output

    def _gdn_prefill_wrapper_run(
        self,
        mixed_qkvzba: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.cuda.is_current_stream_capturing():
            mixed_qkvzba = mixed_qkvzba.contiguous()
            _mixed_qkvzba = tensor_to_no_ref_tensor(mixed_qkvzba)
            pre_capture_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
            pre_capture_graph.__exit__(None, None, None)

            # _gdn_prefill_kernel returns the pre-projection value stream. Its
            # logical size is num_tokens * local value heads * value head dim.
            # We avoid a dry-run because FlashQLA may do host-side syncs while
            # preparing varlen chunk metadata, which is illegal during capture.
            num_tokens = mixed_qkvzba.shape[0]
            o_shape = (num_tokens, self.tp_num_v_heads, self.head_v_dim)
            o_dtype = mixed_qkvzba.dtype
            o_device = mixed_qkvzba.device
            z_shape = o_shape

            infer_state.prefill_cuda_graph_create_graph_obj()
            infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
            o = torch.empty(o_shape, dtype=o_dtype, device=o_device)
            _o = tensor_to_no_ref_tensor(o)
            z = torch.empty(z_shape, dtype=o_dtype, device=o_device)
            _z = tensor_to_no_ref_tensor(z)

            def gdn_prefill_func(new_infer_state: Qwen3NextInferStateInfo):
                conv_states, ssm_states = new_infer_state.req_manager.get_mamba_cache(self.layer_num_)
                mixed_qkv, tmp_z, b, a = self._split_qkvzba(_mixed_qkvzba)
                _z.copy_(tmp_z)
                tmp_o = self._gdn_prefill_kernel(
                    mixed_qkv, conv_states, ssm_states, a, b, new_infer_state, layer_weight
                )
                tmp_o = tmp_o.view(_o.shape)
                _o.copy_(tmp_o)
                return

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=gdn_prefill_func, after_graph=pre_capture_graph)
            return o, z

        conv_states, ssm_states = infer_state.req_manager.get_mamba_cache(self.layer_num_)
        mixed_qkv, z, b, a = self._split_qkvzba(mixed_qkvzba)
        core_attn_out = self._gdn_prefill_kernel(mixed_qkv, conv_states, ssm_states, a, b, infer_state, layer_weight)
        return core_attn_out, z

    def _split_qkvzba(self, mixed_qkvzba):
        qkv_dim = self.tp_key_dim * 2 + self.tp_value_dim
        z_end = qkv_dim + self.tp_value_dim
        b_end = z_end + self.tp_num_v_heads
        mixed_qkv = mixed_qkvzba[:, :qkv_dim]
        z = mixed_qkvzba[:, qkv_dim:z_end].view(-1, self.tp_num_v_heads, self.head_v_dim)
        b = mixed_qkvzba[:, z_end:b_end]
        a = mixed_qkvzba[:, b_end:]
        return mixed_qkv, z, b, a

    def _rearrange_mixed_qkv(self, mixed_qkv, decode=False):
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

    def _gdn_prefill_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        g, beta = fused_gdn_gating(layer_weight.linear_A_log.weight, a, b, layer_weight.linear_dt_bias.weight)
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
        if self.needs_ssm_dtype_conversion:
            ssm_states[infer_state.b_buffer_idx] = last_recurrent_state.to(self.ssm_state_dtype, copy=False)
        else:
            ssm_states[infer_state.b_buffer_idx] = last_recurrent_state
        return core_attn_out

    def _gdn_decode_kernel(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        infer_state: Qwen3NextInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
    ):
        # Recurrent processing with fused gating. Decode uses a specialized
        # conv+pack kernel to avoid materializing the post-conv qkv tensor
        # before immediately splitting it into q/k/v.
        query, key, value, z, a, b = conv_pack_gdn_decode_inputs(
            mixed_qkv,
            z,
            a,
            b,
            conv_states,
            layer_weight.linear_conv1d.mm_param.weight,
            layer_weight.linear_conv1d.bias,
            infer_state.b_buffer_idx,
            self.activation,
            self.tp_num_k_heads,
            self.head_k_dim,
            self.tp_num_v_heads,
            self.head_v_dim,
        )
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
        return core_attn_out, z
