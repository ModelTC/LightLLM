import torch
import torch.distributed as dist

from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.distributed import all_reduce
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import (
    Deepseek2TransformerLayerInfer,
)
from lightllm.models.deepseek2.triton_kernel.sample_kv import sample_kv
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.kimi_linear.attnres import BlockAttnResConfig, BlockAttnResState
from lightllm.models.kimi_linear.infer_struct import KimiLinearInferStateInfo
from lightllm.models.kimi_linear.layer_weights.transformer_layer_weight import (
    KimiLinearTransformerLayerWeight,
)
from lightllm.models.kimi_linear.mem_manager import KimiLinearMemManager
from lightllm.models.kimi_linear.triton_kernel.kda_prefill_backend import (
    get_kda_prefill_chunk_fn,
)
from lightllm.models.kimi_linear.triton_kernel.fla.ops import (
    fused_kda_gate,
    fused_recurrent_kda_packed_decode,
)
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from lightllm.models.qwen3next.triton_kernel.shared_expert_gate import sigmoid_mul_
from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


class KimiLinearTransformerLayerInfer(Deepseek2TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        self.is_linear_attention_layer = (layer_num + 1) in network_config["linear_attn_config"]["kda_layers"]
        super().__init__(layer_num, network_config)
        self.use_gated_mla = bool(network_config.get("mla_use_output_gate", False))
        self.moe_latent_size = network_config.get("routed_expert_hidden_size")
        if self.moe_latent_size is not None and (
            not isinstance(self.moe_latent_size, int)
            or isinstance(self.moe_latent_size, bool)
            or self.moe_latent_size <= 0
        ):
            raise ValueError("routed_expert_hidden_size must be a positive integer")
        self.latent_moe_use_norm = bool(network_config.get("latent_moe_use_norm", False))
        self.hidden_act = network_config.get("hidden_act", "silu")
        self.activation_situ_beta = network_config.get("activation_situ_beta", 1.0)
        self.activation_situ_linear_beta = network_config.get("activation_situ_linear_beta")
        self.attnres_config = BlockAttnResConfig.from_network_config(network_config)
        self.is_last_layer = layer_num + 1 == network_config["num_hidden_layers"]
        if self.is_linear_attention_layer:
            config = network_config["linear_attn_config"]
            self.kda_head_dim = config["head_dim"]
            self.kda_num_heads = config["num_heads"]
            self.kda_local_num_heads = self.kda_num_heads // self.tp_world_size_
            self.kda_projection_size = self.kda_head_dim * self.kda_local_num_heads
            self.kda_conv_size = config["short_conv_kernel_size"]
            self.kda_ssm_dtype = {
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[get_env_start_args().linear_att_ssm_data_type]
            self.kda_needs_state_conversion = get_llm_data_type() != self.kda_ssm_dtype
            self.kda_gate_lower_bound = config.get("gate_lower_bound")
            self._kda_prefill_chunk = get_kda_prefill_chunk_fn(self.kda_head_dim, get_llm_data_type())

    def _attnres_forward(self, input, infer_state, layer_weight, is_prefill):
        state = infer_state.attnres_state
        if state is None:
            if self.layer_num_ != 0:
                raise RuntimeError("Block AttnRes state must be initialized by the first layer")
            state = BlockAttnResState.from_embedding(input.view(-1, self.embed_dim_), self.attnres_config.block_size)
            infer_state.attnres_state = state

        attn_input = state.mix(
            query=layer_weight.attnres_attn_query.weight,
            norm_weight=layer_weight.attnres_attn_norm.weight,
            eps=self.eps_,
        )
        state.begin_layer(self.layer_num_)
        attn_input = self._att_norm(attn_input, infer_state, layer_weight)
        if is_prefill:
            attn_output = self.context_attention_forward(attn_input, infer_state, layer_weight)
        else:
            attn_output = self.token_attention_forward(attn_input, infer_state, layer_weight)
        state.add_sublayer_output(attn_output.view(-1, self.embed_dim_))

        mlp_input = state.mix(
            query=layer_weight.attnres_mlp_query.weight,
            norm_weight=layer_weight.attnres_mlp_norm.weight,
            eps=self.eps_,
        )
        mlp_input = self._ffn_norm(mlp_input, infer_state, layer_weight)
        mlp_output = self._ffn(mlp_input, infer_state, layer_weight)
        state.add_sublayer_output(mlp_output.view(-1, self.embed_dim_))

        if self.is_last_layer:
            output = state.finish(
                query=layer_weight.attnres_final_query.weight,
                norm_weight=layer_weight.attnres_final_norm.weight,
                eps=self.eps_,
            )
            infer_state.attnres_state = None
            return output
        return state.prefix_sum

    def context_forward(self, input_embdings, infer_state, layer_weight):
        if self.attnres_config is None:
            return super().context_forward(input_embdings, infer_state, layer_weight)
        return self._attnres_forward(input_embdings, infer_state, layer_weight, is_prefill=True)

    def token_forward(self, input_embdings, infer_state, layer_weight):
        if self.attnres_config is None:
            return super().token_forward(input_embdings, infer_state, layer_weight)
        return self._attnres_forward(input_embdings, infer_state, layer_weight, is_prefill=False)

    def context_attention_forward(
        self,
        input_embdings,
        infer_state: KimiLinearInferStateInfo,
        layer_weight: KimiLinearTransformerLayerWeight,
    ):
        if not self.is_linear_attention_layer:
            return super().context_attention_forward(input_embdings, infer_state, layer_weight)
        output = self._kda_forward(input_embdings, infer_state, layer_weight, is_prefill=True)
        if self.tp_world_size_ > 1:
            all_reduce(output, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return output

    def token_attention_forward(
        self,
        input_embdings,
        infer_state: KimiLinearInferStateInfo,
        layer_weight: KimiLinearTransformerLayerWeight,
    ):
        if not self.is_linear_attention_layer:
            return super().token_attention_forward(input_embdings, infer_state, layer_weight)
        assert not getattr(infer_state, "is_mtp_verify", False), "Kimi Linear does not define MTP layers"
        output = self._kda_forward(input_embdings, infer_state, layer_weight, is_prefill=False)
        if self.tp_world_size_ > 1:
            all_reduce(output, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return output

    def _kda_forward(self, input, infer_state, layer_weight, is_prefill):
        assert isinstance(infer_state.mem_manager, KimiLinearMemManager)
        input = self._tpsp_allgather(input.view(-1, self.embed_dim_), infer_state)
        mixed_qkv = layer_weight.kda_qkv_proj.mm(input)
        raw_g = layer_weight.kda_f_b_proj.mm(layer_weight.kda_f_a_proj.mm(input))
        beta = layer_weight.kda_b_proj.mm(input)
        if layer_weight.use_full_rank_gate:
            output_gate = layer_weight.kda_g_proj.mm(input)
        else:
            output_gate = layer_weight.kda_g_b_proj.mm(layer_weight.kda_g_a_proj.mm(input))
        conv_states, ssm_states = infer_state.req_manager.get_mamba_cache(self.layer_num_)

        if is_prefill:
            core_output = self._kda_prefill_wrapper(
                mixed_qkv, raw_g, beta, conv_states, ssm_states, infer_state, layer_weight
            )
        else:
            core_output = self._kda_decode(mixed_qkv, raw_g, beta, conv_states, ssm_states, infer_state, layer_weight)

        core_output = core_output.view(-1, self.kda_head_dim)
        output_gate = output_gate.view(-1, self.kda_head_dim)
        core_output = layer_weight.kda_o_norm(core_output, output_gate, self.eps_)
        return layer_weight.kda_o_proj.mm(core_output.view(-1, self.kda_projection_size))

    def _kda_prefill_wrapper(self, mixed_qkv, raw_g, beta, conv_states, ssm_states, infer_state, layer_weight):
        if not torch.cuda.is_current_stream_capturing():
            return self._kda_prefill(mixed_qkv, raw_g, beta, conv_states, ssm_states, infer_state, layer_weight)

        mixed_qkv = tensor_to_no_ref_tensor(mixed_qkv.contiguous())
        raw_g = tensor_to_no_ref_tensor(raw_g.contiguous())
        beta = tensor_to_no_ref_tensor(beta.contiguous())
        pre_capture_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
        pre_capture_graph.__exit__(None, None, None)
        output_shape = (
            1,
            mixed_qkv.shape[0],
            self.kda_local_num_heads,
            self.kda_head_dim,
        )

        infer_state.prefill_cuda_graph_create_graph_obj()
        infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
        output = tensor_to_no_ref_tensor(torch.empty(output_shape, dtype=mixed_qkv.dtype, device=mixed_qkv.device))

        def kda_prefill_func(new_infer_state):
            new_conv_states, new_ssm_states = new_infer_state.req_manager.get_mamba_cache(self.layer_num_)
            tmp = self._kda_prefill(
                mixed_qkv,
                raw_g,
                beta,
                new_conv_states,
                new_ssm_states,
                new_infer_state,
                layer_weight,
            )
            output.copy_(tmp)

        infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=kda_prefill_func, after_graph=pre_capture_graph)
        return output

    def _kda_prefill(self, mixed_qkv, raw_g, beta, conv_states, ssm_states, infer_state, layer_weight):
        convolved = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer_weight.kda_conv1d.mm_param.weight,
            bias=layer_weight.kda_conv1d.bias,
            query_start_loc=infer_state.b1_cu_q_seq_len,
            cache_indices=infer_state.b_conv_buffer_idx,
            has_initial_state=infer_state.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation="silu",
        ).transpose(0, 1)
        query, key, value = self._split_kda_qkv(convolved, decode=False)
        initial_state = ssm_states[infer_state.b_buffer_idx]
        core_output, final_state = self._kda_prefill_chunk(
            q=query,
            k=key,
            v=value,
            raw_g=raw_g.view(1, -1, self.kda_local_num_heads, self.kda_head_dim),
            beta=beta.view(1, -1, self.kda_local_num_heads),
            A_log=layer_weight.kda_A_log.weight,
            g_bias=layer_weight.kda_dt_bias.weight,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.kda_gate_lower_bound is not None,
            lower_bound=self.kda_gate_lower_bound,
            cu_seqlens=infer_state.b1_cu_q_seq_len,
        )
        if self.kda_needs_state_conversion:
            final_state = final_state.to(self.kda_ssm_dtype, copy=False)
        ssm_states[infer_state.b_buffer_idx] = final_state
        return core_output

    def _kda_decode(self, mixed_qkv, raw_g, beta, conv_states, ssm_states, infer_state, layer_weight):
        convolved = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer_weight.kda_conv1d.mm_param.weight,
            bias=layer_weight.kda_conv1d.bias,
            activation="silu",
            conv_state_indices=infer_state.b_conv_buffer_idx,
        )
        gate = fused_kda_gate(
            raw_g,
            layer_weight.kda_A_log.weight,
            self.kda_head_dim,
            g_bias=layer_weight.kda_dt_bias.weight,
            lower_bound=self.kda_gate_lower_bound,
        ).unsqueeze(1)
        core_output = fused_recurrent_kda_packed_decode(
            mixed_qkv=convolved,
            g=gate,
            beta=beta.float().sigmoid().view(-1, 1, self.kda_local_num_heads),
            initial_state=ssm_states,
            ssm_state_indices=infer_state.b_buffer_idx,
            head_dim=self.kda_head_dim,
            use_qk_l2norm_in_kernel=True,
        )
        return core_output

    def _split_kda_qkv(self, mixed_qkv, decode):
        query, key, value = mixed_qkv.split([self.kda_projection_size] * 3, dim=-1)
        if decode:
            shape = (-1, 1, self.kda_local_num_heads, self.kda_head_dim)
        else:
            shape = (1, -1, self.kda_local_num_heads, self.kda_head_dim)
        return query.view(shape), key.view(shape), value.view(shape)

    def _latent_moe_ffn(
        self,
        input,
        infer_state,
        layer_weight,
        is_prefill=None,
        reduce_tp_output=False,
    ):
        hidden_states = input.view(-1, self.embed_dim_)
        hidden_dim = hidden_states.shape[1]

        shared_output = None
        if self.n_shared_experts is not None:
            shared_output = self._dense_ffn_tp(hidden_states, infer_state, layer_weight)

        moe_gate_dtype = layer_weight.moe_gate.data_type_
        router_logits = layer_weight.moe_gate.mm(hidden_states.to(moe_gate_dtype))
        latent_states = layer_weight.moe_latent_down_proj.mm(hidden_states)
        latent_output = layer_weight.experts.experts(
            latent_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
            is_prefill=is_prefill,
        )
        if reduce_tp_output:
            latent_output = self._tpsp_reduce(input=latent_output, infer_state=infer_state)
        if self.latent_moe_use_norm:
            latent_output = layer_weight.moe_latent_norm(
                input=latent_output,
                eps=self.eps_,
                alloc_func=self.alloc_tensor,
            )
        output = layer_weight.moe_latent_up_proj.mm(latent_output)
        if shared_output is not None:
            if reduce_tp_output:
                shared_output = self._tpsp_reduce(input=shared_output, infer_state=infer_state)
            output.add_(shared_output)
        return output.view(-1, hidden_dim)

    def _dense_ffn_tp(self, input, infer_state, layer_weight):
        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(
            up_gate_out,
            ffn1_out,
            activation=self.hidden_act,
            activation_situ_beta=self.activation_situ_beta,
            activation_situ_linear_beta=self.activation_situ_linear_beta,
        )
        return layer_weight.down_proj.mm(ffn1_out)

    def _ffn_tp(self, input, infer_state, layer_weight):
        return self._dense_ffn_tp(input, infer_state, layer_weight)

    def _ffn_tp_impl(self, input, infer_state, layer_weight):
        if not self.is_moe or self.moe_latent_size is None:
            return super()._ffn_tp_impl(input, infer_state, layer_weight)
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        return self._latent_moe_ffn(
            input,
            infer_state,
            layer_weight,
            reduce_tp_output=True,
        )

    def _moe_ffn_tp(self, input, infer_state, layer_weight):
        if self.moe_latent_size is None:
            return super()._moe_ffn_tp(input, infer_state, layer_weight)
        return self._latent_moe_ffn(
            input,
            infer_state,
            layer_weight,
            reduce_tp_output=True,
        )

    def _moe_ffn_edp(self, input, infer_state, layer_weight):
        if self.moe_latent_size is None:
            return super()._moe_ffn_edp(input, infer_state, layer_weight)
        return self._latent_moe_ffn(input, infer_state, layer_weight, is_prefill=infer_state.is_prefill)

    def overlap_tpsp_token_forward(self, input_embdings, input_embdings1, infer_state, infer_state1, layer_weight):
        if self.attnres_config is not None or (self.is_moe and self.moe_latent_size is not None):
            return LlamaTransformerLayerInfer.overlap_tpsp_token_forward(
                self, input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        return super().overlap_tpsp_token_forward(
            input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
        )

    def overlap_tpsp_context_forward(self, input_embdings, input_embdings1, infer_state, infer_state1, layer_weight):
        if self.attnres_config is not None or (self.is_moe and self.moe_latent_size is not None):
            return LlamaTransformerLayerInfer.overlap_tpsp_context_forward(
                self, input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        return super().overlap_tpsp_context_forward(
            input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
        )

    def _get_qkv(self, input, infer_state, layer_weight):
        input = self._tpsp_allgather(input.view(-1, self.embed_dim_), infer_state)
        mla_output_gate = layer_weight.mla_o_gate_proj.mm(input) if self.use_gated_mla else None
        if self.q_lora_rank is None:
            q = layer_weight.q_weight_.mm(input)
            cache_kv = layer_weight.kv_a_proj_with_mqa_.mm(input)
        else:
            qkv = layer_weight.qkv_a_proj_with_mqa_.mm(input)
            q, cache_kv = qkv.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
            q = layer_weight.q_a_layernorm_(input=q, eps=self.eps_, alloc_func=self.alloc_tensor)
            q = layer_weight.q_b_proj_.mm(q)
        cache_kv = cache_kv.view(-1, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        q = q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim)
        layer_weight.kv_a_layernorm_(
            cache_kv[:, :, : self.kv_lora_rank],
            eps=self.eps_,
            out=cache_kv[:, :, : self.kv_lora_rank],
        )
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
            if mla_output_gate is not None:
                mla_output_gate = infer_state._all_to_all_unbalance_get(data=mla_output_gate)
        infer_state.mla_output_gate = mla_output_gate
        return q, cache_kv

    def _get_o(self, input, infer_state, layer_weight):
        mla_output_gate = infer_state.mla_output_gate if self.use_gated_mla else None
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)
            if mla_output_gate is not None:
                mla_output_gate = infer_state._all_to_all_balance_get(data=mla_output_gate)

        if input.shape[2] == self.kv_lora_rank:
            input = layer_weight.v_b_proj_.bmm(input.transpose(0, 1)).transpose(0, 1)
        input = input.reshape(-1, self.tp_q_head_num_ * self.v_head_dim)
        if mla_output_gate is not None:
            if input.is_cuda:
                sigmoid_mul_(input, mla_output_gate)
            else:
                input.mul_(mla_output_gate.sigmoid())
            infer_state.mla_output_gate = None
        output = layer_weight.o_weight_.mm(input)
        return self._tpsp_reduce(input=output, infer_state=infer_state)

    def _decompress_kv(self, infer_state, layer_weight):
        full_layer_index = infer_state.mem_manager.linear_config.get_full_attention_layer_index(self.layer_num_)
        compressed_kv = infer_state.mem_manager.kv_buffer[full_layer_index]
        total_token_num = infer_state.total_token_num
        sampled_compressed_kv_nope = self.alloc_tensor(
            [total_token_num, 1, layer_weight.kv_lora_rank], dtype=compressed_kv.dtype
        )
        sampled_k_rope = self.alloc_tensor([total_token_num, 1, self.qk_rope_head_dim], dtype=compressed_kv.dtype)
        sample_kv(
            all_compressed_kv=compressed_kv,
            sampled_compressed_kv_nope=sampled_compressed_kv_nope,
            sampled_k_rope=sampled_k_rope,
            b_req_idx=infer_state.b_req_idx,
            req_to_token_indexs=infer_state.req_manager.req_to_token_indexs,
            b_seq_len=infer_state.b_seq_len,
            b_kv_start_loc=infer_state.b1_cu_kv_seq_len[:-1],
            max_kv_seq_len=infer_state.max_kv_seq_len,
        )
        sampled_compressed_kv_nope = sampled_compressed_kv_nope.view(
            total_token_num, layer_weight.kv_lora_rank
        ).contiguous()
        sampled_kv_nope = self.alloc_tensor(
            [total_token_num, self.tp_q_head_num_, self.qk_nope_head_dim + self.v_head_dim],
            dtype=sampled_compressed_kv_nope.dtype,
        )
        layer_weight.cc_kv_b_proj_.mm(
            sampled_compressed_kv_nope,
            out=sampled_kv_nope.view(total_token_num, -1),
        )
        sampled_k_nope, sampled_v = sampled_kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return sampled_k_nope, sampled_k_rope, sampled_v
