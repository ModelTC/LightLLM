# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import (
    Deepseek3_2TransformerLayerInfer,
    NsaInfer,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import (
    silu_and_mul_fwd,
)
from lightllm.models.glm5_next.triton_kernel.mhc import (
    hc_contract,
    hc_expand,
    hc_post,
    hc_pre_norm,
)
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.utils.envs_utils import get_env_start_args


class Glm5NextNsaInfer(NsaInfer):
    """GLM indexer projection without rotary dimensions."""

    def _get_q_k_bf16(self, hidden_states, q_lora, infer_state, layer_weight):
        q = layer_weight.wq_b_proj_.mm(q_lora).view(-1, self.tp_index_n_heads, self.index_head_dim)
        k = layer_weight.wk_proj_.mm(hidden_states.to(q_lora.dtype))
        k = layer_weight.k_norm_(k, eps=self.eps)
        return q, k, k

    def _quantize_indexer_activation(self, value: torch.Tensor):
        from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import (
            hadamard_transform_quant_fp8,
        )

        assert self.block_size == 128 and self.scale_fmt == "ue8m0"
        return hadamard_transform_quant_fp8(value, scale=self.index_head_dim ** -0.5)

    def _scale_indexer_weights(self, weights: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
        from lightllm.models.deepseek3_2.triton_kernel.indexer_weight_scale import (
            scale_indexer_weights_,
        )

        return scale_indexer_weights_(weights, q_scale, self.index_n_heads_scale)

    def _get_indices(self, hidden_states, q_lora, infer_state, att_state, layer_weight):
        # GLM stores weights_proj in FP32, so its activation must match before
        # delegating to the shared NSA scoring and top-k implementation.
        return super()._get_indices(hidden_states.float(), q_lora, infer_state, att_state, layer_weight)


class Glm5NextTransformerLayerInfer(Deepseek3_2TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.num_hidden_layers = network_config["num_hidden_layers"]
        self.autotune_layer_num = network_config.get("autotune_layer_num", self.num_hidden_layers)
        self.is_mtp_layer = layer_num >= self.num_hidden_layers
        self.is_linear_attention_layer = (
            not self.is_mtp_layer and network_config["layer_types"][layer_num] == "linear_attention"
        )
        self.mhc_streams = network_config.get("hc_mult", 4)
        self.hc_eps = network_config.get("hc_eps", 1e-6)
        self.hc_sinkhorn_iters = network_config.get("hc_sinkhorn_iters", 20)
        self.swiglu_limit = network_config["swiglu_limit"]
        linear = network_config["linear_attn_config"]
        self.linear_num_heads = linear["num_heads"]
        self.linear_head_dim = linear["head_dim"]
        self.tp_linear_num_heads = self.linear_num_heads // self.tp_world_size_
        self.tp_linear_projection_size = self.tp_linear_num_heads * self.linear_head_dim
        if not self.is_linear_attention_layer:
            self.indexer = Glm5NextNsaInfer(
                layer_idx=self.layer_num_,
                network_config=self.network_config_,
                tp_world_size=self.tp_world_size_,
            )
            # GLM's recurrent EAGLE drafter processes one row per logical
            # request.  Only target-model decode uses the widened verification
            # layout of mtp_step + 1 rows.
            self.indexer.decode_mtp_step = 0 if self.is_mtp_layer else get_env_start_args().mtp_step

    def _ffn_tp(self, input, infer_state, layer_weight):
        """Dense/shared GLM FFN with the checkpoint's clamp semantics."""

        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(
            up_gate_out,
            ffn1_out,
            limit=self.swiglu_limit,
            alpha=1.0,
            clamp_up_add_one=False,
        )
        return layer_weight.down_proj.mm(ffn1_out)

    def _shared_ffn_tp(self, input, infer_state, layer_weight):
        return self._ffn_tp(input, infer_state, layer_weight)

    def _get_qkv(self, input, infer_state, layer_weight):
        if self.is_linear_attention_layer:
            raise AssertionError("KDA projections use _kda_projections")

        input = input.view(-1, self.embed_dim_)
        if not infer_state.use_replicated_attention_ep:
            input = self._tpsp_allgather(input=input, infer_state=infer_state)
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_unbalance_get(data=input)

        q, cache_kv = layer_weight.qkv_a_proj_with_mqa_.mm(input).split([self.q_lora_rank, self.kv_lora_rank], dim=-1)
        q = rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_)
        infer_state.get_topk_indices_params = {"hidden_states": input, "q_lora": q}
        q = layer_weight.q_b_proj_.mm(q).view(-1, self.tp_q_head_num_, self.qk_nope_head_dim)
        cache_kv = cache_kv.view(-1, 1, self.kv_lora_rank)
        rmsnorm_forward(
            cache_kv[:, :, : self.kv_lora_rank],
            weight=layer_weight.kv_a_layernorm_.weight,
            eps=self.eps_,
            out=cache_kv[:, :, : self.kv_lora_rank],
        )
        return q, cache_kv

    def _get_o(self, input, infer_state, layer_weight):
        if not infer_state.use_replicated_attention_ep:
            return super()._get_o(input, infer_state, layer_weight)

        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)
        if input.shape[2] == self.kv_lora_rank:
            input = layer_weight.v_b_proj_.bmm(input.transpose(0, 1)).transpose(0, 1)
        output = layer_weight.o_weight_.mm(input.reshape(-1, self.tp_q_head_num_ * self.v_head_dim))
        all_reduce(output, group=infer_state.dist_group)
        return output

    def _token_attention_kernel(self, q, infer_state, layer_weight, out=None):
        if self.is_linear_attention_layer:
            raise AssertionError("KDA uses its dedicated backend")
        q_nope = layer_weight.k_b_proj_.bmm(q.transpose(0, 1)).transpose(0, 1)
        topk_mem_indices, _ = self.indexer._get_indices(
            hidden_states=infer_state.get_topk_indices_params["hidden_states"],
            q_lora=infer_state.get_topk_indices_params["q_lora"],
            infer_state=infer_state,
            att_state=infer_state.decode_att_state,
            layer_weight=layer_weight,
        )
        del infer_state.get_topk_indices_params
        q_rope = q_nope[..., :0]
        return infer_state.decode_att_state.decode_att(
            q=(q_nope, q_rope),
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=AttControl(
                nsa_decode=True,
                nsa_decode_dict={
                    "layer_index": self.layer_num_,
                    "topk_mem_indices": topk_mem_indices,
                    "softmax_scale": self.softmax_scale,
                    "kv_lora_rank": self.kv_lora_rank,
                    "qk_rope_head_dim": 0,
                },
            ),
        )

    def _kda_projections(self, input, infer_state, layer_weight):
        # KDA shards heads across TP ranks, so every rank still needs every
        # token before updating its recurrent head state.  In TP/SP mode the
        # layer input is sequence-sharded; gather it here just like the MLA
        # projection path and reduce-scatter the output in _kda_post.
        input = input.view(-1, self.embed_dim_)
        if not infer_state.use_replicated_attention_ep:
            input = self._tpsp_allgather(input=input, infer_state=infer_state)
        projected = layer_weight.linear_qkvbfg_a_proj.mm(input)
        qkv_size = 3 * self.tp_linear_projection_size
        mixed_qkv, raw_beta, f_a, g_a = projected.split(
            [
                qkv_size,
                self.tp_linear_num_heads,
                self.linear_head_dim,
                self.linear_head_dim,
            ],
            dim=-1,
        )
        raw_gate, norm_gate = layer_weight.project_kda_fg_b(f_a, g_a)
        return mixed_qkv, raw_gate, raw_beta, norm_gate

    def _kda_post(self, core_output, norm_gate, infer_state, layer_weight):
        tokens = norm_gate.shape[0]
        core_output = core_output.view(-1, self.linear_head_dim)
        norm_gate = norm_gate.view(tokens, self.tp_linear_num_heads, self.linear_head_dim)
        output = layer_weight.linear_o_norm(
            input=core_output,
            gate_value=norm_gate,
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )
        output = layer_weight.linear_o_proj.mm(output.view(tokens, -1))
        if infer_state.use_replicated_attention_ep:
            all_reduce(output, group=infer_state.dist_group)
            return output
        return self._tpsp_reduce(input=output, infer_state=infer_state)

    def context_attention_forward(self, input_embeddings, infer_state, layer_weight):
        if not self.is_linear_attention_layer:
            return super().context_attention_forward(input_embeddings, infer_state, layer_weight)
        mixed_qkv, raw_gate, raw_beta, norm_gate = self._kda_projections(input_embeddings, infer_state, layer_weight)
        core_output = infer_state.prefill_att_state1.prefill_att(
            q=None,
            k=None,
            v=None,
            att_control=AttControl(
                linear_att_prefill=True,
                linear_att_prefill_dict={
                    "mixed_qkv": mixed_qkv,
                    "raw_gate": raw_gate,
                    "raw_beta": raw_beta,
                    "layer_weight": layer_weight,
                    "layer_num": self.layer_num_,
                },
            ),
            alloc_func=self.alloc_tensor,
        )
        return self._kda_post(core_output, norm_gate, infer_state, layer_weight)

    def token_attention_forward(self, input_embeddings, infer_state, layer_weight):
        if not self.is_linear_attention_layer:
            return super().token_attention_forward(input_embeddings, infer_state, layer_weight)
        mixed_qkv, raw_gate, raw_beta, norm_gate = self._kda_projections(input_embeddings, infer_state, layer_weight)
        core_output = infer_state.decode_att_state1.decode_att(
            q=None,
            k=None,
            v=None,
            att_control=AttControl(
                linear_att_decode=True,
                linear_att_decode_dict={
                    "mixed_qkv": mixed_qkv,
                    "raw_gate": raw_gate,
                    "raw_beta": raw_beta,
                    "layer_weight": layer_weight,
                    "layer_num": self.layer_num_,
                },
            ),
            alloc_func=self.alloc_tensor,
        )
        return self._kda_post(core_output, norm_gate, infer_state, layer_weight)

    def _hc_pre(self, streams, layer_weight, prefix, norm_weight):
        return hc_pre_norm(
            x=streams,
            fn=getattr(layer_weight, f"hc_{prefix}_fn").weight,
            scale=getattr(layer_weight, f"hc_{prefix}_scale").weight,
            base=getattr(layer_weight, f"hc_{prefix}_base").weight,
            norm_weight=norm_weight.weight,
            streams=self.mhc_streams,
            rms_eps=self.eps_,
            norm_eps=self.eps_,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.hc_sinkhorn_iters,
        )

    def _forward_mhc(self, input_embeddings, infer_state, layer_weight, *, prefill):
        streams = input_embeddings
        if self.layer_num_ == 0:
            streams = hc_expand(streams.view(-1, self.embed_dim_), self.mhc_streams)

        layer_input, residual_mix, post_mix = self._hc_pre(streams, layer_weight, "attn", layer_weight.att_norm_weight_)
        if prefill:
            layer_output = self.context_attention_forward(layer_input, infer_state, layer_weight)
        else:
            layer_output = self.token_attention_forward(layer_input, infer_state, layer_weight)
        streams = hc_post(layer_output, streams, residual_mix, post_mix, self.mhc_streams)

        layer_input, residual_mix, post_mix = self._hc_pre(streams, layer_weight, "ffn", layer_weight.ffn_norm_weight_)
        if infer_state.use_replicated_attention_ep:
            if self.is_moe:
                local_input = self._tpsp_sp_split(input=layer_input, infer_state=infer_state)
                local_output = self._ffn(local_input, infer_state, layer_weight)
                layer_output = self._tpsp_allgather(input=local_output, infer_state=infer_state)
            else:
                layer_output = self._ffn_tp(layer_input, infer_state, layer_weight)
                all_reduce(layer_output, group=infer_state.dist_group)
        else:
            layer_output = self._ffn(layer_input, infer_state, layer_weight)
        streams = hc_post(layer_output, streams, residual_mix, post_mix, self.mhc_streams)
        is_autotune_last_layer = Autotuner.is_autotune_warmup() and self.layer_num_ == self.autotune_layer_num - 1
        if self.layer_num_ == self.num_hidden_layers - 1 or is_autotune_last_layer:
            return hc_contract(streams, self.mhc_streams)
        return streams

    def context_forward(self, input_embeddings, infer_state, layer_weight):
        if self.is_mtp_layer:
            return super().context_forward(input_embeddings, infer_state, layer_weight)
        return self._forward_mhc(input_embeddings, infer_state, layer_weight, prefill=True)

    def token_forward(self, input_embeddings, infer_state, layer_weight):
        if self.is_mtp_layer:
            return super().token_forward(input_embeddings, infer_state, layer_weight)
        return self._forward_mhc(input_embeddings, infer_state, layer_weight, prefill=False)
