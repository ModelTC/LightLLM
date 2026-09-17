# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import (
    silu_and_mul_fwd,
)
from lightllm.common.basemodel.triton_kernel.mhc import (
    hc_contract,
    hc_post,
    hc_pre_norm,
)
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.models.glm5_next.indexer import Glm5NextNsaInfer


class Glm5NextTransformerLayerInfer(TransformerLayerInferTpl):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.eps_ = network_config["rms_norm_eps"]
        self.embed_dim_ = network_config["hidden_size"]
        self.num_hidden_layers = network_config["num_hidden_layers"]
        self.autotune_layer_num = network_config.get("autotune_layer_num", self.num_hidden_layers)
        self.is_linear_attention_layer = (
            layer_num < self.num_hidden_layers and network_config["layer_types"][layer_num] == "linear_attention"
        )
        self.use_mhc = network_config.get("mhc", True)
        self.mhc_streams = network_config.get("hc_mult", 4)
        self.hc_eps = network_config.get("hc_eps", 1e-6)
        self.hc_sinkhorn_iters = network_config.get("hc_sinkhorn_iters", 20)
        self.swiglu_limit = network_config["swiglu_limit"]
        self.is_moe = (
            network_config["n_routed_experts"] is not None
            and layer_num >= network_config["first_k_dense_replace"]
            and layer_num % network_config.get("moe_layer_freq", 1) == 0
        )
        self.n_shared_experts = network_config["n_shared_experts"]
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.norm_topk_prob = network_config["norm_topk_prob"]
        self.n_group = network_config["n_group"]
        self.topk_group = network_config["topk_group"]
        linear = network_config["linear_attn_config"]
        self.linear_num_heads = linear["num_heads"]
        self.linear_head_dim = linear["head_dim"]
        self.tp_linear_num_heads = self.linear_num_heads // self.tp_world_size_
        self.tp_linear_projection_size = self.tp_linear_num_heads * self.linear_head_dim
        if not self.is_linear_attention_layer:
            self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
            self.qk_nope_head_dim = network_config["qk_nope_head_dim"]
            self.q_lora_rank = network_config["q_lora_rank"]
            self.kv_lora_rank = network_config["kv_lora_rank"]
            self.v_head_dim = network_config["v_head_dim"]
            self.softmax_scale = self.qk_nope_head_dim ** -0.5
            self.indexer = Glm5NextNsaInfer(
                layer_idx=self.layer_num_,
                network_config=self.network_config_,
                tp_world_size=self.tp_world_size_,
            )

    def _att_norm(self, input, infer_state, layer_weight):
        return layer_weight.att_norm_weight_(input=input, eps=self.eps_, alloc_func=self.alloc_tensor)

    def _ffn_norm(self, input, infer_state, layer_weight):
        return layer_weight.ffn_norm_weight_(input=input, eps=self.eps_, alloc_func=self.alloc_tensor)

    def _ffn(self, input, infer_state, layer_weight):
        input = self._tpsp_allgather(input=input.view(-1, self.embed_dim_), infer_state=infer_state)
        if self.is_moe:
            output = self._moe_ffn_tp(input, infer_state, layer_weight)
        else:
            output = self._ffn_tp(input, infer_state, layer_weight)
        return self._tpsp_reduce(input=output, infer_state=infer_state)

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

    def _moe_ffn_tp(self, input, infer_state, layer_weight) -> torch.Tensor:
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        # if fused_shared_experts is not enabled, compute shared_output
        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            shared_output = self._ffn_tp(hidden_states, infer_state, layer_weight)

        moe_gate_dtype = layer_weight.moe_gate.data_type_
        router_logits = layer_weight.moe_gate.mm(hidden_states.to(moe_gate_dtype))
        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
            infer_state=infer_state,
            alpha=1.0,
            limit=self.swiglu_limit,
            clamp_up_add_one=False,
        )

        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            hidden_states.add_(shared_output)

        return hidden_states.view(num_tokens, hidden_dim)

    def _get_qkv(self, input, infer_state, layer_weight):
        if self.is_linear_attention_layer:
            raise AssertionError("KDA projections use _kda_projections")

        input = input.view(-1, self.embed_dim_)
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

    def _context_attention_kernel(self, q, kv, infer_state, layer_weight, out=None):
        q = layer_weight.k_b_proj_.bmm(q.transpose(0, 1)).transpose(0, 1).contiguous()
        topk_mem_indices, topk_indices = self.indexer._get_indices(
            hidden_states=infer_state.get_topk_indices_params["hidden_states"],
            q_lora=infer_state.get_topk_indices_params["q_lora"],
            infer_state=infer_state,
            att_state=infer_state.prefill_att_state,
            layer_weight=layer_weight,
        )
        del infer_state.get_topk_indices_params
        return infer_state.prefill_att_state.prefill_att(
            q=q,
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=AttControl(
                nsa_prefill=True,
                nsa_prefill_dict={
                    "topk_mem_indices": topk_mem_indices,
                    "topk_indices": topk_indices,
                    "prefill_cache_kv": kv,
                    "softmax_scale": self.softmax_scale,
                    "kv_lora_rank": self.kv_lora_rank,
                },
            ),
        )

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

    def _get_o(self, input, infer_state, layer_weight):
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)
        # Both sparse prefill and decode return attention in MLA latent space.
        input = layer_weight.v_b_proj_.bmm(input.transpose(0, 1)).transpose(0, 1)
        output = layer_weight.o_weight_.mm(input.reshape(-1, self.tp_q_head_num_ * self.v_head_dim))
        return self._tpsp_reduce(input=output, infer_state=infer_state)

    def _kda_projections(self, hidden_states, infer_state, layer_weight):
        # Gather sequence-sharded tokens for each rank's KDA heads;
        # _kda_post reduces the output back to the sequence shard.
        hidden_states = hidden_states.view(-1, self.embed_dim_)
        hidden_states = self._tpsp_allgather(input=hidden_states, infer_state=infer_state)
        qkv_gate_proj = layer_weight.linear_qkvbfg_a_proj.mm(hidden_states)
        qkv_dim = 3 * self.tp_linear_projection_size
        qkv, beta_logits, decay_gate_hidden, output_gate_hidden = qkv_gate_proj.split(
            [
                qkv_dim,
                self.tp_linear_num_heads,
                self.linear_head_dim,
                self.linear_head_dim,
            ],
            dim=-1,
        )
        raw_decay_gate, raw_output_gate = layer_weight.project_kda_fg_b(decay_gate_hidden, output_gate_hidden)
        return qkv, raw_decay_gate, beta_logits, raw_output_gate

    def _kda_post(self, core_output, raw_output_gate, infer_state, layer_weight):
        tokens = raw_output_gate.shape[0]
        core_output = core_output.view(-1, self.linear_head_dim)
        raw_output_gate = raw_output_gate.view(tokens, self.tp_linear_num_heads, self.linear_head_dim)
        output = layer_weight.linear_o_norm(
            input=core_output,
            gate_value=raw_output_gate,
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )
        output = layer_weight.linear_o_proj.mm(output.view(tokens, -1))
        return self._tpsp_reduce(input=output, infer_state=infer_state)

    def context_attention_forward(self, input_embeddings, infer_state, layer_weight):
        if not self.is_linear_attention_layer:
            return super().context_attention_forward(input_embeddings, infer_state, layer_weight)
        qkv, raw_decay_gate, beta_logits, raw_output_gate = self._kda_projections(
            input_embeddings, infer_state, layer_weight
        )
        core_output = infer_state.prefill_att_state1.prefill_att(
            q=None,
            k=None,
            v=None,
            att_control=AttControl(
                linear_att_prefill=True,
                linear_att_prefill_dict={
                    "mixed_qkv": qkv,
                    "raw_gate": raw_decay_gate,
                    "raw_beta": beta_logits,
                    "layer_weight": layer_weight,
                    "layer_num": self.layer_num_,
                },
            ),
            alloc_func=self.alloc_tensor,
        )
        return self._kda_post(core_output, raw_output_gate, infer_state, layer_weight)

    def token_attention_forward(self, input_embeddings, infer_state, layer_weight):
        if not self.is_linear_attention_layer:
            return super().token_attention_forward(input_embeddings, infer_state, layer_weight)
        qkv, raw_decay_gate, beta_logits, raw_output_gate = self._kda_projections(
            input_embeddings, infer_state, layer_weight
        )
        core_output = infer_state.decode_att_state1.decode_att(
            q=None,
            k=None,
            v=None,
            att_control=AttControl(
                linear_att_decode=True,
                linear_att_decode_dict={
                    "mixed_qkv": qkv,
                    "raw_gate": raw_decay_gate,
                    "raw_beta": beta_logits,
                    "layer_weight": layer_weight,
                    "layer_num": self.layer_num_,
                },
            ),
            alloc_func=self.alloc_tensor,
        )
        return self._kda_post(core_output, raw_output_gate, infer_state, layer_weight)

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

        layer_input, residual_mix, post_mix = self._hc_pre(streams, layer_weight, "attn", layer_weight.att_norm_weight_)
        if prefill:
            layer_output = self.context_attention_forward(layer_input, infer_state, layer_weight)
        else:
            layer_output = self.token_attention_forward(layer_input, infer_state, layer_weight)
        streams = hc_post(layer_output, streams, residual_mix, post_mix, self.mhc_streams)

        layer_input, residual_mix, post_mix = self._hc_pre(streams, layer_weight, "ffn", layer_weight.ffn_norm_weight_)
        layer_output = self._ffn(layer_input, infer_state, layer_weight)
        streams = hc_post(layer_output, streams, residual_mix, post_mix, self.mhc_streams)
        # Only prefill autotuning truncates the model. Decode autotuning still
        # executes every layer and must keep all residual streams until the end.
        is_autotune_last_layer = (
            prefill and Autotuner.is_autotune_warmup() and self.layer_num_ == self.autotune_layer_num - 1
        )
        if self.layer_num_ == self.num_hidden_layers - 1 or is_autotune_last_layer:
            return hc_contract(streams, self.mhc_streams)
        return streams

    def context_forward(self, input_embeddings, infer_state, layer_weight):
        if not self.use_mhc:
            return super().context_forward(input_embeddings, infer_state, layer_weight)
        return self._forward_mhc(input_embeddings, infer_state, layer_weight, prefill=True)

    def token_forward(self, input_embeddings, infer_state, layer_weight):
        if not self.use_mhc:
            return super().token_forward(input_embeddings, infer_state, layer_weight)
        return self._forward_mhc(input_embeddings, infer_state, layer_weight, prefill=False)
