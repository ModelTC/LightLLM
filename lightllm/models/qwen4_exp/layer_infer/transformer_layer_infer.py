import torch

from lightllm.common.basemodel.tp_reduce import ordered_tp_sum
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer


class Qwen4ExpTransformerLayerInfer(Qwen35TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        layer_types = network_config.get("layer_types")
        if layer_types is not None:
            self.is_linear_attention_layer = layer_types[layer_num] == "linear_attention"

    def _reduce_linear_output(self, output, infer_state):
        return self._tpsp_reduce(output, infer_state)

    def _moe_ffn_edp(self, input, infer_state, layer_weight):
        mask = infer_state.qwen4_exp_metadata.token_mask
        mask = self._tpsp_sp_split(mask[:, None], infer_state).view(-1)
        return layer_weight.experts.experts(
            input,
            router_logits=layer_weight.moe_gate.mm(input),
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            is_prefill=infer_state.is_prefill,
            infer_state=infer_state,
            token_mask=mask,
            shared_expert_func=lambda: self._compute_shared_expert(input, infer_state, layer_weight),
        )

    def _tpsp_reduce(self, input, infer_state):
        if self.tp_world_size_ == 1:
            return input
        scatter_rank = self.tp_rank_ if get_env_start_args().enable_tpsp_mix_mode else None
        return ordered_tp_sum(input, infer_state.dist_group, self.tp_world_size_, scatter_rank=scatter_rank)

    def _forward(self, residual, infer_state, layer_weight, is_prefill):
        if hasattr(layer_weight, "ple"):
            ple_batch = infer_state.qwen4_exp_ple[self.layer_num_]
            lexical_tokens = getattr(infer_state, "qwen4_exp_lexical_input_ids", infer_state.input_ids)
            lexical = ple_batch.apply(
                self._tpsp_allgather(residual, infer_state),
                lexical_tokens,
                layer_weight,
                self.tp_world_size_,
                infer_state.dist_group,
            )
            auto_commit = getattr(infer_state, "qwen4_exp_auto_commit", None)
            if auto_commit:
                infer_state.qwen4_exp_lexical_state.lexical_caches[self.layer_num_].commit(
                    infer_state.qwen4_exp_metadata, ple_batch
                )
            elif auto_commit is False:
                infer_state.qwen4_exp_lexical_state.stage_speculative_lexical(
                    self.layer_num_, infer_state.qwen4_exp_metadata, ple_batch
                )
            residual = residual + self._tpsp_sp_split(lexical, infer_state)
        attention_residual = layer_weight.attn_hyper_connection.component
        mixed, gates = attention_residual.read(residual)
        if is_prefill:
            update = self.context_attention_forward(mixed, infer_state, layer_weight)
        else:
            update = self.token_attention_forward(mixed, infer_state, layer_weight)
        mlp_residual = layer_weight.mlp_hyper_connection.component
        residual, (mixed, gates) = mlp_residual.combine_and_read(residual, update, gates)
        update = self._ffn(mixed, infer_state, layer_weight)
        return mlp_residual.write(residual, update, gates)

    def context_forward(self, input_embdings, infer_state, layer_weight):
        return self._forward(input_embdings, infer_state, layer_weight, True)

    def token_forward(self, input_embdings, infer_state, layer_weight):
        return self._forward(input_embdings, infer_state, layer_weight, False)

    def _qsa_attention_forward(self, input_embdings, infer_state, layer_weight):
        hidden = self._tpsp_allgather(input_embdings, infer_state)
        index_projection = None
        if (
            not infer_state.is_prefill
            and infer_state.qwen4_exp_fixed_decode
            and hidden.shape[0] <= 128
            and hasattr(layer_weight, "qsa_projection")
        ):
            projected = torch.mm(hidden, layer_weight.qsa_projection.t())
            width = layer_weight.qkvo_gate_proj.mm_param.weight.shape[0]
            qkv_projection, index_projection = projected.split(
                (width, layer_weight.indexer.component.projection.shape[0]), dim=-1
            )
            query, kv = self._project_qkv(hidden, infer_state, layer_weight, qkv_gate_out=qkv_projection)
        else:
            query, kv = self._project_qkv(hidden, infer_state, layer_weight)
        self._post_cache_kv(kv, infer_state, layer_weight)
        from ..attention_state import Qwen4ExpAttentionState

        state = infer_state.att_state_manager.get_state(
            infer_state.att_state_role, "qwen4_gqa", self.layer_num_, Qwen4ExpAttentionState
        )
        key, value = state.get_layer_view(self.layer_num_)
        output = infer_state.qwen4_exp_qsa[self.layer_num_].attend(
            layer_weight.indexer.component,
            hidden,
            query.view(-1, self.tp_q_head_num_, self.head_dim_),
            key,
            value,
            infer_state.qwen4_exp_batch,
            decode=not infer_state.is_prefill,
            index_projection=index_projection,
            attention_splits=None if infer_state.is_prefill else (16 if infer_state.qwen4_exp_fixed_decode else 4),
            max_visible_length=getattr(infer_state, "max_kv_seq_len", None) if infer_state.is_prefill else None,
            tp_world_size=self.tp_world_size_,
            tp_rank=self.tp_rank_,
            dist_group=infer_state.dist_group,
        )
        return self._get_o(output, infer_state, layer_weight)

    def _linear_prefill_cuda_graph_wrapper(self, mixed_qkvzba, infer_state, layer_weight):
        from lightllm.common.basemodel.attention.base_att import AttControl
        from lightllm.common.basemodel.attention.linear.flashinfer import FlashInferLinearAttBackend
        from lightllm.common.basemodel.attention.linear.replay_gdn import ReplayGDNFlashInferAttBackend

        backend = infer_state.prefill_att_state1.backend
        if not isinstance(backend, (FlashInferLinearAttBackend, ReplayGDNFlashInferAttBackend)):
            return super()._linear_prefill_cuda_graph_wrapper(mixed_qkvzba, infer_state, layer_weight)

        mixed_qkvzba = mixed_qkvzba.contiguous()
        captured_input = tensor_to_no_ref_tensor(mixed_qkvzba)
        request_count = infer_state.b_req_idx.numel()

        def prefill(state):
            return state.prefill_att_state1.prefill_att(
                q=None,
                k=None,
                v=None,
                att_control=AttControl(
                    linear_att_prefill=True,
                    linear_att_prefill_dict={
                        "mixed_qkvzba": captured_input,
                        "layer_weight": layer_weight,
                        "layer_num": self.layer_num_,
                    },
                ),
                alloc_func=self.alloc_tensor,
            )

        previous = infer_state.prefill_cuda_graph_get_current_capture_graph()
        previous.__exit__(None, None, None)
        infer_state.prefill_cuda_graph_create_graph_obj()
        infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
        infer_state.prefill_att_state1.init_state()
        output, z = prefill(infer_state)
        output, z = output.contiguous(), z.contiguous()
        infer_state.prefill_cuda_graph_get_current_capture_graph().__exit__(None, None, None)
        graph, _ = infer_state.prefill_cuda_graph_exe_list.pop()
        captured_output = tensor_to_no_ref_tensor(output)
        captured_z = tensor_to_no_ref_tensor(z)

        def attention(state):
            if state.b_req_idx.numel() == request_count:
                graph.replay()
            else:
                result, result_z = prefill(state)
                captured_output.copy_(result)
                captured_z.copy_(result_z)

        infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=attention, after_graph=previous)
        infer_state.prefill_cuda_graph_create_graph_obj()
        infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
        return output, z

    def context_attention_forward(self, input_embdings, infer_state, layer_weight):
        if self.is_linear_attention_layer:
            input_embdings = self._tpsp_allgather(input_embdings, infer_state)
            return super().context_attention_forward(input_embdings, infer_state, layer_weight)
        if torch.cuda.is_current_stream_capturing() and hasattr(infer_state, "prefill_cuda_graph_exe_list"):
            captured_input = tensor_to_no_ref_tensor(input_embdings)
            previous_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
            previous_graph.__exit__(None, None, None)
            infer_state.prefill_cuda_graph_create_graph_obj()
            infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
            output = torch.empty_like(input_embdings)
            captured_output = tensor_to_no_ref_tensor(output)

            def attention(new_infer_state):
                result = self._qsa_attention_forward(captured_input, new_infer_state, layer_weight)
                captured_output.copy_(result)

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=attention, after_graph=previous_graph)
            return output
        return self._qsa_attention_forward(input_embdings, infer_state, layer_weight)

    def token_attention_forward(self, input_embdings, infer_state, layer_weight):
        if self.is_linear_attention_layer:
            input_embdings = self._tpsp_allgather(input_embdings, infer_state)
            return super().token_attention_forward(input_embdings, infer_state, layer_weight)
        return self._qsa_attention_forward(input_embdings, infer_state, layer_weight)
