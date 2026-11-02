from lightllm.common.basemodel.tp_reduce import ordered_tp_sum
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer


class Qwen4ExpTransformerLayerInfer(Qwen35TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        layer_types = network_config.get("layer_types")
        if layer_types is not None:
            self.is_linear_attention_layer = layer_types[layer_num] == "linear_attention"

    def _reduce_linear_output(self, output, infer_state):
        return self._tpsp_reduce(output, infer_state)

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
        residual = attention_residual.write(residual, update, gates)
        mlp_residual = layer_weight.mlp_hyper_connection.component
        mixed, gates = mlp_residual.read(residual)
        update = self._ffn(mixed, infer_state, layer_weight)
        return mlp_residual.write(residual, update, gates)

    def context_forward(self, input_embdings, infer_state, layer_weight):
        return self._forward(input_embdings, infer_state, layer_weight, True)

    def token_forward(self, input_embdings, infer_state, layer_weight):
        return self._forward(input_embdings, infer_state, layer_weight, False)

    def _qsa_attention_forward(self, input_embdings, infer_state, layer_weight):
        query, kv = self._get_qkv(input_embdings, infer_state, layer_weight)
        self._post_cache_kv(kv, infer_state, layer_weight)
        from ..attention_state import Qwen4ExpAttentionState

        state = infer_state.att_state_manager.get_state(
            infer_state.att_state_role, "qwen4_gqa", self.layer_num_, Qwen4ExpAttentionState
        )
        key, value = state.get_layer_view(self.layer_num_)
        hidden = self._tpsp_allgather(input_embdings, infer_state)
        output = infer_state.qwen4_exp_qsa[self.layer_num_].attend(
            layer_weight.indexer.component,
            hidden,
            query.view(-1, self.tp_q_head_num_, self.head_dim_),
            key,
            value,
            infer_state.qwen4_exp_batch,
            attention_splits=32,
        )
        return self._get_o(output, infer_state, layer_weight)

    def context_attention_forward(self, input_embdings, infer_state, layer_weight):
        if self.is_linear_attention_layer:
            input_embdings = self._tpsp_allgather(input_embdings, infer_state)
            return super().context_attention_forward(input_embdings, infer_state, layer_weight)
        return self._qsa_attention_forward(input_embdings, infer_state, layer_weight)

    def token_attention_forward(self, input_embdings, infer_state, layer_weight):
        if self.is_linear_attention_layer:
            input_embdings = self._tpsp_allgather(input_embdings, infer_state)
            return super().token_attention_forward(input_embdings, infer_state, layer_weight)
        return self._qsa_attention_forward(input_embdings, infer_state, layer_weight)
