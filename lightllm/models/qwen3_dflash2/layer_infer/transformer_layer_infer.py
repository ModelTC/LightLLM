from lightllm.common.att_resources.full_att import FullAttentionState
from lightllm.common.basemodel.attention import AttControl
from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer
from lightllm.models.qwen3_dflash2.triton_kernel import grouped_dynamic_conv


class Qwen3DFlash2TransformerLayerInfer(Qwen3DFlashTransformerLayerInfer):
    """DFlash2 layer with a grouped dynamic convolution around each sublayer."""

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.block_size_ = int(network_config["block_size"])
        self.conv_group_size_ = int(network_config["conv_group_size"])
        self.sliding_window_ = int(network_config.get("sliding_window", 0) or 0)

    def _token_attention_kernel(self, q, infer_state, layer_weight):
        state = infer_state.att_resource_manager.get_att_state(
            model_name=infer_state.att_resource_role,
            att_type="gqa",
            logical_layer_id=self.layer_num_,
            state_class=FullAttentionState,
        )
        k, v = state.get_layer_view(logical_layer_id=self.layer_num_)
        k = k.view(k.shape[0], self.tp_k_head_num_, self.head_dim_)
        v = v.view(v.shape[0], self.tp_k_head_num_, self.head_dim_)
        q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        use_sliding_window = self.sliding_window_ > 0
        # DFlash2 attention is non-causal, so its local window must retain both
        # the visible prefix on the left and the complete draft block on the right.
        window = self.sliding_window_ - 1
        output = infer_state.decode_att_state.decode_att(
            q=q,
            k=k,
            v=v,
            att_control=AttControl(
                use_sliding_window=use_sliding_window,
                sliding_window=(window, window) if use_sliding_window else (-1, -1),
            ),
            alloc_func=self.alloc_tensor,
        )
        return output.view(-1, self.tp_q_head_num_ * self.head_dim_)

    def _post_cache_kv(self, cache_kv, infer_state, layer_weight):
        state = infer_state.att_resource_manager.get_att_state(
            model_name=infer_state.att_resource_role,
            att_type="gqa",
            logical_layer_id=self.layer_num_,
            state_class=FullAttentionState,
        )
        storage_head_num = state.config.head_num
        storage_head_dim = state.config.head_dim
        storage_kv = cache_kv.view(cache_kv.shape[0], 2 * storage_head_num, storage_head_dim)
        return super()._post_cache_kv(storage_kv, infer_state, layer_weight)

    def _run_dynamic_conv(self, hidden, dynamic_weights, base_weight, conv_side):
        return grouped_dynamic_conv(
            hidden=hidden.contiguous(),
            dynamic_weights=dynamic_weights.contiguous(),
            base_kernel=base_weight.weight.contiguous(),
            block_size=self.block_size_,
            group_size=self.conv_group_size_,
            conv_side=conv_side,
        )

    def token_forward(self, input_embdings, infer_state, layer_weight):
        attention_input = self._att_norm(input_embdings, infer_state, layer_weight)
        attention_dynamic_weights = layer_weight.attention_conv_projection_weight_.mm(attention_input)
        attention_input = self._run_dynamic_conv(
            attention_input,
            attention_dynamic_weights,
            layer_weight.attention_conv_base_weight_,
            conv_side=0,
        )
        attention_output = self.token_attention_forward(attention_input, infer_state, layer_weight)
        attention_output = self._run_dynamic_conv(
            attention_output,
            attention_dynamic_weights,
            layer_weight.attention_conv_base_weight_,
            conv_side=1,
        )
        input_embdings.add_(attention_output.view(-1, self.embed_dim_))

        mlp_input = self._ffn_norm(input_embdings, infer_state, layer_weight)
        mlp_dynamic_weights = layer_weight.mlp_conv_projection_weight_.mm(mlp_input)
        mlp_input = self._run_dynamic_conv(
            mlp_input,
            mlp_dynamic_weights,
            layer_weight.mlp_conv_base_weight_,
            conv_side=0,
        )
        mlp_output = self._ffn(mlp_input, infer_state, layer_weight)
        mlp_output = self._run_dynamic_conv(
            mlp_output,
            mlp_dynamic_weights,
            layer_weight.mlp_conv_base_weight_,
            conv_side=1,
        )
        input_embdings.add_(mlp_output.view(-1, self.embed_dim_))
        return input_embdings
