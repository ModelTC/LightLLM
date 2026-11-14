from lightllm.common.attention.api import CacheEncoding, CacheLayout
from lightllm.common.basemodel.attention_adapters import AttentionAdapter


class DeepseekV4Adapter(AttentionAdapter):
    state_type = "swa_"

    def computation_config(self, model, layer, state):
        from lightllm.models.deepseek_v4.config import create_attention_config

        return create_attention_config(
            model.config, tuple(item.layer_num_ for item in model.layers_infer), model.tp_rank_, model.tp_world_size_
        )

    def dimensions(self, model, layer):
        return layer.tp_q_head_num_, 1, layer.head_dim_, layer.v_head_dim

    def cache(self, model, state, states, heads, head_dim, value_dim):
        import torch

        return CacheLayout(
            encoding=CacheEncoding.FP8_SPARSE_MLA,
            page_size=model.args.page_size,
            dtype=torch.uint8,
            key_heads=heads,
            key_dim=head_dim,
            value_dim=value_dim,
            storage_schema=tuple(state.describe_layout() for state in states),
        )

    def scale(self, layer):
        return layer.softmax_scale

    def bind(self, model, layer, weight):
        from lightllm.models.deepseek_v4.attention.sparse_mla import DeepseekV4Resources

        manager, role, layer_id = model.att_state_manager, model.att_state_role, layer.layer_num_
        state = manager.get_state(role, self.state_type, layer_id)
        swa = state.layer_resources(layer_id)
        prefill_swa = model.swa_prefill_workspace[0][state.config.logical_layer_ids.index(layer_id)].view(
            -1, swa.shape[-1]
        )
        compressed = (
            manager.get_state(role, "compressed", layer_id).layer_resources(layer_id) if layer.compress_ratio else None
        )
        return DeepseekV4Resources(
            swa,
            prefill_swa,
            compressed,
            layer.compress_ratio,
            layer.v_head_dim,
            layer.softmax_scale,
            weight.attn_sink_.weight,
        )
