from lightllm.common.state_cache_manager import LayerCache


class IndexerStateManager:
    def __init__(self, max_request_num, linear_config, mtp_step, device="cuda"):
        self.linear_config = linear_config
        self.state = LayerCache(
            size=max_request_num + 1,
            dtype=linear_config.full_att_dtype,
            shape=(linear_config.index_kpool + mtp_step, 2 * linear_config.index_head_dim),
            layer_num=linear_config.get_full_att_kv_layer_num_with_draft_model(),
            device=device,
        )

    def get_layer_buffer(self, layer_index):
        return self.state.buffer[self.linear_config.get_full_att_kv_layer_index(layer_index)]

    def reset(self, req_idx):
        self.state.buffer[:, req_idx].zero_()
