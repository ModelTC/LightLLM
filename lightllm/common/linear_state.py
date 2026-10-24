import torch

from lightllm.common.state_cache_manager import LayerCache


class LinearStateManager:
    """Manage request-local convolution and SSM state with native MTP slots."""

    def __init__(self, max_request_num, linear_config, args):
        self.linear_config = linear_config
        self.mtp_step = args.mtp_step
        self.mtp_state_index = (
            torch.zeros((max_request_num + 1,), dtype=torch.int32, device="cuda") if self.mtp_step > 0 else None
        )
        self.conv_state = LayerCache(
            size=max_request_num + 1,
            dtype=linear_config.conv_state_dtype,
            shape=linear_config.get_mtp_conv_state_shape(self.mtp_step),
            layer_num=linear_config.linear_layer_num,
            device="cuda",
        )
        self.ssm_state = LayerCache(
            size=(max_request_num + 1) * (self.mtp_step + 1),
            dtype=linear_config.ssm_state_dtype,
            shape=linear_config.get_ssm_state_shape(),
            layer_num=linear_config.linear_layer_num,
            device="cuda",
        )

    def reset(self, req_idx):
        ssm_start = req_idx * (self.mtp_step + 1)
        self.conv_state.buffer[:, req_idx].zero_()
        self.ssm_state.buffer[:, ssm_start : ssm_start + self.mtp_step + 1].zero_()
        if self.mtp_state_index is not None:
            self.mtp_state_index[req_idx] = 0

    def get_layer_buffers(self, layer_idx_in_all):
        assert 0 <= layer_idx_in_all < self.linear_config.all_layer_num
        layer_idx = layer_idx_in_all - layer_idx_in_all // self.linear_config.full_attention_interval
        return self.conv_state.buffer[layer_idx], self.ssm_state.buffer[layer_idx]

    def copy_state_to(self, req_idx, conv_dst, ssm_dst):
        conv_width = self.linear_config.get_conv_state_shape()[-1]
        conv_dst.copy_(self.conv_state.buffer[:, req_idx, ..., :conv_width], non_blocking=True)
        ssm_dst.copy_(self.ssm_state.buffer[:, req_idx * (self.mtp_step + 1)], non_blocking=True)

    def save_checkpoints(self, b_req_idx, req_indexes, buffer_indexes, state_cache_manager, checkpoint_lens=None):
        from lightllm.common.basemodel.triton_kernel.linear_att_copy import copy_linear_att_state_to_kv_buffer

        buffer_indexes = torch.tensor(buffer_indexes, dtype=torch.int32, device="cpu").cuda(non_blocking=True)
        copy_linear_att_state_to_kv_buffer(
            b_req_idx=b_req_idx,
            big_page_buffer_ids=buffer_indexes,
            gpu_conv_state=self.conv_state.buffer,
            gpu_ssm_state=self.ssm_state.buffer,
            cpu_kv_conv_state=state_cache_manager.conv_state_cache.buffer,
            cpu_kv_ssm_state=state_cache_manager.ssm_state_cache.buffer,
            mtp_step=self.mtp_step,
        )

    def restore(self, req_idx, conv_state, ssm_state):
        self.conv_state.buffer[:, req_idx, ..., : conv_state.shape[-1]] = conv_state
        self.ssm_state.buffer[:, req_idx * (self.mtp_step + 1)] = ssm_state
        if self.mtp_state_index is not None:
            self.mtp_state_index[req_idx] = 0

    def commit_accepted_state(self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width):
        from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update

        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=self.mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=verify_width,
        )

    capture_intermediate_checkpoints = False

    def create_checkpoint_pool(self, size, keep_num=0):
        from lightllm.common.state_cache_manager import LinearAttCacheManager

        return LinearAttCacheManager(size, self.linear_config, keep_num=keep_num)

    def create_cpu_transfer(self, mem_manager, big_page_buffers):
        from lightllm.common.hybrid_cache_transfer import HybridCpuCacheTransfer

        return HybridCpuCacheTransfer(mem_manager.kv_buffer, self.linear_config, big_page_buffers)

    def create_state_transfer(self, indexer_state_manager=None):
        from lightllm.common.linear_state_transfer import LinearAttentionStateTransfer

        return LinearAttentionStateTransfer(self, indexer_state_manager)

    def save_state(self, req_idx, buffer_idx, state_cache_manager, checkpoint_len):
        self.copy_state_to(req_idx, *state_cache_manager.get_state_cache(buffer_idx))

    def restore_checkpoint(self, req_idx, restored_len, state_cache_manager, buffer_idx):
        self.restore(req_idx, *state_cache_manager.get_state_cache(buffer_idx))

    def restore_peer_state(self, req_idx, restored_len, peer_state, peer_req_idx):
        conv_width = self.linear_config.get_conv_state_shape()[-1]
        self.restore(
            req_idx,
            peer_state.req_to_conv_state.buffer[:, peer_req_idx, ..., :conv_width],
            peer_state.req_to_ssm_state.buffer[:, peer_req_idx * (peer_state.mtp_step + 1)],
        )
