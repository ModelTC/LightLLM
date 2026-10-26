import torch
from typing import List


class LinearAttentionStateTransfer:
    """Pack live Conv/SSM and optional indexer state into TP-independent PD pages."""

    def __init__(self, linear_state_manager, indexer_state_manager=None):
        self.linear_config = linear_state_manager.linear_config
        self.req_to_conv_state = linear_state_manager.conv_state
        self.req_to_ssm_state = linear_state_manager.ssm_state
        self.mtp_step = linear_state_manager.mtp_step
        self.req_to_indexer_tail = None if indexer_state_manager is None else indexer_state_manager.state
        self.kv_move_buffer = None
        self.global_linear_k_heads = self.linear_config.global_linear_k_heads
        self.global_linear_v_heads = self.linear_config.global_linear_v_heads

        self.global_q_dim = self.global_linear_k_heads * self.linear_config.head_linear_k_dim
        self.global_k_dim = self.global_q_dim
        self.global_v_heads = self.global_linear_v_heads
        self.global_v_dim = self.global_v_heads * self.linear_config.head_linear_v_dim
        # conv state follows mixed_qkv layout: [q, k, v], each as a flat channel block.
        self.conv_shape = (
            self.linear_config.linear_layer_num,
            self.global_q_dim + self.global_k_dim + self.global_v_dim,
            self.linear_config.conv_kernel_size - 1,
        )
        self.ssm_shape = (
            self.linear_config.linear_layer_num,
            self.global_v_heads,
            self.linear_config.head_linear_k_dim,
            self.linear_config.head_linear_v_dim,
        )

        self.conv_nbytes = (
            self.conv_shape[0] * self.conv_shape[1] * self.conv_shape[2] * self.req_to_conv_state.buffer.element_size()
        )
        ssm_alignment = self.req_to_ssm_state.buffer.element_size()
        # 做一下字节对齐，防止切出来的不对齐，导致一些操作的性能下降。
        self.ssm_offset = ((self.conv_nbytes + ssm_alignment - 1) // ssm_alignment) * ssm_alignment
        self.ssm_nbytes = (
            self.ssm_shape[0]
            * self.ssm_shape[1]
            * self.ssm_shape[2]
            * self.ssm_shape[3]
            * self.req_to_ssm_state.buffer.element_size()
        )
        self.state_nbytes = self.ssm_offset + self.ssm_nbytes
        if self.req_to_indexer_tail is not None:
            tail = self.req_to_indexer_tail.buffer
            self.tail_dtype = tail.dtype
            self.tail_shape = (self.linear_config.get_full_att_kv_layer_num_with_draft_model(), *tail.shape[2:])
            self.tail_offset = ((self.state_nbytes + 15) // 16) * 16
            self.tail_nbytes = self.tail_shape[0] * self.tail_shape[1] * self.tail_shape[2] * tail.element_size()
            self.state_nbytes = self.tail_offset + self.tail_nbytes

    def bind_page_buffer(self, kv_move_buffer):
        self.kv_move_buffer = kv_move_buffer
        self.assert_page_size()

    def _view_page_to_tail(self, page_index):
        page_bytes = self.kv_move_buffer[page_index].view(torch.uint8).reshape(-1)
        return (
            page_bytes[self.tail_offset : self.tail_offset + self.tail_nbytes]
            .view(self.tail_dtype)
            .view(self.tail_shape)
        )

    def assert_page_size(self):
        kv_move_buffer = self.kv_move_buffer
        page_nbytes = kv_move_buffer[0].numel() * kv_move_buffer.element_size()
        assert (
            page_nbytes >= self.state_nbytes
        ), f"nixl kv move page bytes {page_nbytes} is smaller than global linear att state bytes {self.state_nbytes}"
        return

    def get_dp_states(self, state_transfers: List["LinearAttentionStateTransfer"], dp_index: int, dp_world_size: int):
        dp_states = state_transfers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        assert len(dp_states) == dp_world_size
        for state in dp_states:
            assert hasattr(state, "req_to_conv_state") and hasattr(state, "req_to_ssm_state")
            assert state.linear_config.linear_layer_num == self.linear_config.linear_layer_num
            assert state.linear_config.conv_kernel_size == self.linear_config.conv_kernel_size
            assert state.linear_config.head_linear_k_dim == self.linear_config.head_linear_k_dim
            assert state.linear_config.head_linear_v_dim == self.linear_config.head_linear_v_dim
            assert state.linear_config.num_linear_k_heads == self.linear_config.num_linear_k_heads
            assert state.linear_config.num_linear_v_heads == self.linear_config.num_linear_v_heads
            assert state.mtp_step == self.mtp_step
            assert (state.req_to_indexer_tail is None) == (self.req_to_indexer_tail is None)
        return dp_states

    def view_page_to_linear_att_state(self, page_index: int):
        page_bytes = self.kv_move_buffer[page_index].view(torch.uint8).reshape(-1)
        conv_page = page_bytes[0 : self.conv_nbytes].view(self.req_to_conv_state.buffer.dtype).view(self.conv_shape)
        ssm_page = (
            page_bytes[self.ssm_offset : self.ssm_offset + self.ssm_nbytes]
            .view(self.req_to_ssm_state.buffer.dtype)
            .view(self.ssm_shape)
        )
        return conv_page, ssm_page

    def write_req_to_page(
        self,
        page_index: int,
        req_idx: int,
        dp_states: List["LinearAttentionStateTransfer"],
    ):
        assert req_idx is not None
        conv_page, ssm_page = self.view_page_to_linear_att_state(page_index)
        for tp_index, state in enumerate(dp_states):
            self._write_one_rank(state, tp_index, req_idx, conv_page, ssm_page)
        if self.req_to_indexer_tail is not None:
            self._view_page_to_tail(page_index).copy_(
                dp_states[0].req_to_indexer_tail.buffer[:, req_idx], non_blocking=True
            )
        return

    def read_page_to_req(
        self,
        page_index: int,
        req_idx: int,
        dp_states: List["LinearAttentionStateTransfer"],
    ):
        assert req_idx is not None
        conv_page, ssm_page = self.view_page_to_linear_att_state(page_index)
        for tp_index, state in enumerate(dp_states):
            self._read_one_rank(state, tp_index, req_idx, conv_page, ssm_page)
        if self.req_to_indexer_tail is not None:
            tail = self._view_page_to_tail(page_index)
            for state in dp_states:
                state.req_to_indexer_tail.buffer[:, req_idx].copy_(tail, non_blocking=True)
        return

    def _get_req_state_indexes(self, req_idx: int):
        mtp_size = self.mtp_step + 1
        # Conv is one widened slot per request; SSM keeps the historical S+1 block layout.
        return req_idx, req_idx * mtp_size

    def _write_one_rank(
        self,
        state: "LinearAttentionStateTransfer",
        tp_index: int,
        req_idx: int,
        conv_page: torch.Tensor,
        ssm_page: torch.Tensor,
    ):
        conv_req_idx, ssm_req_idx = self._get_req_state_indexes(req_idx)
        conv_state = state.req_to_conv_state.buffer[:, conv_req_idx, ..., : self.conv_shape[-1]]
        ssm_state = state.req_to_ssm_state.buffer[:, ssm_req_idx, ...]
        self._copy_conv_state_to_page(conv_state, conv_page, state, tp_index)
        self._copy_ssm_state_to_page(ssm_state, ssm_page, state, tp_index)
        return

    def _copy_conv_state_to_page(
        self,
        conv_state: torch.Tensor,
        conv_page: torch.Tensor,
        state: "LinearAttentionStateTransfer",
        tp_index: int,
    ):
        local_q_heads = state.linear_config.num_linear_k_heads
        local_v_heads = state.linear_config.num_linear_v_heads
        head_k_dim = state.linear_config.head_linear_k_dim
        head_v_dim = state.linear_config.head_linear_v_dim

        local_q_state = conv_state[:, 0 : local_q_heads * head_k_dim, :]
        local_k_state = conv_state[:, local_q_heads * head_k_dim : 2 * local_q_heads * head_k_dim, :]
        local_v_state = conv_state[:, 2 * local_q_heads * head_k_dim :, :]
        global_q_page = conv_page[:, 0 : self.global_q_dim, :]
        global_k_page = conv_page[:, self.global_q_dim : self.global_q_dim + self.global_k_dim, :]
        global_v_page = conv_page[:, self.global_q_dim + self.global_k_dim :, :]

        qk_head_slice = self._get_head_slice(
            tp_index, local_q_heads, self.global_linear_k_heads, state.linear_config.tp_world_size, is_write=True
        )
        if qk_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = qk_head_slice
            local_dim_start = local_head_start * head_k_dim
            local_dim_end = local_head_end * head_k_dim
            global_dim_start = global_head_start * head_k_dim
            global_dim_end = global_head_end * head_k_dim
            global_q_page[:, global_dim_start:global_dim_end, :].copy_(
                local_q_state[:, local_dim_start:local_dim_end, :], non_blocking=True
            )
            global_k_page[:, global_dim_start:global_dim_end, :].copy_(
                local_k_state[:, local_dim_start:local_dim_end, :], non_blocking=True
            )

        v_head_slice = self._get_head_slice(
            tp_index, local_v_heads, self.global_linear_v_heads, state.linear_config.tp_world_size, is_write=True
        )
        if v_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = v_head_slice
            local_dim_start = local_head_start * head_v_dim
            local_dim_end = local_head_end * head_v_dim
            global_dim_start = global_head_start * head_v_dim
            global_dim_end = global_head_end * head_v_dim
            global_v_page[:, global_dim_start:global_dim_end, :].copy_(
                local_v_state[:, local_dim_start:local_dim_end, :], non_blocking=True
            )
        return

    def _copy_ssm_state_to_page(
        self,
        ssm_state: torch.Tensor,
        ssm_page: torch.Tensor,
        state: "LinearAttentionStateTransfer",
        tp_index: int,
    ):
        head_slice = self._get_head_slice(
            tp_index,
            state.linear_config.num_linear_v_heads,
            self.global_linear_v_heads,
            state.linear_config.tp_world_size,
            is_write=True,
        )
        if head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = head_slice
            ssm_page[:, global_head_start:global_head_end, :, :].copy_(
                ssm_state[:, local_head_start:local_head_end, :, :],
                non_blocking=True,
            )
        return

    def _get_head_slice(
        self,
        tp_index: int,
        local_heads: int,
        global_heads: int,
        tp_world_size: int,
        is_write: bool,
    ):
        if local_heads == 0 or global_heads == 0:
            return None
        total_local_heads = local_heads * tp_world_size
        repeat_count = max(1, total_local_heads // global_heads)
        if is_write and repeat_count > 1 and tp_index % repeat_count != 0:
            return None
        unique_tp_index = tp_index // repeat_count
        global_head_start = unique_tp_index * local_heads
        global_head_end = min(global_head_start + local_heads, global_heads)
        local_head_start = 0
        local_head_end = global_head_end - global_head_start
        if local_head_end <= local_head_start:
            return None
        return local_head_start, local_head_end, global_head_start, global_head_end

    def _copy_page_to_conv_state(
        self,
        conv_page: torch.Tensor,
        conv_state: torch.Tensor,
        state: "LinearAttentionStateTransfer",
        tp_index: int,
    ):
        local_q_heads = state.linear_config.num_linear_k_heads
        local_v_heads = state.linear_config.num_linear_v_heads
        head_k_dim = state.linear_config.head_linear_k_dim
        head_v_dim = state.linear_config.head_linear_v_dim

        local_q_state = conv_state[:, 0 : local_q_heads * head_k_dim, :]
        local_k_state = conv_state[:, local_q_heads * head_k_dim : 2 * local_q_heads * head_k_dim, :]
        local_v_state = conv_state[:, 2 * local_q_heads * head_k_dim :, :]
        global_q_page = conv_page[:, 0 : self.global_q_dim, :]
        global_k_page = conv_page[:, self.global_q_dim : self.global_q_dim + self.global_k_dim, :]
        global_v_page = conv_page[:, self.global_q_dim + self.global_k_dim :, :]

        qk_head_slice = self._get_head_slice(
            tp_index, local_q_heads, self.global_linear_k_heads, state.linear_config.tp_world_size, is_write=False
        )
        if qk_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = qk_head_slice
            local_dim_start = local_head_start * head_k_dim
            local_dim_end = local_head_end * head_k_dim
            global_dim_start = global_head_start * head_k_dim
            global_dim_end = global_head_end * head_k_dim
            local_q_state[:, local_dim_start:local_dim_end, :].copy_(
                global_q_page[:, global_dim_start:global_dim_end, :], non_blocking=True
            )
            local_k_state[:, local_dim_start:local_dim_end, :].copy_(
                global_k_page[:, global_dim_start:global_dim_end, :], non_blocking=True
            )

        v_head_slice = self._get_head_slice(
            tp_index, local_v_heads, self.global_linear_v_heads, state.linear_config.tp_world_size, is_write=False
        )
        if v_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = v_head_slice
            local_dim_start = local_head_start * head_v_dim
            local_dim_end = local_head_end * head_v_dim
            global_dim_start = global_head_start * head_v_dim
            global_dim_end = global_head_end * head_v_dim
            local_v_state[:, local_dim_start:local_dim_end, :].copy_(
                global_v_page[:, global_dim_start:global_dim_end, :], non_blocking=True
            )
        return

    def _copy_page_to_ssm_state(
        self,
        ssm_page: torch.Tensor,
        ssm_state: torch.Tensor,
        state: "LinearAttentionStateTransfer",
        tp_index: int,
    ):
        head_slice = self._get_head_slice(
            tp_index,
            state.linear_config.num_linear_v_heads,
            self.global_linear_v_heads,
            state.linear_config.tp_world_size,
            is_write=False,
        )
        if head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = head_slice
            ssm_state[:, local_head_start:local_head_end, :, :].copy_(
                ssm_page[:, global_head_start:global_head_end, :, :],
                non_blocking=True,
            )
        return

    def _read_one_rank(
        self,
        state: "LinearAttentionStateTransfer",
        tp_index: int,
        req_idx: int,
        conv_page: torch.Tensor,
        ssm_page: torch.Tensor,
    ):
        conv_req_idx, ssm_req_idx = self._get_req_state_indexes(req_idx)
        conv_state = state.req_to_conv_state.buffer[:, conv_req_idx, ..., : self.conv_shape[-1]]
        ssm_state = state.req_to_ssm_state.buffer[:, ssm_req_idx, ...]
        self._copy_page_to_conv_state(conv_page, conv_state, state, tp_index)
        self._copy_page_to_ssm_state(ssm_page, ssm_state, state, tp_index)
        return
