"""Request-local draft windows and their existing checkpoint/PD interfaces."""

import torch


class WindowKVStore:
    """Use the normal [layer, token, head, dim] KV layout for each request's ring.

    Request r owns token slots [r * window, (r + 1) * window). Its committed
    sequence length determines the visible positions; noise KV stays temporary.
    """

    def __init__(self, requests, layers, kv_heads, head_dim, dtype, device, window):
        self.capacity = window
        self.kv_buffer = torch.empty((layers, requests * window, 2 * kv_heads, head_dim), dtype=dtype, device=device)
        self.req_to_seq_len = torch.zeros(requests, dtype=torch.int64, device=device)

    def reset(self):
        self.req_to_seq_len.zero_()

    def reset_req(self, req_idx):
        self.req_to_seq_len[req_idx] = 0

    def save_checkpoint(self, req_idx, buffers, slot):
        start = req_idx * self.capacity
        buffers.kv_buffer[slot].copy_(self.kv_buffer[:, start : start + self.capacity], non_blocking=True)
        buffers.seq_lens[slot : slot + 1].copy_(self.req_to_seq_len[req_idx : req_idx + 1], non_blocking=True)

    def restore_checkpoint(self, req_idx, buffers, slot):
        start = req_idx * self.capacity
        self.kv_buffer[:, start : start + self.capacity].copy_(buffers.kv_buffer[slot], non_blocking=True)
        self.req_to_seq_len[req_idx : req_idx + 1].copy_(buffers.seq_lens[slot : slot + 1], non_blocking=True)
        # The CPU slot can be recycled as soon as prefix matching returns.
        if self.kv_buffer.is_cuda:
            torch.cuda.current_stream(self.kv_buffer.device).synchronize()

    def prepare_write_indices(
        self,
        b_req_idx,
        b_start_loc,
        b_ready_cache_len,
        b_commit_len,
        token_num,
        alloc_func=torch.empty,
    ):
        """Map committed window rows to cache slots without changing the source layout.

        token_num is the source hidden row count; b_start_loc indexes that same layout.
        b_commit_len counts each request's new prompt rows or accepted verify rows.

        This also updates req_to_seq_len before any layer's KV is written. The caller
        must enqueue all layer writes on the same CUDA stream before reading or
        checkpointing the window state; this is not a side-effect-free index query.
        """
        from lightllm.common.basemodel.triton_kernel.windowed_mtp import prepare_window_write_indices

        return prepare_window_write_indices(
            b_req_idx,
            b_start_loc,
            b_ready_cache_len,
            b_commit_len,
            self.req_to_seq_len,
            self.capacity,
            token_num=token_num,
            alloc_func=alloc_func,
        )

    def write(self, layer_index, mem_index, kv):
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        destindex_copy_kv(kv, mem_index, self.kv_buffer[layer_index])

    def pack(self, layer_index, b_req_idx, noise_kv, block_size, alloc_func=torch.empty):
        from lightllm.common.basemodel.triton_kernel.windowed_mtp import pack_window_kv

        return pack_window_kv(
            self.kv_buffer[layer_index],
            noise_kv,
            b_req_idx,
            self.req_to_seq_len,
            self.capacity,
            block_size,
            alloc_func=alloc_func,
        )


class WindowKVPageHelper:
    """Append a request's draft window to the existing PD att_state page.

    The page uses global KV heads, just like ordinary PD KV pages, so P and D
    may use different TP sizes. The ring's physical slot order is preserved.
    """

    def __init__(self, mem_manager, offset=0):
        from lightllm.utils.envs_utils import get_env_start_args

        self.mem_manager = mem_manager
        store = mem_manager.windowed_draft_kv
        args = get_env_start_args()
        tp_world_size = args.tp // args.dp
        layers, _, heads, dim = store.kv_buffer.shape
        window = store.capacity
        self.shape = (window, layers, heads * tp_world_size, dim)
        self.kv_offset = (offset + 7) // 8 * 8
        self.kv_nbytes = window * layers * heads * tp_world_size * dim * store.kv_buffer.element_size()
        self.seq_len_offset = (self.kv_offset + self.kv_nbytes + 7) // 8 * 8
        self.state_nbytes = self.seq_len_offset + 8

    def assert_page_size(self):
        page = self.mem_manager.kv_move_buffer[0]
        page_nbytes = page.numel() * page.element_size()
        assert page_nbytes >= self.state_nbytes, (
            f"PD page bytes {page_nbytes} is smaller than attention state bytes {self.state_nbytes}; "
            "increase --pd_kv_page_size on both P and D nodes"
        )

    def copy_req_page(self, page_index, req_idx, dp_mems, mode):
        from lightllm.common.kv_trans_kernel.nixl_kv_trans import page_io

        page = self.mem_manager.kv_move_buffer[page_index].view(torch.uint8).reshape(-1)
        kv_page = (
            page[self.kv_offset : self.kv_offset + self.kv_nbytes]
            .view(self.mem_manager.windowed_draft_kv.kv_buffer.dtype)
            .view(self.shape)
        )
        seq_len_page = page[self.seq_len_offset : self.seq_len_offset + 8].view(torch.int64)
        window = self.shape[0]
        indexes = torch.arange(req_idx * window, (req_idx + 1) * window, dtype=torch.int64, device=page.device)
        for tp_index, mem in enumerate(dp_mems):
            store = mem.windowed_draft_kv
            page_io(
                mem_indexes=indexes,
                page_tensor=kv_page,
                kv_buffer=store.kv_buffer,
                tp_index=tp_index,
                tp_world_size=len(dp_mems),
                mode=mode,
            )
            if mode == "read":
                store.req_to_seq_len[req_idx : req_idx + 1].copy_(seq_len_page, non_blocking=True)
            elif tp_index == 0:
                seq_len_page.copy_(store.req_to_seq_len[req_idx : req_idx + 1], non_blocking=True)
