from types import SimpleNamespace

from lightllm.common.kv_page_transfer import KVPageTransfer


class KVCachePageIO:
    def __init__(self, kv_views, state_transfers, device_id, dp_world_size):
        self.kv_views = kv_views
        self.state_transfers = state_transfers
        self.device_id = device_id
        self.dp_world_size = dp_world_size
        self.transfer = KVPageTransfer(kv_views[device_id])

    def alloc_buffer(self, page_num, page_size):
        self.buffer = self.transfer.alloc_paged_kv_move_buffer(page_num, page_size)
        helper = self.state_transfers[self.device_id]
        if helper is not None:
            helper.bind_page_buffer(self.buffer)
        return self.buffer

    def pack(self, task):
        if task.page_kind == "att_state":
            helper = self.state_transfers[self.device_id]
            states = helper.get_dp_states(self.state_transfers, task.prefill_dp_index, self.dp_world_size)
            helper.write_req_to_page(task.src_page_index, task.req_idx, states)
        else:
            assert task.page_kind == "kv"
            self.transfer.write_mem_to_page_kv_move_buffer(
                task.mem_indexes, task.src_page_index, task.prefill_dp_index, self.kv_views, self.dp_world_size
            )
        return self.buffer[task.src_page_index].numel() * self.buffer.element_size()

    def unpack(self, task):
        if task.page_kind == "att_state":
            helper = self.state_transfers[self.device_id]
            states = helper.get_dp_states(self.state_transfers, task.decode_dp_index, self.dp_world_size)
            helper.read_page_to_req(task.dst_page_index, task.req_idx, states)
        else:
            assert task.page_kind == "kv"
            self.transfer.read_page_kv_move_buffer_to_mem(
                task.mem_indexes, task.dst_page_index, task.decode_dp_index, self.kv_views, self.dp_world_size
            )

    def warmup(self, *, write):
        for dp_index in range(len(self.kv_views) // self.dp_world_size):
            task = SimpleNamespace(
                mem_indexes=[0],
                src_page_index=0,
                dst_page_index=0,
                prefill_dp_index=dp_index,
                decode_dp_index=dp_index,
                page_kind="kv",
            )
            self.pack(task) if write else self.unpack(task)


def create_cache_page_io(kv_views, state_transfers, device_id, dp_world_size):
    return kv_views[device_id].create_page_io(kv_views, state_transfers, device_id, dp_world_size)
