"""Model cache encoding at the PD boundary; transport consumes opaque page bytes."""

from types import SimpleNamespace


class KVCachePageIO:
    def __init__(self, mem_managers, device_id, dp_world_size):
        self.mem_managers = mem_managers
        self.mem = mem_managers[device_id]
        self.dp_world_size = dp_world_size

    def pack(self, task):
        self.mem.write_mem_to_page_kv_move_buffer(
            task.mem_indexes,
            task.src_page_index,
            task.prefill_dp_index,
            self.mem_managers,
            self.dp_world_size,
            page_kind=task.page_kind,
            req_idx=task.req_idx,
        )
        page = self.mem.kv_move_buffer[task.src_page_index]
        return page.numel() * page.element_size()

    def unpack(self, task):
        self.mem.read_page_kv_move_buffer_to_mem(
            task.mem_indexes,
            task.dst_page_index,
            task.decode_dp_index,
            self.mem_managers,
            self.dp_world_size,
            page_kind=task.page_kind,
            req_idx=task.req_idx,
        )

    def warmup(self, *, write):
        for dp_index in range(len(self.mem_managers) // self.dp_world_size):
            task = SimpleNamespace(
                mem_indexes=[self.mem.HOLD_TOKEN_MEMINDEXES[0]],
                src_page_index=0,
                dst_page_index=0,
                prefill_dp_index=dp_index,
                decode_dp_index=dp_index,
                page_kind="kv",
                req_idx=self.mem.req_to_token_indexs.shape[0] - 1,
                start_kv_index=0,
            )
            self.pack(task) if write else self.unpack(task)


def create_cache_page_io(mem_managers, device_id, dp_world_size):
    from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager

    page_io_class = KVCachePageIO
    if isinstance(mem_managers[device_id], DeepseekV4MemoryManager):
        from lightllm.models.deepseek_v4.pd_cache import DeepseekV4CachePageIO

        page_io_class = DeepseekV4CachePageIO
    return page_io_class(mem_managers, device_id, dp_world_size)
