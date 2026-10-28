import math

import torch
import triton

from .mem_manager import MemoryManager
from .operator import DeepseekV4MemOperator
from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4CacheConfig, DeepseekV4StateCacheManager
from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_compressed_kv_pages
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name


class DeepseekV4MemoryManager(MemoryManager):
    operator_class = DeepseekV4MemOperator

    def __init__(
        self,
        size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        cache_config: DeepseekV4CacheConfig,
        always_copy=False,
        mem_fraction=0.9,
    ):
        self.cache_config = cache_config
        super().__init__(
            size,
            torch.uint8,
            head_num,
            max(cache_config.compressed_kv_slot_bytes),
            len(cache_config.compressed_kv_slot_bytes),
            always_copy,
            mem_fraction,
        )

    def get_cell_size(self):
        return sum(self.cache_config.compressed_kv_slot_bytes)

    def get_fixed_memory_size(self):
        # Request rings and prefill scratch are allocated before profiling; only HOLD KV remains.
        return self.get_cell_size() * get_env_start_args().page_size

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = [
            torch.zeros((size + self.page_size, head_num, slot_bytes), dtype=dtype, device="cuda")
            for slot_bytes in self.cache_config.compressed_kv_slot_bytes
        ]
        args = get_env_start_args()
        big_page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        # Reserve the same CPU-cache load/offload staging slots as the other hybrid models.
        self.big_page_buffers = DeepseekV4StateCacheManager.create_dp_shared(
            name=f"{get_unique_server_name()}_dsv4_big_pages",
            size=triton.cdiv(size, big_page_tokens) + 2,
            cache_config=self.cache_config,
            keep_num=2,
        )
        from lightllm.utils.dist_utils import get_dp_rank_in_node

        staging_start = self.big_page_buffers.size - self.big_page_buffers.keep_num + 2 * get_dp_rank_in_node()
        self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = staging_start
        self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = staging_start + 1

    def _free_buffers(self):
        self.kv_buffer = self.big_page_buffers = None

    def get_att_input_params(self, layer_index):
        index = self.cache_config.layer_to_compressed_kv_idx.get(layer_index)
        return self.kv_buffer[index] if index is not None else None

    def get_index_kv_buffer(self, index):
        return {f"layer_{layer}": buffer[index].unsqueeze(0) for layer, buffer in enumerate(self.kv_buffer)}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        for layer, buffer in enumerate(self.kv_buffer):
            buffer[index] = load_tensor_dict[f"layer_{layer}"][0].to(buffer.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["big_page_buffers"] = None  # Keep process-local pinned checkpoints out of CUDA IPC.
        return state

    def write_to_shm(self, req_manager):
        # Transport receives GPU views, just as Qwen's mem manager exports linear state.
        for name in (
            "req_to_swa",
            "req_to_c4_state",
            "req_to_c4_indexer_state",
            "req_to_c128_state",
        ):
            setattr(self, name, getattr(req_manager, name))
        return super().write_to_shm(req_manager)

    def get_paged_kv_move_buffer_shape(self, page_num, page_size):
        compressed_kv_bytes = page_size * self.get_cell_size()
        swa_shape, *compressor_shapes = self.cache_config.get_runtime_state_shapes(get_env_start_args().mtp_step)
        state_bytes = (
            math.prod(swa_shape) + sum(math.prod(shape) for shape in compressor_shapes) * torch.float32.itemsize
        )
        return (page_num, 1, 1, 1, max(compressed_kv_bytes, state_bytes))

    def alloc_paged_kv_move_buffer(self, page_num, page_size):
        self.pd_kv_page_size = page_size
        kv_move_buffer = super().alloc_paged_kv_move_buffer(page_num, page_size)
        self.att_state_page_helper = DeepseekV4AttStatePageHelper(self)
        return kv_move_buffer

    def write_mem_to_page_kv_move_buffer(
        self, mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind="kv", req_idx=None
    ):
        if page_kind != "kv":
            assert page_kind == "att_state", f"unknown page_kind={page_kind}"
            dp_mems = mem_managers[dp_index * dp_world_size : (dp_index + 1) * dp_world_size]
            return self.att_state_page_helper.write_req_to_page(page_index, req_idx, dp_mems)
        pin_indexes = self._buffer_mem_indexes_tensors[page_index][: len(mem_indexes)]
        pin_indexes.numpy()[:] = mem_indexes
        page = self.kv_move_buffer[page_index]
        copy_compressed_kv_pages(
            mem_manager=mem_managers[dp_index * dp_world_size],
            mem_indexes=pin_indexes.cuda(non_blocking=True),
            page_indexes=torch.zeros(1, dtype=torch.int32, device=page.device),
            kv_move_buffer=page.view(1, -1),
            to_pages=True,
            page_size=self.pd_kv_page_size,
            round_tail=True,
        )

    def read_page_kv_move_buffer_to_mem(
        self, mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind="kv", req_idx=None
    ):
        if page_kind != "kv":
            assert page_kind == "att_state", f"unknown page_kind={page_kind}"
            dp_mems = mem_managers[dp_index * dp_world_size : (dp_index + 1) * dp_world_size]
            return self.att_state_page_helper.read_page_to_req(page_index, req_idx, dp_mems)
        pin_indexes = self._buffer_mem_indexes_tensors[page_index][: len(mem_indexes)]
        pin_indexes.numpy()[:] = mem_indexes
        indexes = pin_indexes.cuda(non_blocking=True)
        page = self.kv_move_buffer[page_index]
        page_ids = torch.zeros(1, dtype=torch.int32, device=page.device)
        for mem in mem_managers[dp_index * dp_world_size : (dp_index + 1) * dp_world_size]:
            copy_compressed_kv_pages(
                mem_manager=mem,
                mem_indexes=indexes,
                page_indexes=page_ids,
                kv_move_buffer=page.view(1, -1),
                to_pages=False,
                page_size=self.pd_kv_page_size,
                round_tail=True,
            )


class DeepseekV4AttStatePageHelper:
    """整块迁移请求的 SWA 和 compressor ring，保持物理槽位顺序。"""

    def __init__(self, mem_manager):
        self.mem_manager = mem_manager

    def write_req_to_page(self, page_index, req_idx, dp_mems):
        page = self.mem_manager.kv_move_buffer[page_index]
        self._copy_req_state(page, dp_mems[0], req_idx, write=True)

    def read_page_to_req(self, page_index, req_idx, dp_mems):
        page = self.mem_manager.kv_move_buffer[page_index]
        for mem in dp_mems:
            self._copy_req_state(page, mem, req_idx, write=False)

    def _copy_req_state(self, page, mem, req_idx, *, write):
        data = page.reshape(-1)
        offset = 0
        for buffer in (mem.req_to_swa, mem.req_to_c4_state, mem.req_to_c4_indexer_state, mem.req_to_c128_state):
            if buffer is None:
                continue
            state = buffer[:, req_idx]
            nbytes = state.numel() * state.element_size()
            page_state = data[offset : offset + nbytes].view(state.dtype).view(state.shape)
            if write:
                page_state.copy_(state, non_blocking=True)
            else:
                state.copy_(page_state, non_blocking=True)
            offset += nbytes
