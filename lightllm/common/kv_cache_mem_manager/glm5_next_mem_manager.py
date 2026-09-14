import torch

from lightllm.common.kv_cache_mem_manager.operator import LinearAttMemOperator
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import (
    Qwen3NextMemManager,
    Qwen3NextLinearAttPageHelper,
)
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.common.kv_trans_kernel.nixl_kv_trans import mla_page_io


class Glm5NextMemOperator(LinearAttMemOperator):
    def copy_kv_to_mem_manager(self, layer_index, mem_index, kv):
        output = self.mem_manager.get_att_input_params(layer_index)[:, :, : kv.shape[-1]]
        destindex_copy_kv(kv, mem_index, output)


class Glm5NextMemManager(Qwen3NextMemManager):
    """One packed token buffer; KDA uses the standard big/small page pools."""

    operator_class = Glm5NextMemOperator

    def __init__(self, *args, mla_head_dim=512, **kwargs):
        self.mla_head_dim = mla_head_dim
        super().__init__(*args, **kwargs)

    def get_cell_size(self):
        return self.head_dim * self.dtype.itemsize * self.layer_num

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        assert head_num == 1
        self.kv_buffer = torch.empty((layer_num, size + 1, 1, head_dim), dtype=dtype, device="cuda")
        self._init_linear_att_buffers()

    def _layer_buffer(self, layer_index):
        return self.kv_buffer[self.linear_config.get_full_att_kv_layer_index(layer_index)]

    def get_att_input_params(self, layer_index):
        return self._layer_buffer(layer_index)[:, :, : self.mla_head_dim]

    def get_indexer_k_buffer(self, layer_index):
        return self._layer_buffer(layer_index).view(torch.uint8)[:, :, -132:]

    def write_to_shm(self, req_manager):
        self.req_to_indexer_tail = req_manager.req_to_indexer_tail
        return super().write_to_shm(req_manager)

    def _create_att_state_page_helper(self):
        return Glm5NextAttStatePageHelper(self)

    def get_paged_kv_move_buffer_shape(self, page_num, page_size):
        # Packed MLA/index KV is replicated across TP ranks.
        return (page_num, page_size, self.layer_num, self.head_num, self.head_dim)

    def write_mem_to_page_kv_move_buffer(
        self, mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind="kv", req_idx=None
    ):
        if page_kind != "kv":
            return super().write_mem_to_page_kv_move_buffer(
                mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind, req_idx
            )
        pin_indexes = self._buffer_mem_indexes_tensors[page_index][: len(mem_indexes)]
        pin_indexes.numpy()[:] = mem_indexes
        mla_page_io(
            mem_indexes=pin_indexes.cuda(non_blocking=True),
            page_tensor=self.kv_move_buffer[page_index],
            kv_buffer=mem_managers[dp_index * dp_world_size].kv_buffer,
            mode="write",
        )

    def read_page_kv_move_buffer_to_mem(
        self, mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind="kv", req_idx=None
    ):
        if page_kind != "kv":
            return super().read_page_kv_move_buffer_to_mem(
                mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind, req_idx
            )
        pin_indexes = self._buffer_mem_indexes_tensors[page_index][: len(mem_indexes)]
        pin_indexes.numpy()[:] = mem_indexes
        indexes = pin_indexes.cuda(non_blocking=True)
        for mem in mem_managers[dp_index * dp_world_size : (dp_index + 1) * dp_world_size]:
            mla_page_io(
                mem_indexes=indexes,
                page_tensor=self.kv_move_buffer[page_index],
                kv_buffer=mem.kv_buffer,
                mode="read",
            )


class Glm5NextAttStatePageHelper(Qwen3NextLinearAttPageHelper):
    """Append the replicated K-pool tail to the global Conv/SSM state page."""

    def __init__(self, mem_manager):
        super().__init__(mem_manager)
        self.tail_dtype = mem_manager.req_to_indexer_tail.buffer.dtype
        self.tail_shape = (
            self.linear_config.get_main_model_full_att_layer_num(),
            self.linear_config.index_kpool,
            2 * self.linear_config.index_head_dim,
        )
        self.tail_offset = ((self.state_nbytes + 15) // 16) * 16
        self.tail_nbytes = self.tail_shape[0] * self.tail_shape[1] * self.tail_shape[2] * self.tail_dtype.itemsize
        self.state_nbytes = self.tail_offset + self.tail_nbytes

    def _view_page_to_tail(self, page_index):
        page_bytes = self.mem_manager.kv_move_buffer[page_index].view(torch.uint8).reshape(-1)
        return (
            page_bytes[self.tail_offset : self.tail_offset + self.tail_nbytes]
            .view(self.tail_dtype)
            .view(self.tail_shape)
        )

    def write_req_to_page(self, page_index, req_idx, dp_mems):
        super().write_req_to_page(page_index, req_idx, dp_mems)
        # All four slots travel together; the existing sequence length determines
        # which 0-3 entries are live. No additional PD metadata is needed.
        self._view_page_to_tail(page_index).copy_(dp_mems[0].req_to_indexer_tail.buffer[:, req_idx], non_blocking=True)

    def read_page_to_req(self, page_index, req_idx, dp_mems):
        super().read_page_to_req(page_index, req_idx, dp_mems)
        tail = self._view_page_to_tail(page_index)
        for mem in dp_mems:
            mem.req_to_indexer_tail.buffer[:, req_idx].copy_(tail, non_blocking=True)
