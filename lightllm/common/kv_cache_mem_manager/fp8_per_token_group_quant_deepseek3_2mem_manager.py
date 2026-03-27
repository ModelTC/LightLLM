import torch
from typing import Any, List

from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.distributed.pynccl import PyNcclCommunicator
from lightllm.common.kv_trans_kernel.kv_trans import kv_trans
from lightllm.common.kv_trans_kernel.nixl_kv_trans import mla_page_io

from .deepseek2_mem_manager import Deepseek2MemoryManager


class FP8PerTokenGroupQuantDeepseek3_2MemoryManager(Deepseek2MemoryManager):
    flashmla_bytes_per_token = 656
    indexer_bytes_per_token = 132
    kv_head_dim = 576
    kv_nope_dim = 512
    kv_rope_dim = 64
    quant_group_size = 128
    quant_group_num = kv_nope_dim // quant_group_size

    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        assert head_num == 1, "DeepSeek-V3.2 DSA FP8 path expects MQA-style head_num == 1"
        self.prefill_dtype = dtype
        super().__init__(
            size, torch.uint8, head_num, self.flashmla_bytes_per_token, layer_num, always_copy, mem_fraction
        )

    def get_cell_size(self):
        return self.layer_num * (self.flashmla_bytes_per_token + self.indexer_bytes_per_token)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = torch.empty(
            (layer_num, size + 1, head_num, self.flashmla_bytes_per_token), dtype=torch.uint8, device="cuda"
        )
        self.indexer_k_buffer = torch.empty(
            (layer_num, size + 1, head_num, self.indexer_bytes_per_token), dtype=torch.uint8, device="cuda"
        )

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_kv_flashmla_fp8 import (
            destindex_copy_kv_flashmla_fp8,
        )

        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv_lora_rank == 512, f"Expected kv_lora_rank=512, got {kv_lora_rank}"

        o_nope = self.kv_buffer[layer_index][:, :, :512].view(torch.float8_e4m3fn)
        o_scale = self.kv_buffer[layer_index][:, :, 512:528].view(torch.float32)
        o_rope = self.kv_buffer[layer_index][:, :, 528:].view(torch.bfloat16)
        destindex_copy_kv_flashmla_fp8(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            o_nope,
            o_scale,
            o_rope,
        )

    def get_att_input_params(self, layer_index: int) -> Any:
        return self.get_flashmla_kv_cache(layer_index)

    def get_flashmla_kv_cache(self, layer_index: int) -> torch.Tensor:
        return self.kv_buffer[layer_index].view(-1, 1, 1, self.flashmla_bytes_per_token)

    def _dequantize_packed_kv(self, packed_kv: torch.Tensor) -> torch.Tensor:
        kv_nope = packed_kv[:, :, : self.kv_nope_dim].view(torch.float8_e4m3fn)
        kv_scale = packed_kv[:, :, self.kv_nope_dim : self.kv_nope_dim + self.quant_group_num * 4].view(torch.float32)
        kv_rope = packed_kv[:, :, self.kv_nope_dim + self.quant_group_num * 4 :].view(torch.bfloat16)

        kv_nope = kv_nope.view(-1, 1, self.quant_group_num, self.quant_group_size).to(self.prefill_dtype)
        kv_scale = kv_scale.to(self.prefill_dtype).unsqueeze(-1)
        kv_nope = (kv_nope * kv_scale).view(-1, 1, self.kv_nope_dim)

        kv = torch.empty(
            (packed_kv.shape[0], packed_kv.shape[1], self.kv_head_dim),
            dtype=self.prefill_dtype,
            device=packed_kv.device,
        )
        kv[:, :, : self.kv_nope_dim] = kv_nope
        kv[:, :, self.kv_nope_dim :] = kv_rope.to(self.prefill_dtype)
        return kv

    def get_prefill_kv_cache(self, layer_index: int) -> torch.Tensor:
        return self._dequantize_packed_kv(self.kv_buffer[layer_index])

    def get_prefill_kv_cache_and_remap_indices(self, layer_index: int, topk_indices: torch.Tensor):
        squeeze_h_kv = topk_indices.ndim == 2
        if squeeze_h_kv:
            topk_indices = topk_indices.unsqueeze(1)

        valid_mask = topk_indices != -1
        valid_indices = topk_indices[valid_mask]

        if valid_indices.numel() == 0:
            empty_kv = torch.empty(
                (0, 1, self.kv_head_dim),
                dtype=self.prefill_dtype,
                device=topk_indices.device,
            )
            remapped = topk_indices.clone()
            if squeeze_h_kv:
                remapped = remapped.squeeze(1)
            return empty_kv, remapped

        unique_mem_index, inverse = torch.unique(valid_indices, sorted=False, return_inverse=True)
        packed_kv = self.kv_buffer[layer_index].index_select(0, unique_mem_index.to(torch.int64))
        compact_kv = self._dequantize_packed_kv(packed_kv)

        remapped = torch.full_like(topk_indices, -1)
        remapped[valid_mask] = inverse.to(remapped.dtype)

        if squeeze_h_kv:
            remapped = remapped.squeeze(1)
        return compact_kv, remapped

    def get_indexer_k_buffer(self, layer_index: int) -> torch.Tensor:
        return self.indexer_k_buffer[layer_index]

    def alloc_kv_move_buffer(self, max_req_total_len):
        self.kv_move_buffer = torch.empty(
            (1, max_req_total_len + 8, self.head_num, self.flashmla_bytes_per_token), dtype=torch.uint8, device="cuda"
        )
        self.indexer_k_move_buffer = torch.empty(
            (1, max_req_total_len + 8, self.head_num, self.indexer_bytes_per_token), dtype=torch.uint8, device="cuda"
        )
        self.kv_move_buf_indexes = torch.arange(0, max_req_total_len + 8, dtype=torch.int64, device="cuda")
        self.token_dim_size = self.flashmla_bytes_per_token
        self.indexer_token_dim_size = self.indexer_bytes_per_token
        return

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        self.kv_move_buffer = torch.empty(
            (page_num, page_size, self.layer_num, self.head_num, self.flashmla_bytes_per_token),
            dtype=torch.uint8,
            device="cuda",
        )
        self.indexer_k_paged_move_buffer = torch.empty(
            (page_num, page_size, self.layer_num, self.head_num, self.indexer_bytes_per_token),
            dtype=torch.uint8,
            device="cuda",
        )
        self._buffer_mem_indexes_tensors = [
            torch.empty((page_size,), dtype=torch.int64, device="cpu", pin_memory=True) for _ in range(page_num)
        ]
        return self.kv_move_buffer

    def write_mem_to_page_kv_move_buffer(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["FP8PerTokenGroupQuantDeepseek3_2MemoryManager"],
        dp_world_size: int,
    ):
        cur_page = self.kv_move_buffer[page_index]
        cur_indexer_page = self.indexer_k_paged_move_buffer[page_index]
        pin_mem_indexes = self._buffer_mem_indexes_tensors[page_index][0 : len(mem_indexes)]
        pin_mem_indexes.numpy()[:] = mem_indexes
        mem_indexes_gpu = pin_mem_indexes.cuda(non_blocking=True)
        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        mla_page_io(mem_indexes=mem_indexes_gpu, page_tensor=cur_page, kv_buffer=dp_mems[0].kv_buffer, mode="write")
        mla_page_io(
            mem_indexes=mem_indexes_gpu,
            page_tensor=cur_indexer_page,
            kv_buffer=dp_mems[0].indexer_k_buffer,
            mode="write",
        )

    def read_page_kv_move_buffer_to_mem(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["FP8PerTokenGroupQuantDeepseek3_2MemoryManager"],
        dp_world_size: int,
    ):
        cur_page = self.kv_move_buffer[page_index]
        cur_indexer_page = self.indexer_k_paged_move_buffer[page_index]
        pin_mem_indexes = self._buffer_mem_indexes_tensors[page_index][0 : len(mem_indexes)]
        pin_mem_indexes.numpy()[:] = mem_indexes
        mem_indexes_gpu = pin_mem_indexes.cuda(non_blocking=True)
        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        for mem in dp_mems:
            mla_page_io(mem_indexes=mem_indexes_gpu, page_tensor=cur_page, kv_buffer=mem.kv_buffer, mode="read")
            mla_page_io(
                mem_indexes=mem_indexes_gpu,
                page_tensor=cur_indexer_page,
                kv_buffer=mem.indexer_k_buffer,
                mode="read",
            )

    def send_to_decode_node(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["FP8PerTokenGroupQuantDeepseek3_2MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        assert dp_size_in_node == 1
        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])
        cur_device_index = self.kv_buffer.get_device()
        cur_mem = mem_managers[cur_device_index]
        for layer_index in range(cur_mem.layer_num):
            nccl_comm.send(self._get_main_move_data(move_token_indexes, layer_index), dst=1)
            nccl_comm.send(self._get_indexer_move_data(move_token_indexes, layer_index), dst=1)

    def receive_from_prefill_node(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["FP8PerTokenGroupQuantDeepseek3_2MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        assert dp_size_in_node == 1
        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])

        cur_device_index = self.kv_buffer.get_device()
        token_num = len(move_token_indexes)
        main_buffer = self.kv_move_buffer.view(-1)[0 : self.token_dim_size * token_num].view(
            1, token_num, self.head_num, self.flashmla_bytes_per_token
        )
        indexer_buffer = self.indexer_k_move_buffer.view(-1)[0 : self.indexer_token_dim_size * token_num].view(
            1, token_num, self.head_num, self.indexer_bytes_per_token
        )
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                nccl_comm.recv(main_buffer, src=0)
                nccl_comm.recv(indexer_buffer, src=0)
                if i == cur_device_index:
                    mem._write_main_move_data(move_token_indexes, main_buffer, layer_index)
                    mem._write_indexer_move_data(move_token_indexes, indexer_buffer, layer_index)
                else:
                    new_main = mem.kv_move_buffer.view(-1)[0 : self.token_dim_size * token_num].view(main_buffer.shape)
                    new_indexer = mem.indexer_k_move_buffer.view(-1)[0 : self.indexer_token_dim_size * token_num].view(
                        indexer_buffer.shape
                    )
                    from torch.cuda import comm

                    comm.broadcast(main_buffer, out=[new_main])
                    comm.broadcast(indexer_buffer, out=[new_indexer])
                    mem._write_main_move_data(move_token_indexes, new_main, layer_index)
                    mem._write_indexer_move_data(move_token_indexes, new_indexer, layer_index)

    def send_to_decode_node_p2p(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["FP8PerTokenGroupQuantDeepseek3_2MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        assert dp_size_in_node == 1
        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])
        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")
        for layer_index in range(self.layer_num):
            nccl_comm.send(self._get_main_move_data_p2p(move_token_indexes, layer_index, self.kv_move_buffer), dst=1)
            nccl_comm.send(
                self._get_indexer_move_data_p2p(move_token_indexes, layer_index, self.indexer_k_move_buffer), dst=1
            )

    def receive_from_prefill_node_p2p(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["FP8PerTokenGroupQuantDeepseek3_2MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        assert dp_size_in_node == 1
        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])
        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")
        token_num = len(move_token_indexes)
        main_buffer = self.kv_move_buffer.view(-1)[0 : self.token_dim_size * token_num].view(
            token_num, self.head_num, self.flashmla_bytes_per_token
        )
        indexer_buffer = self.indexer_k_move_buffer.view(-1)[0 : self.indexer_token_dim_size * token_num].view(
            token_num, self.head_num, self.indexer_bytes_per_token
        )
        for mem in mem_managers:
            for layer_index in range(mem.layer_num):
                nccl_comm.recv(main_buffer, src=0)
                nccl_comm.recv(indexer_buffer, src=0)
                mem._write_main_move_data_p2p(move_token_indexes, main_buffer, layer_index)
                mem._write_indexer_move_data_p2p(move_token_indexes, indexer_buffer, layer_index)

    def _get_main_move_data(self, token_indexes: List[int], layer_index: int):
        move_size = self.token_dim_size * len(token_indexes)
        move_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(
            1, len(token_indexes), self.head_num, self.flashmla_bytes_per_token
        )
        move_buffer[:, :, :, :] = self.kv_buffer[layer_index, token_indexes, :, :]
        return move_buffer

    def _get_indexer_move_data(self, token_indexes: List[int], layer_index: int):
        move_size = self.indexer_token_dim_size * len(token_indexes)
        move_buffer = self.indexer_k_move_buffer.view(-1)[0:move_size].view(
            1, len(token_indexes), self.head_num, self.indexer_bytes_per_token
        )
        move_buffer[:, :, :, :] = self.indexer_k_buffer[layer_index, token_indexes, :, :]
        return move_buffer

    def _write_main_move_data(self, token_indexes: torch.Tensor, buffer_tensor: torch.Tensor, layer_index: int):
        self.kv_buffer[layer_index : layer_index + 1, token_indexes, :, :] = buffer_tensor

    def _write_indexer_move_data(self, token_indexes: torch.Tensor, buffer_tensor: torch.Tensor, layer_index: int):
        self.indexer_k_buffer[layer_index : layer_index + 1, token_indexes, :, :] = buffer_tensor

    def _get_main_move_data_p2p(self, token_indexes: torch.Tensor, layer_index: int, kv_move_buffer: torch.Tensor):
        move_token_num = len(token_indexes)
        move_size = self.token_dim_size * move_token_num
        move_buffer = kv_move_buffer.view(-1)[0:move_size].view(
            move_token_num, self.head_num, self.flashmla_bytes_per_token
        )
        kv_trans(self.kv_buffer[layer_index], token_indexes, move_buffer, self.kv_move_buf_indexes[0:move_token_num])
        return move_buffer

    def _get_indexer_move_data_p2p(self, token_indexes: torch.Tensor, layer_index: int, kv_move_buffer: torch.Tensor):
        move_token_num = len(token_indexes)
        move_size = self.indexer_token_dim_size * move_token_num
        move_buffer = kv_move_buffer.view(-1)[0:move_size].view(
            move_token_num, self.head_num, self.indexer_bytes_per_token
        )
        kv_trans(
            self.indexer_k_buffer[layer_index], token_indexes, move_buffer, self.kv_move_buf_indexes[0:move_token_num]
        )
        return move_buffer

    def _write_main_move_data_p2p(self, token_indexes: torch.Tensor, buffer_tensor: torch.Tensor, layer_index: int):
        move_token_num = len(token_indexes)
        kv_trans(buffer_tensor, self.kv_move_buf_indexes[0:move_token_num], self.kv_buffer[layer_index], token_indexes)

    def _write_indexer_move_data_p2p(self, token_indexes: torch.Tensor, buffer_tensor: torch.Tensor, layer_index: int):
        move_token_num = len(token_indexes)
        kv_trans(
            buffer_tensor, self.kv_move_buf_indexes[0:move_token_num], self.indexer_k_buffer[layer_index], token_indexes
        )
