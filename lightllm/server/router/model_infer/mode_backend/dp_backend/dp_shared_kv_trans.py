# 该文件用于提供在数据dp并行的推理模式下，共享kv cache trans相关的功能函数模块
import time
import numpy as np
import dataclasses
import torch
from typing import List
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from lightllm.utils.dist_utils import get_dp_rank_in_node
from lightllm.server.core.objs.shm_array import ShmArray
from ...infer_batch import InferReq
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
import torch.distributed as dist
from lightllm.models.deepseek_v4.triton_kernel.dp_cache_io import copy_dsv4_dp_caches


class DPKVSharedMoudle:
    _KV_LEN_INDEX = 0
    _REQ_IDX_INDEX = 1

    def __init__(self, max_req_num: int, dp_size_in_node: int, backend):
        from .impl import DPChunkedPrefillBackend

        self.backend: DPChunkedPrefillBackend = backend
        self.max_req_num = max_req_num

        # 0 代表 kv_len, 1 代表 radix_cache_len
        self.shared_req_infos = ShmArray(
            name=f"{get_unique_server_name()}_dp_shared_req_infos",
            shape=(self.max_req_num, dp_size_in_node, 2),
            dtype=np.int64,
        )
        self.shared_req_infos.create_shm()
        self.dp_rank_in_node = get_dp_rank_in_node()
        assert get_env_start_args().diverse_mode is False

        if self.backend.is_deepseek_v4:
            from lightllm.utils.device_utils import kv_trans_use_p2p

            assert kv_trans_use_p2p(), "DeepSeek-V4 DP prompt-cache fetch requires P2P KV transfer"

    def init_dsv4_cache_transfer(self, mem_managers: List[MemoryManager]) -> None:
        pointer_rows = []
        for mem_manager in mem_managers:
            has_c4 = mem_manager.c4_pool is not None
            has_c128 = mem_manager.c128_pool is not None
            pointer_rows.append(
                [
                    mem_manager.full_to_c4_indexs.data_ptr() if has_c4 else 0,
                    mem_manager.c4_pool.buffer.data_ptr() if has_c4 else 0,
                    mem_manager.c4_indexer_pool.buffer.data_ptr() if has_c4 else 0,
                    mem_manager.full_to_c128_indexs.data_ptr() if has_c128 else 0,
                    mem_manager.c128_pool.buffer.data_ptr() if has_c128 else 0,
                    mem_manager.full_to_swa_indexs.data_ptr(),
                    mem_manager.swa_pool.buffer.data_ptr(),
                    mem_manager.c4_state_buffer.data_ptr() if has_c4 else 0,
                    mem_manager.c4_indexer_state_buffer.data_ptr() if has_c4 else 0,
                ]
            )
        self.dsv4_source_pool_ptrs = torch.tensor(pointer_rows, dtype=torch.uint64, device="cuda")
        return

    def fill_reqs_info(self, reqs: List[InferReq]):
        """
        填充请求的 kv 信息到共享内存中
        """
        if not self.backend.is_deepseek_v4:
            dist.barrier(group=self.backend.node_nccl_group)
        if self.backend.is_master_in_dp:
            self.shared_req_infos.arr[0 : len(reqs), self.dp_rank_in_node, self._KV_LEN_INDEX] = [
                req.cur_kv_len for req in reqs
            ]
            self.shared_req_infos.arr[0 : len(reqs), self.dp_rank_in_node, self._REQ_IDX_INDEX] = [
                req.req_idx for req in reqs
            ]
        return

    def build_shared_kv_trans_tasks(
        self,
        reqs: List[InferReq],
        req_dp_ranks: List[int],
    ) -> List["TransTask"]:
        """
        构建共享kv交换信息
        """
        dist.barrier(group=self.backend.node_nccl_group)

        trans_tasks: List[TransTask] = []
        if self.backend.is_deepseek_v4:
            dsv4_mem_manager = self.backend.model.mem_manager
            (
                dsv4_swa_capacity,
                dsv4_c4_capacity,
                dsv4_c128_capacity,
            ) = g_infer_context.get_can_alloc_dsv4_page_and_slot_num()
            dsv4_prompt_page_size = self.backend.model.req_manager.get_prompt_cache_page_size()

        rank_max_radix_cache_lens = np.max(
            self.shared_req_infos.arr[0 : len(reqs), :, self._KV_LEN_INDEX], axis=1, keepdims=False
        )
        # 如果发现自己是dp_rank 最小， radix_cache_len 最长的请求，则将数据写入到共享内存中。
        for req_index, req, max_req_radix_cache_len, req_dp_rank in zip(
            list(range(len(reqs))), reqs, rank_max_radix_cache_lens, req_dp_ranks
        ):
            # 当前请求是本 dp_rank 负责的
            is_current_dp_handle = req_dp_rank == self.dp_rank_in_node
            # 计算需要传输的 kv 长度， 不能超过 req.get_cur_total_len() - 1
            trans_size = min(max_req_radix_cache_len, req.get_cur_total_len() - 1) - req.cur_kv_len

            can_alloc_dsv4_cache = True
            if self.backend.is_deepseek_v4 and trans_size > 0:
                need_swa_pages = dsv4_prompt_page_size // dsv4_mem_manager.swa_pool.page_size
                need_c4_pages = trans_size // dsv4_prompt_page_size if dsv4_mem_manager.c4_pool is not None else 0
                need_c128_slots = trans_size // 128 if dsv4_mem_manager.c128_pool is not None else 0
                can_alloc_dsv4_cache = (
                    dsv4_swa_capacity >= need_swa_pages
                    and dsv4_c4_capacity >= need_c4_pages
                    and dsv4_c128_capacity >= need_c128_slots
                )

            if (
                is_current_dp_handle
                and trans_size > 0
                and g_infer_context.get_can_alloc_token_num() > trans_size
                and can_alloc_dsv4_cache
            ):
                g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(trans_size)
                mem_indexes = self.backend.model.mem_manager.alloc(trans_size)
                if self.backend.is_deepseek_v4:
                    dsv4_swa_capacity -= need_swa_pages
                    dsv4_c4_capacity -= need_c4_pages
                    dsv4_c128_capacity -= need_c128_slots
                max_kv_len_dp_rank = self.shared_req_infos.arr[req_index, :, self._KV_LEN_INDEX].argmax()
                max_kv_len_req_idx = int(self.shared_req_infos.arr[req_index, max_kv_len_dp_rank, self._REQ_IDX_INDEX])
                max_kv_len_mem_manager_index = max_kv_len_dp_rank * self.backend.dp_world_size + self.backend.rank_in_dp
                max_kv_len_mem_manager: MemoryManager = self.backend.mem_managers[max_kv_len_mem_manager_index]
                max_kv_len_mem_indexes = max_kv_len_mem_manager.req_to_token_indexs[
                    max_kv_len_req_idx, req.cur_kv_len : req.cur_kv_len + trans_size
                ]
                trans_tasks.append(
                    TransTask(
                        req=req,
                        mem_indexes=mem_indexes,
                        max_kv_len_dp_rank=int(max_kv_len_dp_rank),
                        max_kv_len_mem_manager_index=int(max_kv_len_mem_manager_index),
                        max_kv_len_mem_indexes=max_kv_len_mem_indexes,
                    )
                )

        return trans_tasks

    def kv_trans(self, trans_tasks: List["TransTask"]):
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        # kv 传输
        if len(trans_tasks) > 0:
            if self.backend.is_deepseek_v4:
                req_manager = g_infer_context.req_manager
                prompt_cache_page_size = req_manager.get_prompt_cache_page_size()
                req_list = []
                ready_list = []
                seq_list = []
                dst_full_slot_views = []
                task_meta_data = []
                history_block_nums = []
                for trans_task in trans_tasks:
                    start = trans_task.req.cur_kv_len
                    end = start + len(trans_task.mem_indexes)
                    dst_full_slots = req_manager.req_to_token_indexs[trans_task.req.req_idx, start:end]
                    dst_full_slots.copy_(trans_task.mem_indexes, non_blocking=True)
                    req_list.append(trans_task.req.req_idx)
                    ready_list.append(start)
                    seq_list.append(end)
                    dst_full_slot_views.append(dst_full_slots)
                    task_meta_data.extend(
                        [
                            trans_task.max_kv_len_mem_manager_index,
                            end - start,
                            trans_task.max_kv_len_mem_indexes.data_ptr(),
                            dst_full_slots.data_ptr(),
                        ]
                    )
                    history_block_nums.append((end - start) // prompt_cache_page_size)

                # Keep full slots in the same request-major order as req_list.
                new_full_slots = (
                    dst_full_slot_views[0] if len(dst_full_slot_views) == 1 else torch.cat(dst_full_slot_views)
                )
                req_manager.prepare_pd_decode_cache(
                    req_list=req_list,
                    ready_list=ready_list,
                    seq_list=seq_list,
                    new_full_slots=new_full_slots,
                )

                # The history kernel consumes (task index, block index) pairs in block-major order.
                history_meta_data = []
                for block_index in range(max(history_block_nums)):
                    for task_index, block_num in enumerate(history_block_nums):
                        if block_index < block_num:
                            history_meta_data.extend([task_index, block_index])

                # transfer_meta packs two flat uint64 tables into one H2D copy:
                #   task_meta:    (source manager, token count, source slots pointer, destination slots pointer)
                #   history_meta: (task index, block index)
                # For two tasks with 2 and 1 history blocks, the layout is:
                #   [task0 fields, task1 fields, (0, 0), (1, 0), (0, 1)]
                task_meta_size = len(task_meta_data)
                transfer_meta = g_pin_mem_manager.gen_from_list(
                    key="dsv4_dp_cache_transfer_meta",
                    data=task_meta_data + history_meta_data,
                    dtype=torch.uint64,
                ).to(req_manager.req_to_token_indexs.device, non_blocking=True)

                dst_mem_manager = self.backend.model.mem_manager
                copy_dsv4_dp_caches(
                    source_pool_ptrs=self.dsv4_source_pool_ptrs,
                    dst_mem_manager=dst_mem_manager,
                    task_meta=transfer_meta[:task_meta_size],
                    history_meta=transfer_meta[task_meta_size:],
                )
            else:
                max_kv_len_mem_indexes = []
                max_kv_len_dp_ranks = []
                mem_indexes = []

                for i, trans_task in enumerate(trans_tasks):
                    max_kv_len_mem_indexes.append(trans_task.max_kv_len_mem_indexes)
                    max_kv_len_dp_ranks.extend([trans_task.max_kv_len_dp_rank] * len(trans_task.max_kv_len_mem_indexes))
                    mem_indexes.append(trans_task.mem_indexes)

                max_kv_len_mem_indexes_tensor = torch.cat(max_kv_len_mem_indexes).to(dtype=torch.int64, device="cuda")
                max_kv_len_dp_ranks_tensor = torch.tensor(max_kv_len_dp_ranks, dtype=torch.int32, device="cuda")
                mem_indexes_tensor = torch.cat(mem_indexes).to(dtype=torch.int64, device="cuda")
                self.backend.model.mem_manager.operator.copy_kv_from_other_dp_ranks(
                    mem_managers=self.backend.mem_managers,
                    move_token_indexes=max_kv_len_mem_indexes_tensor,
                    token_dp_indexes=max_kv_len_dp_ranks_tensor,
                    mem_indexes=mem_indexes_tensor,
                    dp_size_in_node=self.backend.dp_size_in_node,
                    rank_in_dp=self.backend.rank_in_dp,
                )

            transfer_token_num = sum(len(trans_task.mem_indexes) for trans_task in trans_tasks)
            self.backend.logger.info(f"dp_i {self.dp_rank_in_node} transfer kv tokens num: {transfer_token_num}")

        if self.backend.is_deepseek_v4 and self.backend.args.enable_cpu_cache:
            # CPU-cache restore can evict source radix pages before the scheduler all-gather fences this stream.
            dist.barrier(group=self.backend.node_nccl_group)

        for trans_task in trans_tasks:
            if not self.backend.is_deepseek_v4:
                g_infer_context.req_manager.req_to_token_indexs[
                    trans_task.req.req_idx,
                    trans_task.req.cur_kv_len : (trans_task.req.cur_kv_len + len(trans_task.mem_indexes)),
                ] = trans_task.mem_indexes
            trans_task.req.cur_kv_len += len(trans_task.mem_indexes)
            if self.backend.is_master_in_dp:
                trans_task.req.shm_req.shm_cur_kv_len = trans_task.req.cur_kv_len


@dataclasses.dataclass
class TransTask:
    req: InferReq
    mem_indexes: torch.Tensor
    max_kv_len_dp_rank: int
    max_kv_len_mem_manager_index: int
    max_kv_len_mem_indexes: torch.Tensor
