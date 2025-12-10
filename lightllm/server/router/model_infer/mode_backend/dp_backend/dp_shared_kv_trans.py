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
import torch.distributed as dist


class DPKVSharedMoudle:
    _KV_LEN_INDEX = 0
    _REQ_IDX_INDEX = 1

    def __init__(self, max_req_num: int, max_req_seq_len: int, dp_size_in_node: int, backend):
        from .impl import DPChunkedPrefillBackend

        self.backend: DPChunkedPrefillBackend = backend
        self.max_req_num = max_req_num
        self.max_req_seq_len = max_req_seq_len

        # 0 代表 kv_len, 1 代表 radix_cache_len
        self.shared_req_infos = ShmArray(
            name=f"{get_unique_server_name()}_dp_shared_req_infos",
            shape=(self.max_req_num, dp_size_in_node, 2),
            dtype=np.int64,
        )
        self.shared_req_infos.create_shm()
        self.dp_rank_in_node = get_dp_rank_in_node()
        assert get_env_start_args().diverse_mode is False

    def fill_reqs_info(self, reqs: List[InferReq]):
        """
        填充请求的 kv 信息到共享内存中
        """
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

            if is_current_dp_handle and trans_size > 0 and g_infer_context.get_can_alloc_token_num() > trans_size:
                g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(trans_size)
                mem_indexes = self.backend.model.mem_manager.alloc(trans_size)
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
            self.backend.model.mem_manager.copy_kv_from_other_dp_ranks(
                mem_managers=self.backend.mem_managers,
                move_token_indexes=max_kv_len_mem_indexes_tensor,
                token_dp_indexes=max_kv_len_dp_ranks_tensor,
                mem_indexes=mem_indexes_tensor,
                dp_size_in_node=self.backend.dp_size_in_node,
                rank_in_dp=self.backend.rank_in_dp,
            )
            self.backend.logger.info(f"dp_i {self.dp_rank_in_node} transfer kv tokens num: {len(mem_indexes_tensor)}")

        for trans_task in trans_tasks:
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
