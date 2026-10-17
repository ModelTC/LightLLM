# 该文件用于提供在数据dp并行的推理模式下，共享kv cache trans相关的功能函数模块
import numpy as np
import dataclasses
from types import SimpleNamespace
import torch
from typing import List
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_dp_rank_in_node
from lightllm.server.core.objs.shm_array import ShmArray
from ...infer_batch import InferReq
from lightllm.server.router.model_infer.infer_batch import g_infer_context
import torch.distributed as dist


class DPKVSharedMoudle:
    _KV_LEN_INDEX = 0
    _REQ_IDX_INDEX = 1

    def __init__(self, max_req_num: int, dp_size_in_node: int, backend):
        from .impl import DPChunkedPrefillBackend

        self.backend: DPChunkedPrefillBackend = backend
        self.max_req_num = max_req_num

        # 0 代表 kv_len, 1 代表 radix_cache_len
        self.shared_req_infos = ShmArray(
            name="dp_shared_req_infos",
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
        if not self.backend.is_hybrid_att_model:
            dist.barrier(group=self.backend.node_nccl_group)
        if self.backend.is_master_in_dp:
            lengths = [req.cur_kv_len for req in reqs]
            if self.backend.is_hybrid_att_model:
                lengths = [_hybrid_transfer_len(req, self.backend.args) for req in reqs]
            self.shared_req_infos.arr[0 : len(reqs), self.dp_rank_in_node, self._KV_LEN_INDEX] = lengths
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

            target_kv_len = req.cur_kv_len + trans_size
            if self.backend.is_hybrid_att_model:
                page = self.backend.args.linear_att_hash_page_size
                target_kv_len = target_kv_len // page * page
                if target_kv_len < max_req_radix_cache_len:
                    # A shorter endpoint must have its own saved checkpoint.
                    big_page = page * self.backend.args.linear_att_page_block_num
                    target_kv_len = target_kv_len // big_page * big_page
                trans_size = target_kv_len - req.cur_kv_len
            alloc_token_num = req._kv_cache_alloc_need(target_kv_len) if trans_size > 0 else 0

            if is_current_dp_handle and trans_size > 0 and alloc_token_num <= g_infer_context.get_can_alloc_token_num():
                assert req.hold_kv_len == req.cur_kv_len
                mem_indexes = self.backend._alloc_req_kv_mem(req, alloc_token_num)
                assert mem_indexes is not None
                # mem_indexes 只描述需要复制的逻辑 KV；页尾预留槽位已经由
                # _alloc_req_kv_mem 写入请求表，但不参与本次跨 DP 传输。
                mem_indexes = mem_indexes[:trans_size]
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
            self.backend.model.mem_manager.operator.copy_kv_from_other_dp_ranks(
                mem_managers=self.backend.mem_managers,
                move_token_indexes=max_kv_len_mem_indexes_tensor,
                token_dp_indexes=max_kv_len_dp_ranks_tensor,
                mem_indexes=mem_indexes_tensor,
                dp_size_in_node=self.backend.dp_size_in_node,
                rank_in_dp=self.backend.rank_in_dp,
            )
            self.backend.logger.info(f"dp_i {self.dp_rank_in_node} transfer kv tokens num: {len(mem_indexes_tensor)}")

        if self.backend.is_hybrid_att_model:
            transfer_hybrid_checkpoints(self.backend, trans_tasks)
            if self.backend.args.enable_cpu_cache:
                # Retain source pages until all peers finish copying them.
                dist.barrier(group=self.backend.node_nccl_group)

        for trans_task in trans_tasks:
            if not self.backend.is_hybrid_att_model:
                trans_task.req.cur_kv_len += len(trans_task.mem_indexes)
            assert trans_task.req.cur_kv_len <= trans_task.req.hold_kv_len
            if self.backend.is_master_in_dp:
                trans_task.req.shm_req.shm_cur_kv_len = trans_task.req.cur_kv_len


def _hybrid_transfer_len(req, args):
    """Advertise a live aligned state or an existing boundary checkpoint."""
    page = args.linear_att_hash_page_size
    if req.cur_kv_len % page == 0:
        return req.cur_kv_len
    big_page = page * args.linear_att_page_block_num
    length = min(req.cur_kv_len, req.hybrid_cache_len) // big_page * big_page
    if req.tail_small_page_buffer_id is not None:
        length = max(length, req.hybrid_cache_len)
    return length


def transfer_hybrid_checkpoints(backend, trans_tasks):
    """Use the same save/restore interface for every hybrid request-state format."""
    big_tokens = backend.args.linear_att_hash_page_size * backend.args.linear_att_page_block_num
    group = backend.node_gloo_group
    rank = dist.get_rank(group=group)
    local_tasks = [
        (
            task.max_kv_len_mem_manager_index,
            task.req.req_id,
            task.req.cur_kv_len,
            task.req.cur_kv_len + len(task.mem_indexes),
        )
        for task in trans_tasks
    ]
    all_tasks = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(all_tasks, local_tasks, group=group)
    requests = backend.model.req_manager
    buffers = backend.model.mem_manager.big_page_buffers
    for destination, tasks in enumerate(all_tasks):
        for source, req_id, start, end in tasks:
            if rank not in (source, destination):
                continue
            req = g_infer_context.requests_mapping[req_id]
            lengths = range((start // big_tokens + 1) * big_tokens, end + 1, big_tokens)
            snapshot = tuple(torch.empty_like(tensor, pin_memory=True) for tensor in buffers.get_state_cache(0))
            checkpoint = SimpleNamespace(get_state_cache=lambda buffer_idx: snapshot)
            if rank == source:
                peer = dist.get_global_rank(group, destination)
                shared = backend.radix_cache.get_big_page_ids_by_node(req.shared_kv_node) if lengths else []
                for length in lengths:
                    index = (
                        req.hybrid_len_to_big_page_id[length]
                        if length in req.hybrid_len_to_big_page_id
                        else shared[length // big_tokens - 1]
                    )
                    for tensor in buffers.get_state_cache(index):
                        dist.send(tensor, dst=peer, group=group)
                if end == req.cur_kv_len:
                    requests.save_state(req.req_idx, 0, checkpoint, checkpoint_len=end)
                    torch.cuda.current_stream().synchronize()
                else:
                    if end % big_tokens == 0:
                        index = (
                            req.hybrid_len_to_big_page_id[end]
                            if end in req.hybrid_len_to_big_page_id
                            else shared[end // big_tokens - 1]
                        )
                        saved = buffers.get_state_cache(index)
                    else:
                        assert end == req.hybrid_cache_len and req.tail_small_page_buffer_id is not None
                        saved = backend.small_page_buffers.get_state_cache(req.tail_small_page_buffer_id)
                    for target, tensor in zip(snapshot, saved):
                        target.copy_(tensor)
                for tensor in snapshot:
                    dist.send(tensor, dst=peer, group=group)
            else:
                peer = dist.get_global_rank(group, source)
                for length in lengths:
                    index = buffers.alloc_one_state_cache()
                    assert index is not None
                    for tensor in buffers.get_state_cache(index):
                        dist.recv(tensor, src=peer, group=group)
                    req.hybrid_len_to_big_page_id[length] = index
                for tensor in snapshot:
                    dist.recv(tensor, src=peer, group=group)
                req.cur_kv_len = end
                requests.restore_state(req, checkpoint, 0)
                if end == req.hybrid_cache_len and end % big_tokens:
                    backend.radix_cache.free_one_small_page_buffer()
                    small = backend.small_page_buffers
                    index = small.alloc_one_state_cache()
                    if index is not None:
                        for target, tensor in zip(small.get_state_cache(index), snapshot):
                            target.copy_(tensor)
                        req.tail_small_page_buffer_id = index


@dataclasses.dataclass
class TransTask:
    req: InferReq
    mem_indexes: torch.Tensor
    max_kv_len_dp_rank: int
    max_kv_len_mem_manager_index: int
    max_kv_len_mem_indexes: torch.Tensor
