# 该文件用于提供在数据dp并行的推理模式下，共享kv cache trans相关的功能函数模块
import numpy as np
import dataclasses
import torch
from typing import List
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

        self.big_page_tokens = (
            backend.args.linear_att_hash_page_size * backend.args.linear_att_page_block_num
            if backend.is_hybrid_att_model else 0
        )
        max_big_pages = backend.args.max_req_total_len // self.big_page_tokens if self.big_page_tokens else 0
        # 原共享信息只保存 KV 长度和请求槽位，大页列表使用独立共享数组。
        self.shared_req_infos = ShmArray(
            name="dp_shared_req_infos",
            shape=(self.max_req_num, dp_size_in_node, 2),
            dtype=np.int64,
        )
        self.shared_req_infos.create_shm()
        self.shared_req_big_page_ids: ShmArray | None = None
        if max_big_pages:
            # 第 i 项对应前缀长度 (i + 1) * big_page_tokens，未使用位置填 -1。
            self.shared_req_big_page_ids = ShmArray(
                name="dp_shared_req_big_page_ids",
                shape=(self.max_req_num, dp_size_in_node, max_big_pages),
                dtype=np.int64,
            )
            self.shared_req_big_page_ids.create_shm()
        self.dp_rank_in_node = get_dp_rank_in_node()
        assert get_env_start_args().diverse_mode is False

    def fill_reqs_info(self, reqs: List[InferReq]) -> None:
        """发布每个请求的 KV 长度、请求槽位及完整有序大页列表。"""
        dist.barrier(group=self.backend.node_nccl_group)
        if self.backend.is_master_in_dp:
            for req_index, req in enumerate(reqs):
                row = self.shared_req_infos.arr[req_index, self.dp_rank_in_node]
                row[self._KV_LEN_INDEX] = req.cur_kv_len
                row[self._REQ_IDX_INDEX] = req.req_idx
                if self.shared_req_big_page_ids is None:
                    continue
                page_ids = self.shared_req_big_page_ids.arr[req_index, self.dp_rank_in_node]
                page_ids.fill(-1)  # 批次复用共享行时，清除上次较长请求留下的页 ID。
                # radix 页按路径从根到叶排列；请求持有的页按其前缀长度放到对应位置。
                if req.shared_kv_node is not None:
                    shared_pages = g_infer_context.radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
                    page_ids[:len(shared_pages)] = shared_pages
                for token_len, page_id in req.hybrid_len_to_big_page_id.items():
                    if token_len <= req.cur_kv_len:
                        page_ids[token_len // self.big_page_tokens - 1] = page_id
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
        available_lens = self.shared_req_infos.arr[0 : len(reqs), :, self._KV_LEN_INDEX]
        rank_max_radix_cache_lens = np.max(available_lens, axis=1)
        # 为本 DP 负责的请求选择可用前缀最长的来源。
        for req_index, req, max_req_radix_cache_len, req_dp_rank in zip(
            list(range(len(reqs))),
            reqs,
            rank_max_radix_cache_lens,
            req_dp_ranks,
        ):
            # 当前请求是本 dp_rank 负责的
            is_current_dp_handle = req_dp_rank == self.dp_rank_in_node
            # 计算需要传输的 kv 长度， 不能超过 req.get_cur_total_len() - 1
            trans_size = min(max_req_radix_cache_len, req.get_cur_total_len() - 1) - req.cur_kv_len
            target_kv_len = req.cur_kv_len + trans_size
            alloc_token_num = req._kv_cache_alloc_need(target_kv_len) if trans_size > 0 else 0

            if is_current_dp_handle and trans_size > 0 and alloc_token_num <= g_infer_context.get_can_alloc_token_num():
                assert req.hold_kv_len == req.cur_kv_len
                src_dp_rank = int(available_lens[req_index].argmax())
                src_info = self.shared_req_infos.arr[req_index, src_dp_rank]
                src_req_idx = int(src_info[self._REQ_IDX_INDEX])
                src_big_page_ids, dst_big_page_ids = [], []
                if self.backend.is_hybrid_att_model:
                    # 只复制新增前缀跨过的完整大页，已有前缀页继续使用本地引用。
                    first_page = req.cur_kv_len // self.big_page_tokens
                    end_page = target_kv_len // self.big_page_tokens
                    if end_page > first_page:
                        src_big_page_ids = self.shared_req_big_page_ids.arr[
                            req_index, src_dp_rank, first_page:end_page
                        ].tolist()
                        assert all(page_id >= 0 for page_id in src_big_page_ids), "DP source is missing prefix big pages"
                        allocator = self.backend.model.att_state_manager.bounded_att_big_page_allocator
                        dst_big_page_ids = allocator.alloc(len(src_big_page_ids))
                        assert dst_big_page_ids is not None, "DP fetch big page allocation must succeed"
                dst_token_indices = self.backend._alloc_req_kv_mem(req, alloc_token_num)
                assert dst_token_indices is not None
                # 只搬运有效 KV；页尾预留槽位已写入请求表，不参与本次传输。
                dst_token_indices = dst_token_indices[:trans_size]
                src_token_indices = self.backend.dp_req_to_token_indices[src_dp_rank][
                    src_req_idx, req.cur_kv_len:target_kv_len
                ]
                trans_tasks.append(
                    TransTask(
                        dst_req=req,
                        src_dp_rank=src_dp_rank,
                        src_req_idx=src_req_idx,
                        src_token_indices=src_token_indices,
                        dst_token_indices=dst_token_indices,
                        src_big_page_ids=src_big_page_ids,
                        dst_big_page_ids=dst_big_page_ids,
                    )
                )

        return trans_tasks

    def kv_trans(self, trans_tasks: List["TransTask"]) -> None:
        target = self.backend.model.att_state_manager
        for task in trans_tasks:
            source = self.backend.dp_att_state_managers[task.src_dp_rank]
            # 源索引来自远端 CUDA 请求表；allocator 返回 CPU 索引，在此一次性准备目标 CUDA int64 索引。
            dst_token_indices = task.dst_token_indices.to(device="cuda", dtype=torch.long, non_blocking=True)
            target.copy_from_dp(
                remote_att_manager=source,
                src_token_indices=task.src_token_indices,
                dst_token_indices=dst_token_indices,
                src_big_page_ids=task.src_big_page_ids,
                dst_big_page_ids=task.dst_big_page_ids,
                src_req_idx=task.src_req_idx,
                dst_req_idx=task.dst_req.req_idx,
            )
        # Source requests may be released on other ranks immediately afterwards.
        # All ranks participate, including those without a local transfer task.
        dist.barrier(group=self.backend.node_nccl_group)

        for task in trans_tasks:
            req = task.dst_req
            # 复制完成后，才把目标大页发布为请求持有的快照；后续 radix 插入或释放沿用此表。
            if task.dst_big_page_ids:
                first_page_end = (req.cur_kv_len // self.big_page_tokens + 1) * self.big_page_tokens
                for page_offset, page_id in enumerate(task.dst_big_page_ids):
                    token_len = first_page_end + page_offset * self.big_page_tokens
                    req.hybrid_len_to_big_page_id[token_len] = page_id
            req.cur_kv_len += len(task.dst_token_indices)
            assert req.cur_kv_len <= req.hold_kv_len
            if self.backend.is_master_in_dp:
                req.shm_req.shm_cur_kv_len = req.cur_kv_len


@dataclasses.dataclass
class TransTask:
    dst_req: InferReq  # 本 DP 接收前缀的请求，持有目标请求槽位和新分配的资源。
    src_dp_rank: int  # 来源 DP 在当前节点内的编号。
    src_req_idx: int  # 远端请求槽位，用于复制 prefill 基准运行态。
    src_token_indices: torch.Tensor  # 来源 CUDA 请求表中的有效 KV 索引。
    dst_token_indices: torch.Tensor  # allocator 返回的目标 CPU 索引，提交时统一转为 CUDA int64。
    src_big_page_ids: List[int]  # 按前缀长度递增排列的源大页槽位。
    dst_big_page_ids: List[int]  # 与源列表一一配对的新分配目标大页槽位。
