# 该文件用于提供在数据dp并行的推理模式下，共享kv cache trans相关的功能函数模块
import time
import numpy as np
import dataclasses
import torch
from typing import List, Dict
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_dp_rank_in_node
from lightllm.utils.multimodal_chunk_utils import get_mm_input_spans
from lightllm.server.core.objs.shm_array import ShmArray
from ...infer_batch import InferReq
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
import torch.distributed as dist


@dataclasses.dataclass
class PrefixCacheMatch:
    node: object
    cache: object

    @property
    def matched_len(self) -> int:
        return 0 if self.node is None else self.node.node_prefix_total_len

    def release(self):
        if self.node is not None:
            self.cache.release_checkpoint_pin(self.node)
            self.cache.dec_node_ref_counter(self.node)
            self.node = None


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

    def init_dsv4_cache_transfer(self, mem_managers: List[MemoryManager]) -> None:
        shard_mems = mem_managers[self.backend.rank_in_dp :: self.backend.dp_world_size]
        self.dsv4_source_pool_ptrs = torch.tensor(
            [
                [mem.kv_buffer[layer].data_ptr() for mem in shard_mems]
                for layer in range(len(self.backend.model.mem_manager.kv_buffer))
            ],
            dtype=torch.uint64,
            device="cuda",
        )
        buffers = self.backend.model.mem_manager.kv_buffer
        self.dsv4_destination_pool_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in buffers], dtype=torch.uint64, device="cuda"
        )
        self.dsv4_slot_bytes = torch.tensor([buffer.shape[-1] for buffer in buffers], dtype=torch.int64, device="cuda")
        self.dsv4_max_slot_bytes = max(buffer.shape[-1] for buffer in buffers)
        return

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

    def probe_dsv4_matches(self, reqs: List[tuple]) -> Dict[int, PrefixCacheMatch]:
        """Retain radix history and its terminal checkpoint without allocating request slots."""
        matches = {}
        try:
            for req_id, shm_index, multimodal_params, _ in reqs:
                shm_req = g_infer_context.shm_req_manager.get_req_obj_by_index(shm_index)
                try:
                    shm_req.link_prompt_ids_shm_array()
                    image_spans = (
                        get_mm_input_spans(shm_req.shm_prompt_ids.arr, multimodal_params.to_dict())
                        if self.backend.args.disable_chunked_mm_input
                        else []
                    )
                    node = g_infer_context.retain_hybrid_prefix(
                        shm_req, shm_req.input_len, image_spans, pin_checkpoint=True
                    )
                    matches[req_id] = PrefixCacheMatch(node, self.backend.radix_cache)
                finally:
                    g_infer_context.shm_req_manager.put_back_req_obj(shm_req)
        except BaseException:
            for match in matches.values():
                match.release()
            raise
        return matches

    def gather_dsv4_match_lens(self, reqs: List[tuple], matches: Dict[int, PrefixCacheMatch]):
        local_info = [matches[req[0]].matched_len for req in reqs]
        group = self.backend.node_gloo_group
        all_rank_info = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(all_rank_info, local_info, group=group)
        match_lens = np.zeros((len(reqs), self.backend.dp_size_in_node), dtype=np.int64)
        # Every TP rank must hold a resumable checkpoint at the advertised end.
        # If their cache states differ, this DP offers no remote prefix this round.
        for dp_rank in range(self.backend.dp_size_in_node):
            rank_info = all_rank_info[dp_rank * self.backend.dp_world_size : (dp_rank + 1) * self.backend.dp_world_size]
            for req_index, lengths in enumerate(zip(*rank_info)):
                consistent_len = lengths[0] if len(set(lengths)) == 1 else 0
                match_lens[req_index, dp_rank] = consistent_len
        return match_lens

    def build_shared_kv_trans_tasks(
        self,
        reqs: List[InferReq],
        req_dp_ranks: List[int],
    ) -> List["TransTask"]:
        """
        构建共享kv交换信息
        """
        assert not self.backend.is_deepseek_v4
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

    def build_dsv4_trans_tasks(self, reqs: List[tuple], local_reqs: List[InferReq], match_lens: np.ndarray):
        """Build the cross-DP transfer schedule and allocate destination-side cache resources."""
        if self.backend.dp_size_in_node == 1:
            return [], []

        # ---------------------------------------------------------------------
        # 阶段 1：获取当前 rank 的资源容量，生成本 DP 所属请求的候选传输计划。
        # ---------------------------------------------------------------------
        by_id = {req.req_id: req for req in local_reqs}
        token_capacity = g_infer_context.get_can_alloc_token_num()
        local_plans = {}
        for index, (req_id, _, _, owner_dp) in enumerate(reqs):
            if owner_dp != self.dp_rank_in_node:
                continue
            req = by_id[req_id]
            source_dp = max(
                (dp for dp in range(self.backend.dp_size_in_node) if dp != owner_dp),
                key=lambda dp: int(match_lens[index, dp]),
            )
            source_len = int(match_lens[index, source_dp])
            end = min(source_len, req.get_cur_total_len() - 1)
            trans_size = end - req.cur_kv_len
            if trans_size <= 0:
                continue
            alloc_tokens = req._kv_cache_alloc_need(end)
            if alloc_tokens > token_capacity:
                continue

            # 此处只预占容量计数；所有 TP rank 达成一致前不实际分配资源。
            local_plans[index] = (source_dp, req.cur_kv_len, end)
            token_capacity -= alloc_tokens

        # ---------------------------------------------------------------------
        # 阶段 2：收集节点内所有 rank 的候选计划，供同一 DP 的 TP ranks 达成一致。
        # ---------------------------------------------------------------------
        group = self.backend.node_gloo_group
        all_rank_plans = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(all_rank_plans, local_plans, group=group)

        # ---------------------------------------------------------------------
        # 阶段 3：构造逐 rank 的收发计划，并在请求所属 DP 上实际分配目标资源。
        # ---------------------------------------------------------------------
        tasks = []
        transfer_plan = [[] for _ in all_rank_plans]
        for index, (req_id, _, _, owner_dp) in enumerate(reqs):
            owner_ranks = range(owner_dp * self.backend.dp_world_size, (owner_dp + 1) * self.backend.dp_world_size)
            plans = [all_rank_plans[rank].get(index) for rank in owner_ranks]
            if plans[0] is None or any(plan != plans[0] for plan in plans):
                continue
            source_dp, start, end = plans[0]

            # 源、目标使用相同的 rank_in_dp，使每个 TP shard 与对应 shard 配对。
            for destination in owner_ranks:
                source = source_dp * self.backend.dp_world_size + destination % self.backend.dp_world_size
                transfer_plan[destination].append((source, req_id, start, end))

            # 所有 rank 都构造相同的 transfer_plan；只有 owner DP 分配本地接收资源。
            if owner_dp != self.dp_rank_in_node:
                continue
            req = by_id[req_id]
            assert req.hold_kv_len == req.cur_kv_len
            mem_indexes = self.backend._alloc_req_kv_mem(req, req._kv_cache_alloc_need(end))
            assert mem_indexes is not None
            tasks.append(
                TransTask(
                    req=req,
                    mem_indexes=mem_indexes[: end - start],
                    max_kv_len_dp_rank=source_dp,
                    max_kv_len_mem_manager_index=source_dp * self.backend.dp_world_size + self.backend.rank_in_dp,
                )
            )

        # Group each receiver's requests by source, preserving first-source and per-source order.
        # Every rank builds this schedule from the same gathered plans, so send/recv stay paired.
        for destination, destination_tasks in enumerate(transfer_plan):
            by_source = {}
            for task in destination_tasks:
                by_source.setdefault(task[0], []).append(task)
            reordered_tasks = []
            for source_tasks in by_source.values():
                for task in source_tasks:
                    reordered_tasks.append(task)
            transfer_plan[destination] = reordered_tasks

        return tasks, transfer_plan

    def _transfer_dsv4_source_data(self, trans_tasks, matches, transfer_plan):
        """Exchange slot indexes and shared checkpoint IDs; checkpoint bytes stay node-local."""
        args = self.backend.args
        big_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        big_enabled = big_tokens <= args.max_req_total_len
        group = self.backend.node_gloo_group
        rank = dist.get_rank(group=group)
        local_by_id = {task.req.req_id: task for task in trans_tasks}
        cache = self.backend.radix_cache
        big_buffers = self.backend.model.mem_manager.big_page_buffers
        small_buffers = self.backend.small_page_buffers
        for destination, tasks in enumerate(transfer_plan):
            for source, req_id, start, end in tasks:
                if rank not in (source, destination):
                    continue
                big_lengths = (
                    list(range((start // big_tokens + 1) * big_tokens, end + 1, big_tokens)) if big_enabled else []
                )
                needs_small = end % big_tokens != 0 or not big_enabled
                if rank == source:
                    match = matches[req_id]
                    assert match.matched_len == end
                    indexes = (
                        cache.get_mem_index_value_by_node(match.node, start, end).to(dtype=torch.int64).contiguous()
                    )
                    big_ids = cache.get_big_page_ids_by_node(match.node)
                    ids = [big_ids[length // big_tokens - 1] for length in big_lengths]
                    if needs_small:
                        assert match.node.small_page_buffer_idx is not None
                        ids.append(match.node.small_page_buffer_idx)
                    peer = dist.get_global_rank(group, destination)
                    dist.send(indexes, dst=peer, group=group)
                    dist.send(torch.tensor(ids, dtype=torch.int64), dst=peer, group=group)
                else:
                    task = local_by_id[req_id]
                    peer = dist.get_global_rank(group, source)
                    indexes = torch.empty(end - start, dtype=torch.int64)
                    ids = torch.empty(len(big_lengths) + int(needs_small), dtype=torch.int64)
                    dist.recv(indexes, src=peer, group=group)
                    dist.recv(ids, src=peer, group=group)
                    task.max_kv_len_mem_indexes = indexes.to(device="cuda", dtype=torch.int32)
                    big_ids = ids[: len(big_lengths)].tolist()
                    big_buffers.retain_state_cache(big_ids)
                    task.req.hybrid_len_to_big_page_id.update(zip(big_lengths, big_ids))
                    if needs_small:
                        small_id = int(ids[-1])
                        small_buffers.retain_state_cache([small_id])
                        task.terminal_small_page_buffer_id = small_id
                        if end == task.req.hybrid_cache_len:
                            task.req.tail_small_page_buffer_id = small_id

    def kv_trans_dsv4(self, trans_tasks: List["TransTask"], matches: Dict[int, PrefixCacheMatch], transfer_plan):
        self._transfer_dsv4_source_data(trans_tasks, matches, transfer_plan)
        if trans_tasks:
            req_manager = g_infer_context.req_manager
            from lightllm.common.kv_trans_kernel.kv_trans_v2 import kv_trans_v2_for_p_node_batch

            task_meta = []
            block_counts = []
            for task in trans_tasks:
                req = task.req
                length = len(task.mem_indexes)
                destination = req_manager.req_to_token_indexs[req.req_idx, req.cur_kv_len : req.cur_kv_len + length]
                task_meta.extend(
                    [task.max_kv_len_dp_rank, task.max_kv_len_mem_indexes.data_ptr(), destination.data_ptr(), length]
                )
                block_counts.append((length + 255) // 256)
            # Round-robin request blocks so short requests do not wait behind one long copy.
            blocks = [
                value
                for block in range(max(block_counts))
                for index, count in enumerate(block_counts)
                if block < count
                for value in (index, block)
            ]
            metadata = g_pin_mem_manager.gen_from_list(
                key="dsv4_dp_copy_tasks", data=task_meta + blocks, dtype=torch.uint64
            ).cuda(non_blocking=True)
            task_elements = len(task_meta)
            kv_trans_v2_for_p_node_batch(
                source_pools=self.dsv4_source_pool_ptrs,
                destination_pools=self.dsv4_destination_pool_ptrs,
                slot_bytes=self.dsv4_slot_bytes,
                tasks=metadata[:task_elements].view(-1, 4),
                blocks=metadata[task_elements:].view(-1, 2),
                max_slot_bytes=self.dsv4_max_slot_bytes,
            )
            big_tokens = self.backend.args.linear_att_hash_page_size * self.backend.args.linear_att_page_block_num
            for task in trans_tasks:
                req = task.req
                end = req.cur_kv_len + len(task.mem_indexes)
                if end % big_tokens == 0:
                    checkpoint_id = req.hybrid_len_to_big_page_id[end]
                    state_cache = self.backend.model.mem_manager.big_page_buffers
                else:
                    checkpoint_id = task.terminal_small_page_buffer_id
                    state_cache = self.backend.small_page_buffers
                req.cur_kv_len = end
                req_manager.restore_state(req, state_cache, checkpoint_id)
                assert req.cur_kv_len <= req.hold_kv_len
                if self.backend.is_master_in_dp:
                    req.shm_req.shm_cur_kv_len = end
                    req.shm_req.prompt_cache_len = end
            self.backend.logger.info(
                f"dp_i {self.dp_rank_in_node} transfer kv tokens num: "
                f"{sum(len(task.mem_indexes) for task in trans_tasks)} using shared checkpoint IDs"
            )
        # Destination kernels must finish reading source GPU pools before source radix refs are released.
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=self.backend.node_nccl_group)

    def kv_trans(self, trans_tasks: List["TransTask"]):
        """Generic-model transfer; DeepSeek-V4 uses kv_trans_dsv4 instead."""
        assert not self.backend.is_deepseek_v4
        if trans_tasks:
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

        for trans_task in trans_tasks:
            trans_task.req.cur_kv_len += len(trans_task.mem_indexes)
            assert trans_task.req.cur_kv_len <= trans_task.req.hold_kv_len
            if self.backend.is_master_in_dp:
                trans_task.req.shm_req.shm_cur_kv_len = trans_task.req.cur_kv_len


@dataclasses.dataclass
class TransTask:
    req: InferReq
    mem_indexes: torch.Tensor
    max_kv_len_dp_rank: int
    max_kv_len_mem_manager_index: int
    max_kv_len_mem_indexes: torch.Tensor = None
    terminal_small_page_buffer_id: int = None
