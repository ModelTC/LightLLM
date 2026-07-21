"""Router 多机纯 TP（nnodes>1 and dp==1）相关逻辑 Mixin。

Public 入口：
1. ``multinode_tp_generate_new_batch`` — 跨节点调度（内部会调用 abort 阶段1）
2. ``get_aborted_reqs_from_running_batch_multinode_tp`` — abort 阶段2（running）
3. ``get_stop_str_matched_reqs_from_running_batch_multinode_tp`` — stop_str 阶段2（running）
"""

import torch
import torch.distributed as dist

from typing import List, Optional, Set
from lightllm.server.router.batch import Batch, Req
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class RouterMultiNodeTpHelper:
    """挂到 ``RouterManager``：提供多机 TP 调度与 abort 两阶段接口。"""

    # ==================================================================
    # Public
    # ==================================================================

    def multinode_tp_generate_new_batch(self):
        """跨节点调度入口：barrier → 只调度 → merge → abort 阶段1 → barrier。"""
        try:
            dist.barrier(group=self.mulitnode_group)
            if self.is_multinode_tp_master:
                new_batch = self._multinode_tp_schedule_as_master()
            else:
                new_batch = self._multinode_tp_schedule_as_slave()
            self.schedule_new_batch = Batch.merge_two_batch(self.schedule_new_batch, new_batch)
            self._filter_aborted_from_schedule_new_batch()
            dist.barrier(group=self.mulitnode_group)
        except Exception as e:
            logger.exception(str(e))
            raise e
        return

    def get_aborted_reqs_from_running_batch_multinode_tp(self) -> List[Req]:
        """Abort 阶段2：对 running_batch 同步 abort，提取尚未下发 AbortedReqCmd 的请求。"""
        ans = []
        running_reqs = [] if self.running_batch is None else self.running_batch.reqs
        aborted_req_ids = self._broadcast_aborted_req_ids_from_master(running_reqs)
        if self.is_multinode_tp_slave:
            for req in running_reqs:
                if req.request_id in aborted_req_ids:
                    req.is_aborted = True

        for req in running_reqs:
            if req.is_aborted and req._router_aborted is False:
                req._router_aborted = True
                ans.append(req)
        return ans

    # ==================================================================
    # Private: 调度
    # ==================================================================

    def _multinode_tp_schedule_as_master(self) -> Optional[Batch]:
        """Master：本地 generate_new_batch，广播 req_ids，按就绪标记裁剪；不处理 abort。"""
        # current_batch: 已占用资源（running + 已调度未推理），供调度估算
        # new_batch: 本轮从 waiting 新选出的候选
        current_batch = Batch.merge_two_batch(self.running_batch, self.schedule_new_batch)
        new_batch = self.req_queue.generate_new_batch(current_batch)
        req_ids = [req.request_id for req in new_batch.reqs] if new_batch is not None else []

        dist.broadcast_object_list([len(req_ids)], src=0, group=self.mulitnode_group)
        if len(req_ids) == 0:
            return None

        dist.broadcast_object_list(req_ids, src=0, group=self.mulitnode_group)
        # master 本机一定有这些 req；用全 1 参与 MIN all_reduce，确认各 slave waiting 是否已收到
        select_marks = self._multinode_tp_allreduce_ready_marks([1] * len(req_ids))

        back_req_list = []
        for req_id, select in zip(req_ids, select_marks):
            if select == 1:
                continue
            # 某节点尚未 ready：打回 waiting，下轮再调度
            req = new_batch.pop_req(req_id)
            back_req_list.append(req)
        self.req_queue.waiting_req_list = back_req_list + self.req_queue.waiting_req_list
        return None if new_batch.is_clear() else new_batch

    def _multinode_tp_schedule_as_slave(self) -> Optional[Batch]:
        """Slave：接收 master 的 req_ids，按就绪标记组 batch；不处理 abort。"""
        req_nums = [None]
        dist.broadcast_object_list(req_nums, src=0, group=self.mulitnode_group)
        req_num = req_nums[0]
        if req_num == 0:
            return None

        req_ids = [None for _ in range(req_num)]
        dist.broadcast_object_list(req_ids, src=0, group=self.mulitnode_group)

        id_to_req = {req.request_id: req for req in self.req_queue.waiting_req_list}
        local_ready = [1 if req_id in id_to_req else 0 for req_id in req_ids]
        select_marks = self._multinode_tp_allreduce_ready_marks(local_ready)

        select_reqs = []
        for req_id, select in zip(req_ids, select_marks):
            if select == 1:
                select_reqs.append(id_to_req[req_id])

        handled_req_ids = {req.request_id for req in select_reqs}
        if handled_req_ids:
            self.req_queue.waiting_req_list = [
                req for req in self.req_queue.waiting_req_list if req.request_id not in handled_req_ids
            ]

        if not select_reqs:
            return None
        return Batch(-1, reqs=select_reqs, dp_size_in_node=self.dp_size_in_node)

    def _multinode_tp_allreduce_ready_marks(self, local_marks: List[int]) -> List[int]:
        """对各节点「waiting 是否已有该 req」做 MIN all_reduce：全员 ready 才为 1。"""
        marks = torch.tensor(local_marks, dtype=torch.int32, device="cpu")
        dist.all_reduce(marks, op=dist.ReduceOp.MIN, group=self.mulitnode_group)
        return marks.tolist()

    # ==================================================================
    # Public: StopStr
    # ==================================================================

    def get_stop_str_matched_reqs_from_running_batch_multinode_tp(self) -> List[Req]:
        """同步 stop_str_matched 状态，返回所有节点共同确认的待停止请求。"""
        running_reqs = [] if self.running_batch is None else list(self.running_batch.reqs)
        id_to_req = {req.request_id: req for req in running_reqs}

        local_matched_req_ids = [
            req_id for req_id, req in id_to_req.items() if req.stop_str_matched and not req._router_stop_str_matched
        ]

        matched_req_ids = self._allgather_stop_str_matched_req_ids(local_matched_req_ids)

        ans = []
        for req_id in matched_req_ids:
            req = id_to_req.get(req_id)
            if req is not None and not req._router_stop_str_matched:
                req._router_stop_str_matched = True
                ans.append(req)
        return ans

    # ==================================================================
    # Private: Abort
    # ==================================================================

    def _filter_aborted_from_schedule_new_batch(self):
        """Abort 阶段1：对 merge 后的 schedule_new_batch 同步 abort，并释放已 abort 请求。"""
        reqs = [] if self.schedule_new_batch is None else list(self.schedule_new_batch.reqs)
        # master 用本地 reqs 作为权威源；slave 传空 list，只接收 broadcast 并打标
        if self.is_multinode_tp_master:
            aborted_req_ids = self._broadcast_aborted_req_ids_from_master(reqs)
        else:
            aborted_req_ids = self._broadcast_aborted_req_ids_from_master([])
            for req in reqs:
                if req.request_id in aborted_req_ids:
                    req.is_aborted = True

        if not aborted_req_ids:
            return
        if self.schedule_new_batch is None:
            logger.warning(
                f"aborted_req_ids non-empty but schedule_new_batch is None, "
                f"skip release, aborted_ids={sorted(aborted_req_ids)}"
            )
            return

        for req_id in aborted_req_ids:
            if req_id not in self.schedule_new_batch.id_to_reqs:
                continue
            req = self.schedule_new_batch.pop_req(req_id)
            self.req_queue.release_aborted_req(req)

        if self.schedule_new_batch.is_clear():
            self.schedule_new_batch = None
        return

    def _broadcast_aborted_req_ids_from_master(self, reqs: List[Req]) -> Set[int]:
        """以 master 侧 ``reqs`` 中的 ``is_aborted`` 为源，broadcast 到所有节点。"""
        local_aborted_req_ids = [req.request_id for req in reqs if req.is_aborted]
        if not self.is_multinode_tp_master:
            local_aborted_req_ids = []

        aborted_req_num = torch.tensor([len(local_aborted_req_ids)], dtype=torch.int64, device="cpu")
        dist.broadcast(aborted_req_num, src=0, group=self.mulitnode_group)
        aborted_req_num = int(aborted_req_num.item())
        if aborted_req_num == 0:
            return set()

        if self.is_multinode_tp_master:
            aborted_req_ids = torch.tensor(local_aborted_req_ids, dtype=torch.int64, device="cpu")
        else:
            aborted_req_ids = torch.empty(aborted_req_num, dtype=torch.int64, device="cpu")
        dist.broadcast(aborted_req_ids, src=0, group=self.mulitnode_group)
        return {int(req_id) for req_id in aborted_req_ids.tolist()}

    def _allgather_stop_str_matched_req_ids(self, local_matched_req_ids: List[int]) -> List[int]:
        """各节点独立提供本地 stop_str_matched req_ids，all_gather 后取交集并排序。

        取交集的原因：
        - 多机 TP 下同一请求的各 TP shard 可能分布在不同节点；
        - 只有当一个请求在所有节点的 running_batch 中都 ``stop_str_matched=True`` 时，
          才认为该请求真正匹配停止字符串，需要下发 StopStrMatchedReqCmd；
        - 若某节点未匹配到而其他节点匹配到，说明该请求的 detokenization 状态尚不一致，
          应等下一轮调度再确认，避免不一致的下发。

        排序原因：
        - 保证各节点返回的 req 顺序一致，避免因集合迭代顺序不同导致下游处理顺序不一致。
        """
        gathered = [None] * self.nnodes
        dist.all_gather_object(gathered, local_matched_req_ids, group=self.mulitnode_group)
        all_matched = set(gathered[0])
        for ids in gathered[1:]:
            all_matched &= set(ids)
        return sorted(all_matched)
