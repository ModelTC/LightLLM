import random
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import PDChunckedTransTask, PDChunckedTransTaskGroup, PDAbortReq
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import ChunkedPrefillBackend
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import kv_trans_use_p2p

logger = init_logger(__name__)


class PDDecodeNode(ChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue) -> None:
        super().__init__()
        self.info_queue: mp.Queue = info_queue
        self.classed_req_strict_prefill = False

    def init_custom(self):
        assert kv_trans_use_p2p()

        # TODO 如何支持不支持 P2P的场景
        return

    def _init_reqs(self, reqs: List[Tuple]):
        """
        替换请求初始化操作，替换为 Decode 节点独有的一些特殊初始化流程
        """
        if self.dp_size_in_node != 1:
            dp_rank_in_node = self.dp_rank_in_node
            reqs = [req for req in reqs if req[3] == dp_rank_in_node]

        uninit_reqs = g_infer_context.add_reqs(reqs, init_prefix_cache=True)
        # 匹配radix cache，并更新一些资源的管理。
        self._post_init_reqs(uninit_reqs=uninit_reqs)

        # pd nixl 的 decode 节点模式下当前不支持 cpu cache, 未来可能会支持。
        assert not self.args.enable_cpu_cache

        req_ids = [e[0] for e in reqs]
        return req_ids

    def _post_init_reqs(self, uninit_reqs: List[InferReq]):
        """
        检查请求的 kv len 将可能有问题的请求立即结束掉
        """
        if len(uninit_reqs) == 0:
            return

        for req_obj in uninit_reqs:
            req_obj: InferReq = req_obj  # for easy typing
            # 构建 chuncked trans task
            self._decode_node_gen_trans_tasks(req_obj=req_obj)

        return

    def _filter_not_ready_reqs(self, req_ids: List[int]) -> List[InferReq]:
        """
        将错误请求从 req_ids 中过滤出来, 然后让 _get_classed_reqs 进行处理。 该函数
        主要用于在 nixl pd 分离模式下, 由子类继承重载, prefill 和 decode 节点过滤 kv 传输错误，或者 kv
        传输没有完成的请求。
        """
        self._allocate_ready_kv(req_ids)
        ans_list: List[InferReq] = []
        for request_id in req_ids:
            req_obj: InferReq = g_infer_context.requests_mapping[request_id]

            # D 节点的请求未结束时会反复进入该过滤逻辑，最多发送 6 次
            # PDAbortReq，既提高 abort 消息被传输层处理的概率，也避免持续重复发送。
            pd_abort_req_send_count = getattr(req_obj, "pd_abort_req_send_count", 0)
            if (
                self.is_master_in_dp
                and req_obj.infer_aborted
                and req_obj.pd_task_num != 0
                and pd_abort_req_send_count < 6
            ):
                self.info_queue.put(PDAbortReq(request_id=req_obj.req_id, device_id=req_obj.pd_trans_device_id))
                req_obj.pd_abort_req_send_count = pd_abort_req_send_count + 1

            if req_obj.pd_task_num != (req_obj.pd_task_failed_num + req_obj.pd_task_success_num):
                continue

            if req_obj.pd_task_failed_num > 0:
                # KV 传输失败：强制补 finish token 并结束。
                # abort 优先标 ABORTED；纯传输错误标 ERROR（不再误用 STOP）。
                if not req_obj.finish_status.is_finished():
                    finish_status = (
                        FinishStatus.FINISHED_ABORTED if req_obj.infer_aborted else FinishStatus.FINISHED_ERROR
                    )
                    req_obj.finish_status.set_status(finish_status)
                    if self.is_master_in_dp:
                        # 内部统一在已有输出末尾追加 EOS 作为 finish token。
                        req_obj.shm_req.mark_simulated_finished(
                            finish_status,
                            output_len=req_obj.cur_output_len,
                        )
                        logger.error(
                            f"req_id: {req_obj.req_id} forced to finished "
                            f"(reason={req_obj.finish_status.get_finish_reason()}), kv transfer error"
                        )

                # 提前释放有问题的 mem_index。cur_kv_len 只表示已经传输完成的
                # 逻辑长度，hold_kv_len 还包含最后一个页面中尚未使用的预留槽位。
                old_prefix_len = 0 if req_obj.shared_kv_node is None else req_obj.shared_kv_node.node_prefix_total_len
                error_mem_len = req_obj.hold_kv_len - old_prefix_len
                if error_mem_len > 0:
                    mem_indexes = (
                        self.model.req_manager.req_to_token_indexs[
                            req_obj.req_idx, old_prefix_len : req_obj.hold_kv_len
                        ]
                        .detach()
                        .cpu()
                    )
                    self.model.mem_manager.free(mem_indexes)
                    req_obj.cur_kv_len = old_prefix_len
                    req_obj.hold_kv_len = old_prefix_len
                    if self.is_master_in_dp:
                        req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len

            ans_list.append(req_obj)
        return ans_list

    def _decode_node_gen_trans_tasks(self, req_obj: InferReq):
        """
        decode node 生成所有的传输任务对象。
        """
        group = PDChunckedTransTaskGroup()
        input_len = req_obj.shm_req.input_len
        assert self.args.pd_kv_page_size % self.args.page_size == 0, "pd_kv_page_size must be divisible by page_size"
        deferred = input_len - req_obj.cur_kv_len > 1
        req_obj.pd_first_chunk_ready = False
        req_obj.pd_pending_kv_group = None
        if deferred and req_obj.hold_kv_len:
            # Waiting requests must not pin partial prefixes and prevent each
            # other from acquiring their complete destination. Full hits keep
            # the existing no-transfer path.
            if g_infer_context.is_hybrid_att_model:
                torch.cuda.current_stream().synchronize()
            free_indexes = []
            g_infer_context._free_req_mem_without_radix_insert(free_indexes, req_obj)
            for indexes in free_indexes:
                if len(indexes):
                    self.model.mem_manager.free(indexes)
            req_obj.cur_kv_len = req_obj.hold_kv_len = 0
            if self.is_master_in_dp:
                req_obj.shm_req.shm_cur_kv_len = req_obj.shm_req.prompt_cache_len = 0
        if self.is_master_in_dp:
            req_obj.shm_req.pd_waiting_for_kv = deferred
        # 当 decode 节点不能匹配足够的kv的时候，才进行真实的 kv 传输。
        if input_len - req_obj.cur_kv_len > 1:
            trans_page_size = self.args.pd_kv_page_size
            assert trans_page_size % self.args.page_size == 0, "pd_kv_page_size must be divisible by page_size"
            req_obj.pd_trans_kv_start_index = req_obj.cur_kv_len
            assert req_obj.hold_kv_len == req_obj.cur_kv_len
            # Metadata only: these placeholders cannot reach a copy kernel
            # until the first WRITE request has obtained a real allocation.
            mem_indexes = [-1] * (input_len - req_obj.cur_kv_len)

            while req_obj.pd_trans_kv_start_index < input_len:
                cur_page_size = min(trans_page_size, input_len - req_obj.pd_trans_kv_start_index)
                # 生成页面传输任务， 放入kv move manager 的处理队列中
                start_index = req_obj.pd_trans_kv_start_index
                end_index = req_obj.pd_trans_kv_start_index + cur_page_size
                page_mem_indexes = mem_indexes[start_index - req_obj.cur_kv_len : end_index - req_obj.cur_kv_len]
                self._create_pd_trans_task(
                    req_obj=req_obj,
                    mem_indexes=page_mem_indexes,
                    kv_start_index=start_index,
                    kv_end_index=end_index,
                    group=group,
                )
                # update
                req_obj.pd_trans_kv_start_index += cur_page_size

            # 混合注意力模型还需接收请求运行态 buffer（如 linear attention 的 conv/SSM 状态）。
            # 通过本地 req_idx 定位运行态 buffer 的恢复位置。
            if g_infer_context.is_hybrid_att_model:
                self._create_pd_trans_task(
                    req_obj=req_obj,
                    mem_indexes=[],
                    kv_start_index=input_len,
                    kv_end_index=input_len,
                    group=group,
                    page_kind="att_state",
                )
        else:
            assert req_obj.cur_kv_len == input_len - 1

        if not group.task_list:
            # 需要上报一个包含 0 长度的trans task，触发 kv move manager 给 pd master 上报
            # upkv_status 状态，使推理流程完整。
            self._create_pd_trans_task(
                req_obj=req_obj,
                mem_indexes=[],
                kv_start_index=req_obj.cur_kv_len,
                kv_end_index=req_obj.cur_kv_len,
                group=group,
            )

        if deferred:
            req_obj.pd_pending_kv_group = group
            for task in group.task_list:
                task.defer_allocation = True
        if self.is_master_in_dp:
            self.info_queue.put(group)
        return

    @staticmethod
    def _kv_growth_reservation(req, args):
        # Match Router's output estimate and page/MTP/asynchronous-exit margin.
        output_len = req.cur_output_len
        params = req.sampling_param.shm_param
        estimate = (
            params.max_new_tokens if params.ignore_eos else min(params.max_new_tokens, max(int(1.1 * output_len), 64))
        )
        a_len = max(req.shm_req.input_len + output_len + 1, req.cur_kv_len + 1)
        b_len = max(0, estimate - output_len - 1) + args.page_size + 2 * (args.mtp_step + 1) + 16
        return max(0, a_len + b_len - max(req.shm_req.input_len, req.hold_kv_len))

    def _allocate_ready_kv(self, req_ids):
        for request_id in req_ids:
            req = g_infer_context.requests_mapping[request_id]
            group = getattr(req, "pd_pending_kv_group", None)
            if group is None or not req.pd_first_chunk_ready or req.infer_aborted or req.pd_task_failed_num:
                continue
            needed = req._kv_cache_alloc_need(req.shm_req.input_len)
            growth = sum(
                PDDecodeNode._kv_growth_reservation(r, self.args)
                for r in g_infer_context.requests_mapping.values()
                if getattr(r, "pd_pending_kv_group", None) is None
            ) + PDDecodeNode._kv_growth_reservation(req, self.args)
            available = g_infer_context.get_can_alloc_token_num() - growth
            if self.dp_world_size > 1:
                from lightllm.distributed import dist_group_manager

                budget = torch.tensor([available], dtype=torch.int64, device="cuda")
                dist.all_reduce(budget, op=dist.ReduceOp.MIN, group=dist_group_manager.get_default_group().device_group)
                available = budget.item()
            if needed > available:
                continue
            if needed:
                self._alloc_req_kv_mem(req, needed)
            indexes = self.model.req_manager.req_to_token_indexs[req.req_idx, : req.shm_req.input_len].tolist()
            tasks = []
            for original in group.task_list:
                task = copy.copy(original)
                task.defer_allocation = False
                if task.page_kind == "kv":
                    task.mem_indexes = indexes[task.start_kv_index : task.end_kv_index]
                tasks.append(task)
            req.cur_kv_len = req.shm_req.input_len
            req.pd_pending_kv_group = None
            if self.is_master_in_dp:
                req.shm_req.pd_waiting_for_kv = False
                self.info_queue.put(PDChunckedTransTaskGroup(tasks, allocation_update=True))

    def _create_pd_trans_task(
        self,
        req_obj: InferReq,
        mem_indexes: List[int],
        kv_start_index: int,
        kv_end_index: int,
        group: PDChunckedTransTaskGroup,
        page_kind: str = "kv",
    ):
        # 确定传输设备
        if req_obj.pd_trans_device_id == -1:
            if not hasattr(self, "pd_iter_device_id"):
                self.pd_iter_device_id = 0
            req_obj.pd_trans_device_id = self.pd_iter_device_id
            # only self.is_master_in_dp will be used.
            self.pd_iter_device_id = (self.pd_iter_device_id + 1) % self.node_world_size

        if page_kind == "kv":
            req_idx = None
        elif page_kind == "att_state":
            req_idx = req_obj.req_idx
        else:
            raise ValueError(f"unknown PD trans page kind {page_kind}")

        trans_task = PDChunckedTransTask(
            request_id=req_obj.req_id,
            start_kv_index=kv_start_index,
            end_kv_index=kv_end_index,
            time_out_secs=180,
            pd_master_node_id=req_obj.sampling_param.pd_master_node_id,
            prefill_dp_index=None,
            decode_dp_index=self.dp_rank_in_node,
            src_device_id=None,
            dst_device_id=req_obj.pd_trans_device_id,
            mem_indexes=mem_indexes,
            prefill_agent_name=None,
            prefill_agent_metadata=None,
            prefill_num_pages=None,
            prefill_page_reg_desc=None,
            decode_agent_name=None,
            decode_agent_metadata=None,
            decode_num_pages=None,
            decode_page_reg_desc=None,
            first_gen_token_id=None,
            first_gen_token_logprob=None,
            page_kind=page_kind,
            req_idx=req_idx,
        )
        group.task_list.append(trans_task)
        req_obj.pd_task_num += 1
        return
