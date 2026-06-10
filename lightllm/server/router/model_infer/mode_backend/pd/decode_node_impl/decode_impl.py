import random
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
        ans_list: List[InferReq] = []
        for request_id in req_ids:
            req_obj: InferReq = g_infer_context.requests_mapping[request_id]

            if self.is_master_in_dp and req_obj.infer_aborted and req_obj.pd_task_num != 0:
                self.info_queue.put(PDAbortReq(request_id=req_obj.req_id, device_id=req_obj.pd_trans_device_id))

            if req_obj.pd_task_num != (req_obj.pd_task_failed_num + req_obj.pd_task_success_num):
                continue

            if req_obj.pd_task_failed_num > 0:
                # 强制停止
                if not req_obj.finish_status.is_finished():
                    req_obj.cur_output_len += 1
                    req_obj.set_next_gen_token_id(next_token_id=0, logprob=0.0, output_len=req_obj.cur_output_len)
                    req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)

                    if self.is_master_in_dp:
                        req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len
                        req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                        req_obj.shm_req.finish_status.set_status(FinishStatus.FINISHED_STOP)
                        req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

                        logger.error(f"req_id: {req_obj.req_id} forced to finished, it exits kv transfer error")

                # 提前释放有问题的 mem_index
                old_prefix_len = 0 if req_obj.shared_kv_node is None else req_obj.shared_kv_node.node_prefix_total_len
                error_mem_len = req_obj.cur_kv_len - old_prefix_len
                if error_mem_len > 0:
                    req_obj.cur_kv_len -= error_mem_len

                    mem_indexes = (
                        self.model.req_manager.req_to_token_indexs[
                            req_obj.req_idx, req_obj.cur_kv_len : (req_obj.cur_kv_len + error_mem_len)
                        ]
                        .detach()
                        .cpu()
                    )
                    self.model.mem_manager.free(mem_indexes)
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
        # 当 decode 节点不能匹配足够的kv的时候，才进行真实的 kv 传输。
        if input_len - req_obj.cur_kv_len > 1:
            page_size = self.args.pd_kv_page_size
            req_obj.pd_trans_kv_start_index = req_obj.cur_kv_len
            need_mem_size = input_len - req_obj.cur_kv_len

            if need_mem_size > 0:
                if self.radix_cache is not None:
                    self.radix_cache.free_radix_cache_to_get_enough_token(need_mem_size)

                mem_indexes = self.model.req_manager.mem_manager.alloc(need_size=need_mem_size)
                self.model.req_manager.req_to_token_indexs[
                    req_obj.req_idx, req_obj.cur_kv_len : (req_obj.cur_kv_len + need_mem_size)
                ] = mem_indexes

                while req_obj.pd_trans_kv_start_index < input_len:
                    cur_page_size = min(page_size, input_len - req_obj.pd_trans_kv_start_index)
                    # 生成页面传输任务， 放入kv move manager 的处理队列中
                    start_index = req_obj.pd_trans_kv_start_index
                    end_index = req_obj.pd_trans_kv_start_index + cur_page_size
                    page_mem_indexes = mem_indexes[start_index - req_obj.cur_kv_len : end_index - req_obj.cur_kv_len]
                    self._create_pd_trans_task(
                        req_obj=req_obj,
                        mem_indexes=page_mem_indexes.tolist(),
                        kv_start_index=start_index,
                        kv_end_index=end_index,
                        group=group,
                    )
                    # update
                    req_obj.pd_trans_kv_start_index += cur_page_size

                req_obj.cur_kv_len += len(mem_indexes)

                # 如果当前是linear att 混合模型，则需要创建一个linear att 状态的传输任务
                if g_infer_context.is_linear_att_mixed_model:
                    self._create_pd_trans_task(
                        req_obj=req_obj,
                        mem_indexes=[],
                        kv_start_index=input_len,
                        kv_end_index=input_len,
                        group=group,
                        page_kind="linear_att_state",
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

        if self.is_master_in_dp:
            self.info_queue.put(group)
        return

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
        elif page_kind == "linear_att_state":
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
