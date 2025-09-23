import torch.multiprocessing as mp
import random
from typing import List, Tuple, Optional
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.pd_io_struct import NIXLChunckedTransTask
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import ChunkedPrefillBackend

logger = init_logger(__name__)


class NIXLChunckedPrefillForPrefillNode(ChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__()
        self.support_overlap = False
        self.info_queue: mp.Queue = info_queue
        self.mem_queue: mp.Queue = mem_queue
        self.classed_req_no_decode = True
        self.nixl_prefill_chuncked_handle_func = self._prefill_chuncked_handle_func

    def init_custom(self):
        assert kv_trans_use_p2p()

        if kv_trans_use_p2p():
            from ..p2p_fix import reduce_tensor

            mp.reductions.reduce_tensor.__code__ = reduce_tensor.__code__

        # 将当前的内存管理器放入到队列中，供kv传输进程获取后使用
        for _ in range(self.node_world_size):
            self.mem_queue.put(self.model.mem_manager)
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
            prefill_finished = req_obj.shm_req.input_len <= req_obj.cur_kv_len
            if prefill_finished:
                # 等待所有传输任务都已经完成。
                if req_obj.nixl_pd_task_num == (req_obj.nixl_pd_task_failed_num + req_obj.nixl_pd_task_sunccess_num):
                    ans_list.append(req_obj)
            else:
                if req_obj.infer_aborted:
                    if req_obj.nixl_pd_task_num == (
                        req_obj.nixl_pd_task_failed_num + req_obj.nixl_pd_task_sunccess_num
                    ):
                        ans_list.append(req_obj)
                    else:
                        continue
                else:
                    ans_list.append(req_obj)
        return ans_list

    def _prefill_chuncked_handle_func(
        self, req_obj: InferReq, next_token_id: int, next_token_prob: float, output_len: int
    ):
        """
        在每一步chuncked prefill 后，尝试生成chuncked 传输任务，发个 kv_move_manager 进行处理。
        """
        # 系统内部的 health 请求不创建 kv 传输任务。
        if req_obj.req_id < 0:
            return

        assert req_obj.cur_kv_len <= req_obj.shm_req.input_len
        input_len = req_obj.shm_req.input_len
        page_size = self.args.nixl_pd_kv_page_size
        prefill_finished = req_obj.cur_kv_len == input_len
        trans_task_list: List[NIXLChunckedTransTask] = []
        while req_obj.nixl_trans_kv_start_index < req_obj.cur_kv_len:
            cur_page_size = min(page_size, req_obj.cur_kv_len - req_obj.nixl_trans_kv_start_index)
            # 生成页面传输任务， 放入kv move manager 的处理队列中
            if cur_page_size == page_size or prefill_finished:
                trans_task = self._create_nixl_trans_task(
                    req_obj=req_obj,
                    kv_start_index=req_obj.nixl_trans_kv_start_index,
                    kv_end_index=req_obj.nixl_trans_kv_start_index + cur_page_size,
                )
                req_obj.nixl_trans_kv_start_index += cur_page_size
                trans_task_list.append(trans_task)
            else:
                break

        if prefill_finished and len(trans_task_list) != 0 and output_len == 1:
            trans_task_list[-1].first_gen_token_id = next_token_id
            trans_task_list[-1].first_gen_token_logprob = next_token_prob

        if self.is_master_in_dp:
            for trans_task in trans_task_list:
                self.info_queue.put(trans_task)
        return

    def _create_nixl_trans_task(
        self, req_obj: InferReq, kv_start_index: int, kv_end_index: int
    ) -> NIXLChunckedTransTask:
        # 确定传输设备
        if req_obj.nixl_trans_device_id == -1:
            req_obj.nixl_trans_device_id = random.randint(0, self.node_world_size - 1)

        nixl_decode_node_info = req_obj.sampling_param.nixl_decode_node
        mem_indexes = (
            self.model.req_manager.req_to_token_indexs[req_obj.req_idx, kv_start_index:kv_end_index]
            .detach()
            .cpu()
            .tolist()
        )
        trans_task = NIXLChunckedTransTask(
            request_id=req_obj.req_id,
            start_kv_index=kv_start_index,
            end_kv_index=kv_end_index,
            time_out_secs=82,
            pd_master_node_id=req_obj.sampling_param.pd_master_node_id,
            prefill_dp_index=self.dp_rank_in_node,
            decode_dp_index=None,
            src_device_id=req_obj.nixl_trans_device_id,
            dst_device_id=None,
            mem_indexes=mem_indexes,
            prefill_agent_name=None,
            prefill_agent_metadata=None,
            prefill_num_pages=None,
            prefill_page_reg_desc=None,
            decode_agent_name=nixl_decode_node_info.agent_name,
            decode_agent_metadata=nixl_decode_node_info.agent_metadata,
            decode_num_pages=nixl_decode_node_info.num_pages,
            decode_page_reg_desc=nixl_decode_node_info.page_reg_desc,
            first_gen_token_id=None,
            first_gen_token_logprob=None,
        )
        req_obj.nixl_pd_task_num += 1
        return trans_task
