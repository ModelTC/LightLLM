import random
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import NIXLChunckedTransTask, ChunckedTransTaskGroup
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import ChunkedPrefillBackend
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, g_infer_state_lock
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import kv_trans_use_p2p

logger = init_logger(__name__)


class NIXLDecodeNode(ChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__()
        self.info_queue: mp.Queue = info_queue
        self.mem_queue: mp.Queue = mem_queue
        self.classed_req_strict_prefill = False

    def init_custom(self):

        if kv_trans_use_p2p():
            from ..p2p_fix import reduce_tensor

            mp.reductions.reduce_tensor.__code__ = reduce_tensor.__code__

        # 将当前的内存管理器放入到队列中，供kv传输进程获取后使用
        for _ in range(self.node_world_size):
            self.mem_queue.put(self.model.mem_manager)
        return

    def _init_reqs(self, reqs: List[Tuple]):
        """
        替换请求初始化操作，替换为 Decode 节点独有的一些特殊初始化流程
        """
        if self.dp_size_in_node != 1:
            dp_rank_in_node = self.dp_rank_in_node
            reqs = [req for req in reqs if req[3] == dp_rank_in_node]

        g_infer_state_lock.acquire()

        uninit_reqs = g_infer_context.add_reqs(reqs, init_prefix_cache=False)
        # 匹配radix cache，并更新一些资源的管理。
        self._post_init_reqs(uninit_reqs=uninit_reqs)

        g_infer_state_lock.release()
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
            request_id = req_obj.req_id
            if request_id > 0:
                req_obj._match_radix_cache()
                # 构建 chuncked trans task
                self._decode_node_gen_trans_tasks(req_obj=req_obj)
            else:
                # 对于不合法的请求， 主要是health请求，直接模拟将其finished掉
                req_obj.cur_output_len += 1
                req_obj.set_next_gen_token_id(0, 0.0, 1)
                req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)

                if self.is_master_in_dp:
                    req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
                    req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len
                    req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                    req_obj.shm_req.finish_status.set_status(FinishStatus.FINISHED_STOP)
                    req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

                    req_id = req_obj.shm_req.request_id
                    logger.error(f"req_id: {req_id} forced to finished, it not in g_success_kv_move_task_cache")
        return
    
    def _decode_node_gen_trans_tasks(self, req_obj: InferReq):
        """
        decode node 生成所有的传输任务对象。
        """
        # 传输的 kv 要少一个，不然decode 无法有下一个输入除非推理出下一个token
        input_len = req_obj.shm_req.input_len - 1
        page_size = self.args.nixl_pd_kv_page_size
        req_obj.nixl_trans_kv_start_index = req_obj.cur_kv_len
        
        need_mem_size = input_len - req_obj.cur_kv_len
        if need_mem_size <= 0:
            return
        
        if self.radix_cache is not None:
            self.radix_cache.free_radix_cache_to_get_enough_token(need_mem_size)
        
        mem_indexes = self.model.req_manager.mem_manager.alloc(need_size=need_mem_size)
        self.model.req_manager.req_to_token_indexs[req_obj.req_idx, req_obj.cur_kv_len:(req_obj.cur_kv_len + need_mem_size)] = mem_indexes
         
        group = ChunckedTransTaskGroup() if self.is_master_in_dp else None

        while req_obj.nixl_trans_kv_start_index < input_len:
            cur_page_size = min(page_size, input_len - req_obj.nixl_trans_kv_start_index)
            # 生成页面传输任务， 放入kv move manager 的处理队列中
            start_index = req_obj.nixl_trans_kv_start_index
            end_index = req_obj.nixl_trans_kv_start_index + cur_page_size
            page_mem_indexes = mem_indexes[start_index - req_obj.cur_kv_len : end_index - req_obj.cur_kv_len]
            self._create_nixl_trans_task(req_obj=req_obj,
                                         mem_indexes=page_mem_indexes.tolist(),
                                         kv_start_index=start_index,
                                         kv_end_index=end_index,
                                         group=group)
            # update
            req_obj.nixl_trans_kv_start_index += cur_page_size
            req_obj.nixl_pd_task_num += 1
        
        if self.is_master_in_dp:
            self.info_queue.put(group)
        return
    
    def _create_nixl_trans_task(self, req_obj: InferReq, mem_indexes:List[int], kv_start_index: int, kv_end_index: int, group: ChunckedTransTaskGroup):
        if self.is_master_in_dp:
            # 确定传输设备
            if req_obj.nixl_trans_device_id == -1:
                req_obj.nixl_trans_device_id = random.randint(0, self.node_world_size - 1)
            
            trans_task = NIXLChunckedTransTask(request_id=req_obj.req_id,
                                  start_kv_index=kv_start_index,
                                  end_kv_index=kv_end_index,
                                  prefill_dp_index=None,
                                  decode_dp_index=self.dp_rank_in_node,
                                  src_device_id=None,
                                  dst_device_id=req_obj.nixl_trans_device_id,
                                  mem_indexes=mem_indexes,
                                  peer_agent_name=None,
                                  peer_agent_metadata=None,
                                  peer_num_pages=None,
                                  peer_page_req_desc=None,
                                  peer_page_xfer_handles=None,
                                  nixl_src_page_index=None,
                                  nixl_dst_page_index=None
                                  )
            if group is not None:
                group.task_list.append(trans_task)
            return
