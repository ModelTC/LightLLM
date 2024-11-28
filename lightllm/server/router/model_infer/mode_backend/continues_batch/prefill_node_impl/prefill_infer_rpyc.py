import torch
import torch.distributed as dist
import rpyc
from typing import Dict, List, Tuple
from rpyc.utils.classic import obtain
from .prefill_impl import ContinuesBatchBackendForPrefillNode
from lightllm.common.basemodel.infer_lock import g_router_lock, acquire_lock_until_ready, release_acquired_lock
from .prefill_task_cache import g_kv_move_task_cache
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PDPrefillInferRpcServer(rpyc.Service):
    def __init__(self, backend: ContinuesBatchBackendForPrefillNode) -> None:
        super().__init__()
        self.backend = backend
        return

    def on_connect(self, conn):
        self.rank_id = dist.get_rank()
        torch.cuda.set_device(f"cuda:{self.rank_id}")
        return

    # pd 分离模式会使用的一些接口，用于做一些全局信息管理
    def exposed_remove_req_refs_from_prompt_cache(self, group_req_id: int):
        group_req_id = obtain(group_req_id)
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        if group_req_id in g_kv_move_task_cache:
            task, share_node = g_kv_move_task_cache.pop(group_req_id)
            if share_node is not None:
                self.backend.radix_cache.dec_node_ref_counter(share_node)
            logger.info(f"unfrozen tokens for req id: {group_req_id}")

        # 更新元数据
        if self.rank_id < self.backend.dp_size:
            with g_router_lock.obj:
                self.backend.shared_token_load.add_frozened_token_count(-len(task.input_tokens), self.rank_id)

        release_acquired_lock()
        return

    def exposed_put_mem_manager_to_mem_queue(self):
        self.backend.mem_queue.put(self.backend.model.mem_manager)
        logger.info("put mem manager to mem_queue ok")
        return
