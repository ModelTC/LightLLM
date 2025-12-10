import torch
import torch.distributed as dist
import rpyc
from typing import Dict, List, Tuple
from rpyc.utils.classic import obtain
from .prefill_impl import ChunckedPrefillForPrefillNode
from lightllm.common.basemodel.infer_lock import g_router_lock, acquire_lock_until_ready, release_acquired_lock
from .prefill_task_cache import g_kv_move_task_cache
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PDPrefillInferRpcServer(rpyc.Service):
    def __init__(self, backend: ChunckedPrefillForPrefillNode) -> None:
        super().__init__()
        self.backend = backend
        self.device_id = self.backend.current_device_id
        self.dp_rank_in_node = self.backend.dp_rank_in_node
        self.is_master_in_dp = self.backend.is_master_in_dp
        return

    def on_connect(self, conn):
        torch.cuda.set_device(f"cuda:{self.device_id}")
        return

    # pd 分离模式会使用的一些接口，用于做一些全局信息管理
    def exposed_remove_req_refs_from_prompt_cache(self, group_req_ids: List[int]):
        group_req_ids = obtain(group_req_ids)
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        for group_req_id in group_req_ids:
            if group_req_id in g_kv_move_task_cache:
                task, share_node = g_kv_move_task_cache.pop(group_req_id)
                if share_node is not None:
                    self.backend.radix_cache.dec_node_ref_counter(share_node)
                # 减少日志数量
                if self.is_master_in_dp:
                    logger.info(f"unfrozen tokens for req id: {group_req_id}")

            # 更新调度元数据
            if self.is_master_in_dp:
                with g_router_lock.obj:
                    self.backend.shared_token_load.add_frozened_token_count(
                        -len(task.input_tokens), self.dp_rank_in_node
                    )
        release_acquired_lock()
        return
