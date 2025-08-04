import threading
import torch.distributed as dist
from collections import deque
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
from lightllm.utils.envs_utils import get_env_start_args
from ..infer_batch import InferReq
from lightllm.utils.dist_utils import create_new_group_for_current_dp


class MultiLevelCacheManager(object):
    def __init__(self, backend):
        self.args = get_env_start_args()
        from .base_backend import ModeBackend

        self.backend: ModeBackend = backend
        self.gloo_group = create_new_group_for_current_dp("gloo")

        self.cpu_cache_handle_queue = deque()
        self.cpu_cache_client = CpuKvCacheClient(init_shm_data=False)
        self.cpu_cache_thread = threading.Thread(target=self.cpu_cache_handle_loop, daemon=True)
        self.cpu_cache_thread.start()

    def cpu_cache_handle_loop(self):
        pass

    def req_to_cpu_cache(self, req: InferReq):
        all_token_hash_list = req.shm_req.token_hash_list.get_all()
        block_size = req.cur_kv_len // self.args.cpu_cache_token_chuncked_size
        move_block_size = min(block_size, len(all_token_hash_list))
        if move_block_size == 0:
            req.cpu_cache_task_finished = True
            return
        if self.backend.is_master_in_dp:
            self.cpu_cache_client.lock.acquire_sleep1ms()
            page_list, ready_list = self.cpu_cache_client.allocate_pages(
                all_token_hash_list[:move_block_size],
                disk_offload_enable=self.args.enable_disk_cache,
            )
            self.cpu_cache_client.lock.release()
            item_size = len(page_list)
            dist.broadcast_object_list([item_size], group=self.gloo_group, group_src=0)
            if item_size == 0:
                req.cpu_cache_task_finished = True
                return
            dist.broadcast_object_list(page_list, group=self.gloo_group, group_src=0)
            dist.broadcast_object_list(ready_list, group=self.gloo_group, group_src=0)
        else:
            recv_list = [None]
            dist.broadcast_object_list(recv_list, group=self.gloo_group, group_src=0)
            item_size = recv_list[0]
            if item_size == 0:
                req.cpu_cache_task_finished = True
                return
            page_list = [None] * item_size
            ready_list = [None] * item_size
            dist.broadcast_object_list(page_list, group=self.gloo_group, group_src=0)
            dist.broadcast_object_list(ready_list, group=self.gloo_group, group_src=0)

        # to do 将 gpu tensor 进行复制，复制 cpu cache tensor 中
        pass

        return
