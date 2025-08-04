import threading
import torch.distributed as dist
import torch
import dataclasses
from typing import Optional, List
from collections import deque
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
from lightllm.utils.envs_utils import get_env_start_args
from ..infer_batch import InferReq
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu


class MultiLevelCacheManager(object):
    def __init__(self, backend):
        self.args = get_env_start_args()
        from .base_backend import ModeBackend

        self.backend: ModeBackend = backend
        self.gloo_group = create_new_group_for_current_dp("gloo")
        self.filter_group = create_new_group_for_current_dp("gloo")
        self.sync_group = create_new_group_for_current_dp("nccl")

        self.cpu_cache_handle_queue = deque()
        self.cpu_cache_client = CpuKvCacheClient(init_shm_data=False)

    def req_to_cpu_cache_task(self, req: InferReq, cpu_kv_cache_stream: torch.cuda.Stream) -> Optional["TransTask"]:
        with torch.cuda.stream(cpu_kv_cache_stream):
            all_token_hash_list = req.shm_req.token_hash_list.get_all()
            block_size = req.cur_kv_len // self.args.cpu_cache_token_chuncked_size
            move_block_size = min(block_size, len(all_token_hash_list))
            if move_block_size == 0:
                req.cpu_cache_task_finished = True
                return None
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
                    return None
                dist.broadcast_object_list(page_list, group=self.gloo_group, group_src=0)
                dist.broadcast_object_list(ready_list, group=self.gloo_group, group_src=0)
            else:
                recv_list = [None]
                dist.broadcast_object_list(recv_list, group=self.gloo_group, group_src=0)
                item_size = recv_list[0]
                if item_size == 0:
                    req.cpu_cache_task_finished = True
                    return None
                page_list = [None] * item_size
                ready_list = [None] * item_size
                dist.broadcast_object_list(page_list, group=self.gloo_group, group_src=0)
                dist.broadcast_object_list(ready_list, group=self.gloo_group, group_src=0)

            page_indexes = torch.tensor(page_list, dtype=torch.int32, device="cpu", pin_memory=True)
            page_readies = torch.tensor(ready_list, dtype=torch.bool, device="cpu", pin_memory=True)

            token_indexes = self.backend.model.req_manager.req_to_token_indexs[req.req_idx, 0 : req.cur_kv_len]
            offload_gpu_kv_to_cpu(
                token_indexes=token_indexes,
                gpu_kv_cache=self.backend.model.mem_manager.kv_buffer,
                cpu_kv_cache=self.cpu_cache_client.cpu_kv_cache_tensor,
                page_indexes=page_indexes,
                page_readies=page_readies,
            )
            dist.barrier(group=self.sync_group)
            sync_event = torch.cuda.Event()
            sync_event.record()

            trans_task = TransTask(
                page_indexes=page_indexes, page_readies=page_readies, req_obj=req, sync_event=sync_event
            )

        return trans_task

    def handle_task_queue(self):
        if self.backend.is_master_in_dp:
            trans_ok_reqs = []
            while len(self.cpu_cache_handle_queue) != 0:
                task: TransTask = self.cpu_cache_handle_queue.popleft()
                if task.sync_event.query():
                    trans_ok_reqs.append(task)
                else:
                    self.cpu_cache_handle_queue.appendleft(task)
                    break
            item_size = len(trans_ok_reqs)
            dist.broadcast_object_list([item_size], group=self.filter_group, group_src=0)

        else:
            recv_list = [None]
            dist.broadcast_object_list(recv_list, group=self.filter_group, group_src=0)
            item_size = recv_list[0]
            trans_ok_reqs: List[TransTask] = [self.cpu_cache_handle_queue.popleft() for _ in range(item_size)]

        if item_size > 0:
            page_array_list = [task.page_indexes for task in trans_ok_reqs]
            page_list = torch.cat(page_array_list, dim=0).tolist()
            self.cpu_cache_client.lock.acquire_sleep1ms()
            self.cpu_cache_client.update_pages_status_to_ready(
                page_list=page_list, deref=True, disk_offload_enable=self.args.enable_disk_cache
            )
            self.cpu_cache_client.lock.release()
            for req in trans_ok_reqs:
                req.req_obj.cpu_cache_task_finished = True
        return


@dataclasses.dataclass
class TransTask:
    page_indexes: torch.Tensor
    page_readies: torch.Tensor
    req_obj: InferReq
    sync_event: torch.cuda.Event
