import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional, List, Deque
from collections import deque
from functools import lru_cache
from lightllm.server.x2i_server.past_kv_cache_client import PastKVCacheClient
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu_for_x2i
from lightllm.server.core.objs.token_chunck_hash_list import LIGHTLLM_TOKEN_HASH_LIST_SIZE

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclass
class TransTask:
    req_obj: InferReq
    sync_event: torch.cuda.Event


class PastKVCacheModule(object):
    def __init__(self, backend):
        from .base_backend import ModeBackend
        self.backend: ModeBackend = backend
        self.past_kv_cache_client = PastKVCacheClient(only_create_meta_data=False, init_shm_data=False)
        self.page_index_buffer = torch.empty((LIGHTLLM_TOKEN_HASH_LIST_SIZE * 2,), dtype=torch.int32, device="cuda")
        self.past_kv_cache_task: Deque[TransTask] = deque()
        self.sync_task_status_group = create_new_group_for_current_dp("gloo")

    @lru_cache()
    def need_sync_compute_stream(self) -> bool:
        """
        fa3 在 offload 和 load kv cache 的时候，需要等待计算流完成，否则可能会概率崩溃。
        """

        model = self.backend.model
        att_backends = [
            model.prefill_att_backend,
            model.decode_att_backend,
            model.prefill_att_backend1,
            model.decode_att_backend1,
        ]
        for att_backend in att_backends:
            if att_backend is not None and "fa3" in att_backend.__class__.__name__.lower():
                logger.info("PastKVCacheModule: need sync compute stream for fa3 backend.")
                return True
        logger.info("PastKVCacheModule: no need sync compute stream.")
        return False


    def offload_finished_reqs_to_past_kv_cache(self, finished_reqs: List[InferReq]) -> List[InferReq]:
        """
        Offload the finished reqs to past kv cache, and return the truly finished reqs that can be freed in infer batch.
        """
        true_finished_reqs = []
        for req in finished_reqs:
            # filter out non-img-gen reqs
            if not req.shm_req.sample_params.img_gen_prefill:
                true_finished_reqs.append(req)
                continue

            if req.past_kv_cache_task_status.is_finished():
                true_finished_reqs.append(req)
                continue

            if req.past_kv_cache_task_status.is_running():
                continue

            assert req.past_kv_cache_task_status.is_not_started()

            if self.need_sync_compute_stream():
                g_infer_context.get_overlap_stream().synchronize()

            trans_task = self._start_kv_cache_offload(req=req)
            assert trans_task is not None
            self.past_kv_cache_task.append(trans_task)


        return true_finished_reqs

    def _start_kv_cache_offload(self, req: InferReq) -> Optional[TransTask]:

        with torch.cuda.stream(g_infer_context.get_cpu_kv_cache_stream()):
            page_indexes = torch.tensor(req.shm_req.past_kv_cache_page_indexes.get_all(), dtype=torch.int32, device='cpu', pin_memory=True)
            num_tokens = req.shm_req.input_len

            assert req.cur_kv_len >= num_tokens
            assert num_tokens <= len(page_indexes) * self.past_kv_cache_client.token_page_size

            cuda_page_indexes = self.page_index_buffer[:len(page_indexes)]
            cuda_page_indexes.copy_(page_indexes)

            token_indexes = self.backend.model.req_manager.req_to_token_indexs[req.req_idx, 0: num_tokens]
            mem_manager = self.backend.model.mem_manager


            if hasattr(mem_manager, "scale_buffer") and mem_manager.scale_buffer is not None:
                cpu_cache_meta = self.past_kv_cache_client.kv_cache_tensor_meta
                cpu_kv_cache = self.past_kv_cache_client.cpu_kv_cache_tensor[:, :, :, :, 0:cpu_cache_meta.head_dim]
                cpu_kv_cache_scale = self.past_kv_cache_client.cpu_kv_cache_tensor[
                    :, :, :, :, cpu_cache_meta.head_dim
                ].view(mem_manager.scale_buffer.dtype)
                gpu_kv_cache_scale = mem_manager.scale_buffer
            else:
                cpu_kv_cache = self.past_kv_cache_client.cpu_kv_cache_tensor
                cpu_kv_cache_scale = None
                gpu_kv_cache_scale = None

            grid_num = 16
            offload_gpu_kv_to_cpu_for_x2i(
                token_indexes=token_indexes,
                gpu_kv_cache=mem_manager.kv_buffer,
                gpu_kv_cache_scale=gpu_kv_cache_scale,
                cpu_kv_cache=cpu_kv_cache,
                cpu_kv_cache_scale=cpu_kv_cache_scale,
                page_indexes=cuda_page_indexes,
                tp_index=self.backend.rank_in_dp,
                tp_world_size=self.backend.dp_world_size,
                grid_num=grid_num,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()
            req.past_kv_cache_task_status = InferReq._CpuCacheTaskStatus.RUNNING
            return TransTask(
                req_obj=req,
                sync_event=sync_event,
            )

    def update_past_kv_cache_task_states(self):
        trans_ok_tasks = []
        while len(self.past_kv_cache_task) > 0:
            task: TransTask = self.past_kv_cache_task.popleft()
            if task.sync_event.query():
                trans_ok_tasks.append(task)
            else:
                self.past_kv_cache_task.appendleft(task)
                break

        ok_tasks_num = torch.tensor(len(trans_ok_tasks))
        dist.all_reduce(ok_tasks_num, op=dist.ReduceOp.MIN, group=self.sync_task_status_group)

        if ok_tasks_num.item() > 0:
            finished, unfinished = trans_ok_tasks[:ok_tasks_num.item()], trans_ok_tasks[ok_tasks_num.item():]
            self.past_kv_cache_task.extendleft(reversed(unfinished))
            for task in finished:
                task.req_obj.past_kv_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
