import torch.distributed as dist
import torch
import dataclasses
import bisect
from functools import lru_cache
from typing import Optional, List, Deque
from collections import deque
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
from lightllm.utils.envs_utils import get_env_start_args
from ..infer_batch import InferReq
from lightllm.utils.dist_utils import create_new_group_for_current_dp
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.utils.multimodal_chunk_utils import clamp_mm_cache_len

logger = init_logger(__name__)


class MultiLevelKvCacheModule(object):
    def __init__(self, backend):
        self.args = get_env_start_args()
        assert self.args.cpu_cache_token_page_size % self.args.page_size == 0
        from .base_backend import ModeBackend

        self.backend: ModeBackend = backend
        self.gloo_group = create_new_group_for_current_dp("gloo")
        self.filter_group = create_new_group_for_current_dp("gloo")
        self.init_sync_group = create_new_group_for_current_dp("nccl")
        dist.barrier(group=self.init_sync_group)
        self.offload_sync_group = create_new_group_for_current_dp("nccl")
        dist.barrier(group=self.offload_sync_group)
        self.offload_sync_tensor = torch.empty((1,), dtype=torch.int32, device="cuda")

        self.cpu_cache_handle_queue: Deque[TransTask] = deque()
        self.cpu_cache_client = CpuKvCacheClient(only_create_meta_data=False, init_shm_data=False)
        resources = self.backend.model.att_resource_manager
        meta = self.cpu_cache_client.kv_cache_tensor_meta
        if meta.page_bytes != resources.get_cpu_page_bytes():
            raise ValueError("CPU cache service and model resource page sizes differ")

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
                logger.info("MultiLevelKvCacheModule: need sync compute stream for fa3 backend.")
                return True
        logger.info("MultiLevelKvCacheModule: no need sync compute stream.")
        return False

    def load_cpu_cache_to_reqs(self, reqs: List[InferReq]):
        idle_token_num = g_infer_context.get_can_alloc_token_num()
        all_page_list = []
        is_master_in_dp = self.backend.is_master_in_dp
        for req in reqs:
            page_list = req.shm_req.cpu_cache_match_page_indexes.get_all()
            # 需要返回 prompt logprobs 的请求不应加载 cpu cache：
            # 命中后会复用缓存 kv、跳过推理，拿不到对应 logprobs。
            # match 侧通常已跳过；这里仍要 deref 已 match 的 page，避免引用泄漏。
            if req.sampling_param.shm_param.prompt_logprobs >= 0:
                if is_master_in_dp:
                    req.shm_req.cpu_prompt_cache_len = 0
                all_page_list.extend(page_list)
                continue

            page_len_list = req.shm_req.token_hash_page_len_list.get_all()
            assert len(page_list) <= len(page_len_list)

            matched_page_num = len(page_list)
            original_match_tokens = page_len_list[matched_page_num - 1] if matched_page_num else 0
            # CPU 和 disk 命中统一在恢复前裁剪；图片内部的 KV 可以保留，但不能作为恢复终点。
            while matched_page_num:
                match_tokens = page_len_list[matched_page_num - 1]
                cache_len = clamp_mm_cache_len(match_tokens, req.mm_input_spans)
                if cache_len == match_tokens:
                    break
                matched_page_num = bisect.bisect_right(page_len_list, cache_len, hi=matched_page_num)
            match_tokens = page_len_list[matched_page_num - 1] if matched_page_num else 0

            # 更新命中的 cpu kv cache 长度, 减去radix cache和disk cache的部分.
            if is_master_in_dp:
                req.shm_req.disk_prompt_cache_len = max(
                    0, req.shm_req.disk_prompt_cache_len - (original_match_tokens - match_tokens)
                )
                req.shm_req.cpu_prompt_cache_len = max(
                    0, match_tokens - req.cur_kv_len - req.shm_req.disk_prompt_cache_len
                )

            need_token_num = match_tokens - req.cur_kv_len
            # 多匹配了一定数量的token同时请求长度大于一定的长度，才进行复制操作，不然操作效率不高，代价过高
            if need_token_num >= 128 and req.shm_req.input_len >= 256:
                assert req.cur_kv_len % self.args.page_size == 0
                assert match_tokens % self.args.page_size == 0
                assert need_token_num % self.args.page_size == 0
                assert req.hold_kv_len == req.cur_kv_len
                if need_token_num <= idle_token_num:
                    # 计算需要加载的页面（只加载未匹配的部分）
                    ready_page_num = bisect.bisect_right(page_len_list, req.cur_kv_len)
                    assert ready_page_num <= matched_page_num
                    need_pages = page_list[ready_page_num:matched_page_num]

                    self.read_req_from_cpu(
                        req=req,
                        cpu_page_indices=need_pages,
                        page_lengths=page_len_list[ready_page_num:matched_page_num],
                        start_token_len=page_len_list[ready_page_num - 1] if ready_page_num else 0,
                    )
                    idle_token_num -= need_token_num

                torch.cuda.current_stream().synchronize()

                if self.backend.is_master_in_dp:
                    req.shm_req.shm_cur_kv_len = req.cur_kv_len

            all_page_list.extend(page_list)

        dist.barrier(group=self.init_sync_group)

        if self.backend.is_master_in_dp:
            self.cpu_cache_client.lock.acquire_sleep1ms()
            self.cpu_cache_client.deref_pages(page_list=all_page_list)
            self.cpu_cache_client.lock.release()
        return

    def read_req_from_cpu(
        self,
        req: InferReq,
        cpu_page_indices: List[int],
        page_lengths: List[int],
        start_token_len: int,
    ) -> None:
        """恢复已选定的请求前缀页面，分配资源并登记完整大页引用。

        start_token_len 是首个 CPU 页的逻辑起点，可能早于已有 GPU 命中长度。
        调用方负责选择命中范围、确认容量以及搬运完成后的 CPU 页引用释放。
        """
        if not cpu_page_indices:
            return
        resources = self.backend.model.att_resource_manager
        req_manager = self.backend.model.req_manager
        need_token_num = page_lengths[-1] - req.cur_kv_len

        # 阶段一：分配新增 KV，并补齐首个 CPU 页在 GPU 上已经命中的 token 索引。
        mem_indexes = self.backend._alloc_req_kv_mem(req, need_token_num)
        assert mem_indexes is not None
        if self.need_sync_compute_stream():
            # FA3 的资源搬运需等待已有计算流完成。
            torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        assert 0 <= start_token_len <= req.cur_kv_len, "invalid CPU page start"
        token_indices = torch.cat([
            req_manager.req_to_token_indexs[req.req_idx, start_token_len:req.cur_kv_len],
            mem_indexes.cuda(non_blocking=True),
        ])
        assert token_indices.numel() == page_lengths[-1] - start_token_len

        # 阶段二：只为完整 bounded 页分配普通大页，碎页 LOAD 槽位由资源层管理。
        big_page_slot_ids = None
        if self.backend.is_hybrid_att_model:
            page_tokens = self.args.cpu_cache_token_page_size
            complete_page_lengths = [length for length in page_lengths if length % page_tokens == 0]
            assert len(complete_page_lengths) == len(page_lengths) - int(page_lengths[-1] % page_tokens != 0), (
                "only the final CPU page may be partial"
            )
            allocator = resources.bounded_att_big_page_allocator
            big_page_slot_ids = allocator.alloc(len(complete_page_lengths))
            assert big_page_slot_ids is not None, "not enough bounded state slots for CPU cache load"

        # 阶段三：统一恢复各类 State；异常直接向上传播，不在此处回收或继续执行。
        resources.read_req_from_cpu(
            cpu_page_buffers=self.cpu_cache_client.cpu_kv_cache_tensor,
            cpu_page_indices=cpu_page_indices,
            token_indices=token_indices,
            layouts=resources.get_cpu_transfer_layouts(),
            big_page_slot_ids=big_page_slot_ids,
            req_idx=req.req_idx if self.backend.is_hybrid_att_model else None,
            state_token_len=page_lengths[-1],
        )
        if big_page_slot_ids is not None:
            req.hybrid_len_to_big_page_id.update(zip(complete_page_lengths, big_page_slot_ids))
        req.cur_kv_len = page_lengths[-1]

    def offload_finished_reqs_to_cpu_cache(self, finished_reqs: List[InferReq]) -> List[InferReq]:
        """
        将满足cpu kv cache 卸载条件的请求进行处理, 并返回真的满足退出条件的请求list。
        """
        # 如果开启了cpu cache，将达到finished状态的请求开启将gpu kv cache 卸载到 cpu cache中的操作。
        # 当 kv cache 卸载完成后，才会进行请求的真实退出操作。
        true_finished_reqs = []
        cpu_stream = g_infer_context.get_cpu_kv_cache_stream()
        for req in finished_reqs:
            # 只有 group_req_id 和 request_id 相同的请求才会被卸载到 cpu cache 中。
            # 这个限制是为了兼容 diverse 模式下的请求处理, 只有主请求才 offload kv 到 cpu
            # cache 中
            if req.shm_req.group_req_id != req.shm_req.request_id:
                true_finished_reqs.append(req)
                continue

            # 过滤不适合进行 kv 卸载到 cpu cache 的请求。
            if g_infer_context.is_hybrid_att_model:
                offload_limit_size = self.args.linear_att_hash_page_size
            else:
                offload_limit_size = self.args.cpu_cache_token_page_size

            if req.cur_kv_len < offload_limit_size or req.shm_req.input_len <= offload_limit_size:
                true_finished_reqs.append(req)
                continue

            # 如果请求已经完成了 cpu cache 的任务，则满足了退出条件
            if req.cpu_cache_task_status.is_finished():
                true_finished_reqs.append(req)
                continue

            # 如果请求已经发起过卸载任务且正在卸载过程中，则在当前轮不进行处理
            if req.cpu_cache_task_status.is_running():
                continue

            assert req.cpu_cache_task_status.is_not_started()

            if self.need_sync_compute_stream():
                # TODO fa3 现在必须使用同步模式, 未来需要移除, 必须等待 overlap stream 上的计算任务完成，不然会崩溃
                g_infer_context.get_overlap_stream().synchronize()

            # 发起将请求的 kv cache 卸载到 cpu cache 中的任务
            trans_task = self.write_req_to_cpu(req=req, cpu_kv_cache_stream=cpu_stream)

            # 根据是否成功创建了卸载任务，决定是否将请求加入到处理队列中
            if trans_task is not None:
                self.cpu_cache_handle_queue.append(trans_task)
            else:
                true_finished_reqs.append(req)

        if self.need_sync_compute_stream():
            # TODO fa3 现在必须使用同步模式, 未来需要移除
            cpu_stream.synchronize()

        return true_finished_reqs

    def write_req_to_cpu(
        self, req: InferReq, cpu_kv_cache_stream: torch.cuda.Stream
    ) -> Optional["TransTask"]:
        """选择请求可缓存的前缀，分配 CPU 页并提交写出，返回待确认完成的任务。"""
        assert CacheTier.CPU in req.cache_tiers
        disk_offload_enable = CacheTier.DISK in req.cache_tiers
        producer = torch.cuda.current_stream()
        # 阶段一：等待当前生产流已提交的数据写入，不阻塞主机线程。
        cpu_kv_cache_stream.wait_stream(producer)
        # TODO：将 bounded CPU→CPU 快照搬运改为当前流上的 kernel。
        # 现有主机 copy 不受 wait_stream 约束，kernel 落地前源快照仍须预先就绪。
        with torch.cuda.stream(cpu_kv_cache_stream):
            # 综合考虑后只对prompt做缓存管理，不包含decode内容，这里与radix cache不一致
            token_hash_list = req.shm_req.token_hash_list.get_all()
            page_len_list = req.shm_req.token_hash_page_len_list.get_all()
            assert len(token_hash_list) == len(page_len_list)

            # 阶段二：确定本次可写出范围，申请 CPU 页。
            if self.backend.is_master_in_dp:

                find_index = bisect.bisect_right(page_len_list, req.cur_kv_len)
                move_block_size = find_index

                # hybrid 模型的最后一个页面可能是碎页，需判断该碎页是否满足卸载条件。
                move_block_size = self._handle_hybrid_att_last_page(
                    req=req, move_block_size=move_block_size, page_len_list=page_len_list
                )

                if move_block_size == 0:
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None

                try:
                    self.cpu_cache_client.lock.acquire_sleep1ms()
                    page_list, ready_list = self.cpu_cache_client.allocate_pages(
                        token_hash_list[:move_block_size],
                        disk_offload_enable=disk_offload_enable,
                    )
                finally:
                    self.cpu_cache_client.lock.release()

                # 分配不足时返回 -1 并补齐剩余位置，只保留成功分配的连续前缀。
                # 第一个 -1 及后续页面（包括可能的尾部碎页）均不参与本次写出。
                if -1 in page_list:
                    allocated_page_count = page_list.index(-1)
                    page_list = page_list[:allocated_page_count]
                    ready_list = ready_list[:allocated_page_count]
                item_size = len(page_list)
                if item_size == 0:
                    dist.broadcast_object_list([0], group=self.gloo_group, group_src=0)
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None

                broadcast_data = {"item_size": item_size, "page_list": page_list, "ready_list": ready_list}
                dist.broadcast_object_list([broadcast_data], group=self.gloo_group, group_src=0)
            else:
                recv_list = [None]
                dist.broadcast_object_list(recv_list, group=self.gloo_group, group_src=0)
                if isinstance(recv_list[0], int) and recv_list[0] == 0:
                    req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
                    return None
                broadcast_data = recv_list[0]
                item_size = broadcast_data["item_size"]
                page_list = broadcast_data["page_list"]
                ready_list = broadcast_data["ready_list"]

            page_indexes = torch.tensor(page_list, dtype=torch.int32)
            page_readies = torch.tensor(ready_list, dtype=torch.bool)

            move_token_num = page_len_list[item_size - 1]
            assert req.cur_kv_len >= move_token_num
            token_indexes = self.backend.model.req_manager.req_to_token_indexs[req.req_idx, 0:move_token_num]

            # 阶段三：只查找实际分配的 CPU 页所需快照；完整大页必须存在，不做缺失回退。
            resources = self.backend.model.att_resource_manager
            big_page_slot_ids, small_page_slot_ids = None, None
            if self.backend.is_hybrid_att_model:
                page_tokens = self.args.cpu_cache_token_page_size
                radix_cache = g_infer_context.radix_cache
                shared_slots = radix_cache.get_big_page_ids_by_node(req.shared_kv_node) if radix_cache is not None else []
                big_slots = {(i + 1) * page_tokens: slot for i, slot in enumerate(shared_slots)}
                big_slots.update(req.hybrid_len_to_big_page_id)
                big_page_slot_ids = [
                    big_slots[length] for length in page_len_list[:item_size] if length % page_tokens == 0
                ]
                small_page_slot_ids = []
                # 只有实际分配的前缀包含原始尾页，才可能使用小页快照。
                # 若分配遇到 -1，item_size 已截短，尾页不会进入此分支。
                includes_tail_page = item_size == len(page_len_list)
                if includes_tail_page and page_len_list[-1] % page_tokens != 0:
                    small_page_slot_ids.append(req.tail_small_page_buffer_id)
            # 整批提交并保留 token/page 对齐，由资源层跳过 ready 页。
            resources.write_req_to_cpu(
                token_indices=token_indexes,
                cpu_page_buffers=self.cpu_cache_client.cpu_kv_cache_tensor,
                cpu_page_indices=page_list,
                layouts=resources.get_cpu_transfer_layouts(),
                cpu_page_readies=ready_list,
                big_page_slot_ids=big_page_slot_ids,
                small_page_slot_ids=small_page_slot_ids,
            )

            # 这个操作只是为了在offload 对应的cuda stream中，同步标记下对应的kv cache offload 操作已经完成，
            if self.backend.dp_world_size > 1:
                dist.all_reduce(self.offload_sync_tensor, op=dist.ReduceOp.MAX, group=self.offload_sync_group)

            sync_event = torch.cuda.Event()
            sync_event.record()
            req.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.RUNNING
            trans_task = TransTask(
                move_token_num=move_token_num,
                page_indexes=page_indexes,
                page_readies=page_readies,
                req_obj=req,
                sync_event=sync_event,
            )

        return trans_task

    def _handle_hybrid_att_last_page(self, req: InferReq, move_block_size: int, page_len_list: List[int]) -> int:
        if not g_infer_context.is_hybrid_att_model:
            return move_block_size

        if move_block_size == 0:
            return 0

        if move_block_size == len(page_len_list):
            tail_len = page_len_list[move_block_size - 1]
            if tail_len % self.args.cpu_cache_token_page_size != 0:
                # 全局关闭了碎页的cpu cache 存储功能。
                if self.args.disable_linear_att_small_page_cpu_cache:
                    return move_block_size - 1
                # 说明是碎页，碎页需要判定是否满足cpu cache 的offload条件。
                if req.tail_small_page_buffer_id is None:
                    return move_block_size - 1
        return move_block_size

    def update_cpu_cache_task_states(self):
        if self.backend.is_master_in_dp:
            trans_ok_tasks = []
            while len(self.cpu_cache_handle_queue) != 0:
                task: TransTask = self.cpu_cache_handle_queue.popleft()
                if task.sync_event.query():
                    trans_ok_tasks.append(task)
                else:
                    self.cpu_cache_handle_queue.appendleft(task)
                    break
            item_size = len(trans_ok_tasks)
            dist.broadcast_object_list([item_size], group=self.filter_group, group_src=0)
        else:
            recv_list = [None]
            dist.broadcast_object_list(recv_list, group=self.filter_group, group_src=0)
            item_size = recv_list[0]
            trans_ok_tasks: List[TransTask] = [self.cpu_cache_handle_queue.popleft() for _ in range(item_size)]

        if item_size > 0:
            page_array_list = [task.page_indexes.tolist() for task in trans_ok_tasks]
            move_token_nums = [task.move_token_num for task in trans_ok_tasks]
            if self.backend.is_master_in_dp:
                self.cpu_cache_client.lock.acquire_sleep1ms()
                # 分组update，避免不同请求的page交叉，导致disk cache hash不一致
                for task, pages, move_token_num in zip(trans_ok_tasks, page_array_list, move_token_nums):
                    self.cpu_cache_client.update_pages_status_to_ready(
                        page_list=pages,
                        deref=True,
                        disk_offload_enable=CacheTier.DISK in task.req_obj.cache_tiers,
                        token_num_in_page_list=move_token_num,
                    )
                self.cpu_cache_client.lock.release()
            for task in trans_ok_tasks:
                task.req_obj.cpu_cache_task_status = InferReq._CpuCacheTaskStatus.FINISHED
        return


@dataclasses.dataclass
class TransTask:
    move_token_num: int
    page_indexes: torch.Tensor
    page_readies: torch.Tensor
    req_obj: InferReq
    sync_event: torch.cuda.Event
