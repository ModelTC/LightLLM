import torch
import triton
from .base import BaseMemManagerOperator
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_rank_in_dp
from lightllm.common.state_cache_manager import get_hybrid_cache_config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
    from lightllm.server.router.model_infer.infer_batch import InferReq


class HybridAttMemOperator(BaseMemManagerOperator):
    """Shared CPU transfer lifecycle for token KV with checkpoint state."""

    def __init__(self, mem_manager):
        super().__init__(mem_manager)
        self.cpu_cache_config = get_hybrid_cache_config(linear_config=getattr(mem_manager, "linear_config", None))

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        copy_kv_buffer_to_kv_buffer(
            src_mem_index.cuda(non_blocking=True), dst_mem_index.cuda(non_blocking=True), self.mem_manager.kv_buffer
        )
        return

    def load_cpu_cache_to_gpu(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
        req: "InferReq",
    ):
        assert mem_indexes.is_cuda and page_indexes.is_cuda
        args = get_env_start_args()
        assert triton.cdiv(len(mem_indexes), args.cpu_cache_token_page_size) == len(page_indexes)
        assert len(mem_indexes) % args.linear_att_hash_page_size == 0
        assert args.cpu_cache_token_page_size == args.linear_att_hash_page_size * args.linear_att_page_block_num
        mem_manager = self.mem_manager

        big_page_num = len(mem_indexes) // args.cpu_cache_token_page_size
        max_kv_len = (req.cur_kv_len // args.cpu_cache_token_page_size) * args.cpu_cache_token_page_size
        assert max_kv_len % args.cpu_cache_token_page_size == 0

        big_page_buffer_ids_cpu = []
        for i in range(big_page_num):
            page_id = mem_manager.big_page_buffers.alloc_one_state_cache()
            assert page_id is not None
            req.hybrid_len_to_big_page_id[max_kv_len] = page_id
            big_page_buffer_ids_cpu.append(page_id)
            max_kv_len -= args.cpu_cache_token_page_size
            assert max_kv_len % args.cpu_cache_token_page_size == 0

        big_page_buffer_ids_cpu.reverse()

        # 碎页情况的处理
        has_tail_page = len(mem_indexes) % args.cpu_cache_token_page_size != 0
        if has_tail_page:
            padded_token_num = triton.cdiv(
                len(mem_indexes), args.cpu_cache_token_page_size
            ) * args.cpu_cache_token_page_size - len(mem_indexes)
            mem_indexes = torch.nn.functional.pad(mem_indexes, (0, padded_token_num), mode="constant", value=-1)

            # 将对应的小叶数据拷贝到临时的大页上，再从大页上拷贝到对应的运行态页面上
            big_page_buffer_ids_cpu.append(mem_manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID)

        big_page_buffer_ids_gpu = torch.tensor(big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu").cuda(
            non_blocking=True
        )

        assert len(big_page_buffer_ids_gpu) == len(page_indexes)

        self._load_target_pages(mem_indexes, big_page_buffer_ids_gpu, page_indexes, cpu_cache_client)
        self._copy_window_pages(big_page_buffer_ids_gpu, page_indexes, cpu_cache_client)

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        g_infer_context.req_manager.restore_big_page_state(
            big_page_buffer_idx=big_page_buffer_ids_cpu[-1],
            req=req,
        )

        return

    def offload_gpu_kv_to_cpu_cache(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        page_readies: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
        req: "InferReq",
    ):
        args = get_env_start_args()
        if not hasattr(self, "big_page_ids_buffer_store"):
            self.big_page_ids_buffer_store = torch.empty((1024 * 1024 * 4,), dtype=torch.int64, device="cuda")
            # 多申请3个cpu cache token page size，用于处理碎页情况，碎页情况需要将对应的大页数据拷贝到临时的大页上，
            # 再从大页上拷贝到对应的运行态页面上
            self.mem_indexes_buffer = torch.empty(
                (args.max_req_total_len + 3 * args.cpu_cache_token_page_size,), dtype=torch.int32, device="cuda"
            )

        assert mem_indexes.is_cuda and page_indexes.is_cuda and page_readies.is_cuda

        assert len(mem_indexes) % args.linear_att_hash_page_size == 0
        assert triton.cdiv(len(mem_indexes), args.cpu_cache_token_page_size) == len(page_indexes)

        mem_manager = self.mem_manager

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        big_page_buffer_ids_cpu = g_infer_context.radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
        max_kv_len = (len(mem_indexes) // args.cpu_cache_token_page_size) * args.cpu_cache_token_page_size
        start_kv_len = (len(big_page_buffer_ids_cpu) + 1) * args.cpu_cache_token_page_size
        for seq_len in range(start_kv_len, max_kv_len + 1, args.cpu_cache_token_page_size):
            page_id = req.hybrid_len_to_big_page_id[seq_len]
            big_page_buffer_ids_cpu.append(page_id)

        if len(mem_indexes) % args.cpu_cache_token_page_size != 0:
            # 存在不满大页的碎页的页面存在需要复制的情况
            dst_len = triton.cdiv(len(mem_indexes), args.cpu_cache_token_page_size) * args.cpu_cache_token_page_size
            assert dst_len <= self.mem_indexes_buffer.shape[0]
            dst_mem_indexes = self.mem_indexes_buffer[0:dst_len].fill_(-1)
            dst_mem_indexes[0 : len(mem_indexes)].copy_(mem_indexes, non_blocking=True)
            mem_indexes = dst_mem_indexes
            assert req.tail_small_page_buffer_id is not None
            self._copy_tail_state(
                g_infer_context.radix_cache.small_page_buffers,
                req.tail_small_page_buffer_id,
                mem_manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID,
            )
            big_page_buffer_ids_cpu.append(mem_manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID)

        assert len(big_page_buffer_ids_cpu) == len(page_indexes) == len(page_readies)

        big_page_buffer_ids_cpu = torch.tensor(
            big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu", pin_memory=True
        )
        assert len(big_page_buffer_ids_cpu) <= self.big_page_ids_buffer_store.shape[0]
        big_page_buffer_ids_gpu = self.big_page_ids_buffer_store[0 : len(big_page_buffer_ids_cpu)]
        big_page_buffer_ids_gpu.copy_(big_page_buffer_ids_cpu, non_blocking=True)

        self._offload_target_pages(mem_indexes, big_page_buffer_ids_gpu, page_indexes, page_readies, cpu_cache_client)
        self._copy_window_pages(big_page_buffer_ids_gpu, page_indexes, cpu_cache_client, page_readies)
        return

    def _load_target_pages(self, mem_indexes, buffer_ids, page_indexes, cpu_cache_client):
        raise NotImplementedError()

    def _offload_target_pages(self, mem_indexes, buffer_ids, page_indexes, page_readies, cpu_cache_client):
        raise NotImplementedError()

    def _copy_target_state(self, source, source_id, destination_id):
        # A full-attention target has no additional checkpoint payload.
        return

    def _copy_tail_state(self, source, source_id, destination_id):
        self._copy_target_state(source, source_id, destination_id)
        window = getattr(self.mem_manager.big_page_buffers, "draft_window", None)
        if window is not None:
            from lightllm.common.basemodel.triton_kernel.hybrid_cpu_cache_copy import copy_checkpoint_state

            copy_checkpoint_state(
                source.draft_window.get_state_cache(source_id), window.get_state_cache(destination_id)
            )

    def _copy_window_pages(self, buffer_ids, page_indexes, cpu_cache_client, page_readies=None):
        window = getattr(self.mem_manager.big_page_buffers, "draft_window", None)
        if window is None:
            return
        from lightllm.common.basemodel.triton_kernel.hybrid_cpu_cache_copy import copy_checkpoint_pages

        buffers = (window.kv, window.ends, window.counts)
        pages = self.cpu_cache_config.get_window_views(cpu_cache_client.cpu_kv_cache_tensor, get_current_rank_in_dp())
        if page_readies is None:
            copy_checkpoint_pages(pages, buffers, page_indexes, buffer_ids)
        else:
            copy_checkpoint_pages(buffers, pages, buffer_ids, page_indexes, page_readies)
