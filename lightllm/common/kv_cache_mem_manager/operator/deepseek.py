import dataclasses
from typing import List, Optional

import torch
from .normal import NormalMemOperator
from .base import BaseMemManagerOperator
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Deepseek2MemOperator(NormalMemOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager

        mem_manager: Deepseek2MemoryManager = self.mem_manager

        from ...basemodel.triton_kernel.kv_copy.mla_copy_kv import destindex_copy_kv

        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv_lora_rank + rope_dim == mem_manager.kv_buffer.shape[-1]

        destindex_copy_kv(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            mem_manager.kv_buffer[layer_index][:, :, :kv_lora_rank],
            mem_manager.kv_buffer[layer_index][:, :, kv_lora_rank:],
        )
        return


class Deepseek3_2MemOperator(Deepseek2MemOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.deepseek3_2mem_manager import Deepseek3_2MemoryManager

        mem_manager: Deepseek3_2MemoryManager = self.mem_manager
        from ...basemodel.triton_kernel.kv_copy.mla_copy_kv import destindex_copy_kv

        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv_lora_rank + rope_dim == mem_manager.kv_buffer.shape[-1] - (144 // 2)

        destindex_copy_kv(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            mem_manager.kv_buffer[layer_index][:, :, :kv_lora_rank],
            mem_manager.kv_buffer[layer_index][:, :, kv_lora_rank : (kv_lora_rank + rope_dim)],
        )
        return


class FP8PerTokenGroupQuantDeepseek3_2MemOperator(BaseMemManagerOperator):
    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        from lightllm.common.kv_cache_mem_manager.fp8_per_token_group_quant_deepseek3_2mem_manager import (
            FP8PerTokenGroupQuantDeepseek3_2MemoryManager,
        )

        mem_manager: FP8PerTokenGroupQuantDeepseek3_2MemoryManager = self.mem_manager
        from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_kv_flashmla_fp8 import (
            destindex_copy_kv_flashmla_fp8,
        )

        rope_dim = 64
        kv_lora_rank = kv.shape[2] - rope_dim
        assert kv_lora_rank == 512, f"Expected kv_lora_rank=512, got {kv_lora_rank}"

        flashmla_bytes_per_token = mem_manager.flashmla_bytes_per_token

        o_nope = mem_manager.kv_buffer[layer_index][:, :, :512].view(torch.float8_e4m3fn)
        o_scale = mem_manager.kv_buffer[layer_index][:, :, 512:528].view(torch.float32)
        o_rope = mem_manager.kv_buffer[layer_index][:, :, 528:flashmla_bytes_per_token].view(torch.bfloat16)
        destindex_copy_kv_flashmla_fp8(
            kv[:, :, :kv_lora_rank],
            kv[:, :, kv_lora_rank:],
            mem_index,
            o_nope,
            o_scale,
            o_rope,
        )
        return


class DeepseekV4MemOperator(BaseMemManagerOperator):
    def __init__(self, mem_manager):
        super().__init__(mem_manager)
        self.cpu_cache_staging_slots = [Dsv4StagingSlot(), Dsv4StagingSlot()]

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        """Copy packed history pages; continuation is restored by the request manager."""
        manager = self.mem_manager
        src, dst = src_mem_index, dst_mem_index
        page_size = manager.page_size
        assert src.numel() == dst.numel() and src.numel() % page_size == 0
        src_pages = (src.reshape(-1, page_size)[:, 0] // page_size).to(device="cuda", dtype=torch.int64)
        dst_pages = (dst.reshape(-1, page_size)[:, 0] // page_size).to(device="cuda", dtype=torch.int64)
        for pool in (manager.c4_pool, manager.c4_indexer_pool, manager.c128_pool):
            if pool is not None:
                pool.buffer.index_copy_(1, dst_pages, pool.buffer.index_select(1, src_pages))

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        raise NotImplementedError("DeepSeek-V4 writes packed KV using request-owned SWA slots")

    def pack_cpu_cache_pages(
        self, source_mem_indexes: torch.Tensor, source_req_meta: torch.Tensor, staging: torch.Tensor
    ) -> None:
        """Pack complete DS4 checkpoints into caller-owned CUDA staging."""
        from lightllm.models.deepseek_v4.triton_kernel.cpu_cache_io import pack_gpu_cache_to_staging

        pack_gpu_cache_to_staging(self.mem_manager, source_mem_indexes, source_req_meta, staging)
        return

    def scatter_packed_cpu_cache_pages(
        self,
        staging: torch.Tensor,
        page_indexes: torch.Tensor,
        cpu_cache_client,
    ) -> None:
        """Write packed checkpoints to their pinned shared-memory pages."""
        from lightllm.models.deepseek_v4.triton_kernel.cpu_cache_io import scatter_staging_to_cpu_pages

        scatter_staging_to_cpu_pages(staging, cpu_cache_client.cpu_kv_cache_tensor, page_indexes)
        return

    def load_cpu_cache_pages(
        self,
        plan,
        page_indexes: torch.Tensor,
        cpu_cache_client,
        first_page_history_offset_tokens: int = 0,
    ) -> None:
        """Selectively restore compressed history and the final resume window."""
        from lightllm.models.deepseek_v4.triton_kernel.cpu_cache_io import unpack_cpu_cache_to_gpu

        unpack_cpu_cache_to_gpu(
            self.mem_manager,
            plan,
            cpu_cache_client.cpu_kv_cache_tensor,
            page_indexes,
            first_page_history_offset_tokens,
        )
        return

    def store_cpu_cache_pages(
        self,
        staging_slot: int,
        source_mem_indexes: List[torch.Tensor],
        source_req_meta: List[List[int]],
        page_indexes: List[int],
        cpu_cache_client,
        producer_stream: torch.cuda.Stream,
        cpu_stream: torch.cuda.Stream,
    ):
        """Pack into owned staging, then asynchronously scatter to CPU pages."""
        slot = self.cpu_cache_staging_slots[staging_slot]
        assert not slot.in_use
        page_num = len(page_indexes)
        layout = self.mem_manager.cpu_cache_layout
        if slot.buffer is None or slot.buffer.shape[0] < page_num:
            with torch.cuda.stream(producer_stream):
                slot.buffer = torch.empty((page_num, layout.page_nbytes), dtype=torch.uint8, device="cuda")
                slot.source_mem_indexes = torch.empty(
                    (page_num, layout.token_page_size), dtype=torch.int32, device="cuda"
                )
                slot.page_indexes_cuda = torch.empty((page_num,), dtype=torch.int32, device="cuda")
            slot.page_indexes_cpu = torch.empty((page_num,), dtype=torch.int32, device="cpu", pin_memory=True)
        slot.in_use = True
        slot.page_indexes_cpu[:page_num].numpy()[:] = page_indexes
        with torch.cuda.stream(producer_stream):
            indexes = slot.source_mem_indexes[:page_num]
            torch.stack(source_mem_indexes, out=indexes)
            staging = slot.buffer[:page_num]
            req_meta = torch.tensor(source_req_meta, dtype=torch.int32, device="cuda")
            self.pack_cpu_cache_pages(indexes, req_meta, staging)
            pack_event = torch.cuda.Event()
            pack_event.record()

        # Request teardown and the next prefill may recycle the original slabs.
        torch.cuda.current_stream().wait_event(pack_event)
        with torch.cuda.stream(cpu_stream):
            cpu_stream.wait_event(pack_event)
            indexes_cuda = slot.page_indexes_cuda[:page_num]
            indexes_cuda.copy_(slot.page_indexes_cpu[:page_num], non_blocking=True)
            self.scatter_packed_cpu_cache_pages(staging, indexes_cuda, cpu_cache_client)
            store_event = torch.cuda.Event()
            store_event.record()
        return pack_event, store_event

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, cpu_cache_client, req):
        """Restore reserved history slots and request-owned continuation state."""
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        manager = self.mem_manager
        req_manager = g_infer_context.req_manager
        loaded_start = int(req.cur_kv_len)
        loaded_end = loaded_start + mem_indexes.numel()
        req_manager.prepare_swa(req.req_idx, loaded_end - 256, loaded_end)
        resume_slots = req_manager.get_swa_slots(
            req.req_idx, torch.arange(loaded_end - 256, loaded_end, device=mem_indexes.device)
        )
        try:
            plan = manager.prepare_cpu_cache_load(
                token_num=mem_indexes.numel(),
                loaded_end=loaded_end,
                resume_swa_slots=resume_slots,
                mem_indexes=mem_indexes,
            )
            self.load_cpu_cache_pages(
                plan=plan,
                page_indexes=page_indexes,
                cpu_cache_client=cpu_cache_client,
                first_page_history_offset_tokens=loaded_start % manager.cpu_cache_layout.token_page_size,
            )
        except Exception:
            req_manager.clear_runtime_state(req.req_idx)
            raise

        if g_infer_context.radix_cache is not None:
            req_manager.restore_cpu_cache_checkpoints(
                req, loaded_start, loaded_end, cpu_cache_client, g_infer_context.radix_cache
            )


@dataclasses.dataclass
class Dsv4StagingSlot:
    """Operator-owned buffers leased until the cache module completes a store."""

    buffer: Optional[torch.Tensor] = None
    source_mem_indexes: Optional[torch.Tensor] = None
    page_indexes_cpu: Optional[torch.Tensor] = None
    page_indexes_cuda: Optional[torch.Tensor] = None
    in_use: bool = False
