from typing import TYPE_CHECKING, List

import torch
from .normal import NormalMemOperator
from .base import BaseMemManagerOperator
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from ..deepseek4_mem_manager import DeepseekV4MemoryManager
    from lightllm.common.req_manager.deepseek4 import DeepseekV4ReqManager
    from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
    from lightllm.server.router.model_infer.infer_batch import InferReq

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
    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        """Copy exactly the requested token slots; request state has a separate owner."""
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        # Copy history only.
        # The request manager restores SWA and compressor state from checkpoints.
        src = src_mem_index.to(device="cuda", dtype=torch.int64, non_blocking=True)
        dst = dst_mem_index.to(device="cuda", dtype=torch.int64, non_blocking=True)
        for buffer in self.mem_manager.kv_buffer:
            copy_kv_buffer_to_kv_buffer(src, dst, buffer.view(1, buffer.shape[0], -1))

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        # _get_qkv() calls req_manager.pack_mla_kv_to_cache_fused_norm_rope().
        # One kernel runs RMSNorm, RoPE, FP8 packing, and SWA writes.
        # This avoids extra tensors and kernel launches.
        # Prefill and decode skip _post_cache_kv(). They do not call this method.
        # The compressor and indexer write history to mem_manager.kv_buffer.
        raise NotImplementedError("DeepSeek-V4 writes packed KV using request-owned SWA slots")

    def offload_gpu_kv_to_cpu_cache(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        page_readies: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
        req: "InferReq",
    ) -> None:
        from lightllm.utils.dist_utils import get_current_rank_in_dp
        from lightllm.server.router.model_infer.infer_batch import g_infer_context
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_history_pages, copy_checkpoint_pages
        from lightllm.utils.custom_kernel_utis import tensor_copy_3dim

        # All ranks retain their checkpoints until the common completion event.
        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        # MLA KV and compression states are replicated across TP ranks.
        if get_current_rank_in_dp() != 0:
            return

        mem_manager: "DeepseekV4MemoryManager" = self.mem_manager
        req_manager: "DeepseekV4ReqManager" = g_infer_context.req_manager
        page_token_num = mem_manager.cache_config.cpu_cache_token_page_size

        shared_big_page_buffer_ids = g_infer_context.radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
        big_page_buffer_ids_cpu: List[int] = []
        for page_index in range(page_indexes.numel()):
            seq_len = (page_index + 1) * page_token_num
            if seq_len > mem_indexes.numel():
                big_page_buffer_id = mem_manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID
                assert req.tail_small_page_buffer_id is not None
                src_states = req_manager.small_page_buffers.get_state_cache(buffer_idx=req.tail_small_page_buffer_id)
                dst_states = mem_manager.big_page_buffers.get_state_cache(buffer_idx=big_page_buffer_id)
                # Keep pinned checkpoint copies ordered on the offload stream.
                for dst_state, src_state in zip(dst_states, src_states):
                    if src_state.numel():
                        tensor_copy_3dim(dst_state, src_state)
            elif page_index < len(shared_big_page_buffer_ids):
                big_page_buffer_id = shared_big_page_buffer_ids[page_index]
            else:
                big_page_buffer_id = req.hybrid_len_to_big_page_id[seq_len]
            big_page_buffer_ids_cpu.append(big_page_buffer_id)

        big_page_buffer_ids_gpu = torch.tensor(big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu").cuda(
            non_blocking=True
        )
        cpu_cache_tensor = cpu_cache_client.cpu_kv_cache_tensor.view(cpu_cache_client.cpu_kv_cache_tensor.shape[0], -1)
        copy_history_pages(
            manager=mem_manager,
            indexes=mem_indexes,
            page_ids=page_indexes,
            pages=cpu_cache_tensor,
            to_pages=True,
            readies=page_readies,
        )
        copy_checkpoint_pages(
            manager=mem_manager,
            checkpoint_ids=big_page_buffer_ids_gpu,
            page_ids=page_indexes,
            pages=cpu_cache_tensor,
            to_pages=True,
            readies=page_readies,
        )

    def load_cpu_cache_to_gpu(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
        req: "InferReq",
    ) -> None:
        from lightllm.server.router.model_infer.infer_batch import g_infer_context
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_history_pages, copy_checkpoint_pages
        from lightllm.utils.custom_kernel_utis import tensor_copy_3dim

        mem_manager: "DeepseekV4MemoryManager" = self.mem_manager
        req_manager: "DeepseekV4ReqManager" = g_infer_context.req_manager
        page_token_num = mem_manager.cache_config.cpu_cache_token_page_size

        cpu_cache_tensor = cpu_cache_client.cpu_kv_cache_tensor.view(cpu_cache_client.cpu_kv_cache_tensor.shape[0], -1)
        copy_history_pages(
            manager=mem_manager,
            indexes=mem_indexes,
            page_ids=page_indexes,
            pages=cpu_cache_tensor,
            to_pages=False,
        )

        big_page_num = mem_indexes.numel() // page_token_num
        first_seq_len = req.cur_kv_len // page_token_num * page_token_num - (big_page_num - 1) * page_token_num
        big_page_buffer_ids_cpu = mem_manager.big_page_buffers.alloc_state_cache(big_page_num)
        if big_page_buffer_ids_cpu is None:
            raise RuntimeError("No hybrid checkpoint slots available for CPU cache load")
        for page_index, big_page_buffer_id in enumerate(big_page_buffer_ids_cpu):
            seq_len = first_seq_len + page_index * page_token_num
            req.hybrid_len_to_big_page_id[seq_len] = big_page_buffer_id

        has_tail_page = mem_indexes.numel() % page_token_num != 0
        if has_tail_page:
            big_page_buffer_ids_cpu.append(mem_manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID)

        big_page_buffer_ids_gpu = torch.tensor(big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu").cuda(
            non_blocking=True
        )
        copy_checkpoint_pages(
            manager=mem_manager,
            checkpoint_ids=big_page_buffer_ids_gpu,
            page_ids=page_indexes,
            pages=cpu_cache_tensor,
            to_pages=False,
        )

        if has_tail_page and req.cur_kv_len == req.hybrid_cache_len:
            g_infer_context.radix_cache.free_one_small_page_buffer()
            small_page_buffer_id = req_manager.small_page_buffers.alloc_one_state_cache()
            if small_page_buffer_id is not None:
                req.tail_small_page_buffer_id = small_page_buffer_id
                src_states = mem_manager.big_page_buffers.get_state_cache(buffer_idx=big_page_buffer_ids_cpu[-1])
                dst_states = req_manager.small_page_buffers.get_state_cache(buffer_idx=small_page_buffer_id)
                # Retain the tail before the shared load slot is reused.
                for dst_state, src_state in zip(dst_states, src_states):
                    if src_state.numel():
                        tensor_copy_3dim(dst_state, src_state)

        req_manager.restore_big_page_state(big_page_buffer_idx=big_page_buffer_ids_cpu[-1], req=req)
