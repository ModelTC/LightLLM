from typing import TYPE_CHECKING

import torch
from .normal import NormalMemOperator
from .base import BaseMemManagerOperator
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args

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
    def pack_cpu_cache_pages(self, mem_indexes, big_page_buffer_ids, staging, small_page_state_buffer):
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_kv_buffer_to_cpu_cache

        copy_kv_buffer_to_cpu_cache(
            mem_manager=self.mem_manager,
            mem_indexes=mem_indexes,
            page_indexes=torch.arange(len(staging), dtype=torch.int32, device=staging.device),
            page_readies=None,
            big_page_buffer_ids=big_page_buffer_ids,
            cpu_cache_tensor=staging,
            small_page_state_buffer=small_page_state_buffer,
        )

    def scatter_packed_cpu_cache_pages(self, staging, page_indexes, cpu_cache_client):
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import scatter_cpu_cache_pages

        scatter_cpu_cache_pages(staging, page_indexes, cpu_cache_client.cpu_kv_cache_tensor)

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        """仅复制压缩 KV 和 indexer K；SWA、compressor 状态由 ReqManager 恢复。"""
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        src = src_mem_index.to(device="cuda", dtype=torch.int64, non_blocking=True)
        dst = dst_mem_index.to(device="cuda", dtype=torch.int64, non_blocking=True)
        for buffer in self.mem_manager.kv_buffer:
            copy_kv_buffer_to_kv_buffer(src, dst, buffer.view(1, buffer.shape[0], -1))

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        # 模型推理层通过融合算子写入 SWA，compressor/indexer 直接写入压缩 KV。
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
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_kv_buffer_to_cpu_cache, copy_state_cache

        # MLA KV and compression states are replicated across TP ranks.
        if get_current_rank_in_dp() != 0:
            return

        mem_manager: "DeepseekV4MemoryManager" = self.mem_manager
        req_manager: "DeepseekV4ReqManager" = g_infer_context.req_manager
        page_token_num = get_env_start_args().cpu_cache_token_page_size

        big_page_buffer_ids_cpu = g_infer_context.radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
        max_kv_len = (len(mem_indexes) // page_token_num) * page_token_num
        start_kv_len = (len(big_page_buffer_ids_cpu) + 1) * page_token_num
        for seq_len in range(start_kv_len, max_kv_len + 1, page_token_num):
            big_page_buffer_ids_cpu.append(req.hybrid_len_to_big_page_id[seq_len])

        # 碎页：将尾部小页状态复制到临时大页，KV 由拷贝算子按实际 token 数处理。
        has_tail_page = len(mem_indexes) % page_token_num != 0
        if has_tail_page:
            assert req.tail_small_page_buffer_id is not None
            temp_page_id = mem_manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID
            copy_state_cache(
                src_state=req_manager.small_page_buffers.buffer[req.tail_small_page_buffer_id],
                dst_state=mem_manager.big_page_buffers.buffer[temp_page_id],
            )
            big_page_buffer_ids_cpu.append(temp_page_id)

        big_page_buffer_ids_gpu = torch.tensor(big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu").cuda(
            non_blocking=True
        )
        copy_kv_buffer_to_cpu_cache(
            mem_manager=mem_manager,
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            page_readies=page_readies,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
        )

    def load_cpu_cache_to_gpu(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
        req: "InferReq",
    ) -> list:
        from lightllm.server.router.model_infer.infer_batch import g_infer_context
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_cpu_cache_to_kv_buffer, copy_state_cache

        mem_manager: "DeepseekV4MemoryManager" = self.mem_manager
        req_manager: "DeepseekV4ReqManager" = g_infer_context.req_manager
        page_token_num = get_env_start_args().cpu_cache_token_page_size

        big_page_num = len(mem_indexes) // page_token_num
        max_kv_len = (req.cur_kv_len // page_token_num) * page_token_num
        big_page_buffer_ids_cpu = []
        for _ in range(big_page_num):
            page_id = mem_manager.big_page_buffers.alloc_one_state_cache()
            assert page_id is not None
            req.hybrid_len_to_big_page_id[max_kv_len] = page_id
            big_page_buffer_ids_cpu.append(page_id)
            max_kv_len -= page_token_num
        big_page_buffer_ids_cpu.reverse()
        checkpoint_leases = [(mem_manager.big_page_buffers, list(big_page_buffer_ids_cpu))]

        # 碎页：先将状态读入临时大页，再恢复请求运行态。
        has_tail_page = len(mem_indexes) % page_token_num != 0
        if has_tail_page:
            big_page_buffer_ids_cpu.append(mem_manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID)

        big_page_buffer_ids_gpu = torch.tensor(big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu").cuda(
            non_blocking=True
        )
        copy_cpu_cache_to_kv_buffer(
            mem_manager=mem_manager,
            mem_indexes=mem_indexes,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            page_indexes=page_indexes,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
        )

        if has_tail_page and req.cur_kv_len == req.hybrid_cache_len:
            g_infer_context.radix_cache.free_one_small_page_buffer()
            small_page_buffer_id = req_manager.small_page_buffers.alloc_one_state_cache()
            if small_page_buffer_id is not None:
                req.tail_small_page_buffer_id = small_page_buffer_id
                checkpoint_leases.append((req_manager.small_page_buffers, [small_page_buffer_id]))
                # 临时大页会被复用，先将碎页状态保留到小页池。
                copy_state_cache(
                    src_state=mem_manager.big_page_buffers.buffer[big_page_buffer_ids_cpu[-1]],
                    dst_state=req_manager.small_page_buffers.buffer[small_page_buffer_id],
                )

        req_manager.restore_big_page_state(big_page_buffer_idx=big_page_buffer_ids_cpu[-1], req=req)
        return checkpoint_leases
