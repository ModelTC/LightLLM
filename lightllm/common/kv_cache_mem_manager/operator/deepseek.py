from typing import TYPE_CHECKING, List

import torch
from .normal import NormalMemOperator
from .base import BaseMemManagerOperator
from lightllm.common.state_cache_manager.deepseek4 import DSV4_CACHE_BLOCK_SIZE
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from ..mem_manager import MemoryManager

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
    def copy_kv_from_other_dp_ranks(
        self, mem_managers, move_token_indexes, token_dp_indexes, mem_indexes, dp_size_in_node, rank_in_dp
    ):
        """The standard DP copy interface; request checkpoints use the hybrid path."""
        from lightllm.utils.device_utils import kv_trans_use_p2p
        from lightllm.models.deepseek_v4.triton_kernel.dp_cache_io import copy_dsv4_dp_history

        if not hasattr(self, "dp_history_ptrs"):
            if not kv_trans_use_p2p():
                raise RuntimeError("DeepSeek-V4 DP prompt-cache fetch requires P2P KV transfer")
            for source in mem_managers:
                assert source.history_buffer.shape == self.mem_manager.history_buffer.shape
                assert source.history_offsets == self.mem_manager.history_offsets
            self.dp_history_ptrs = torch.tensor(
                [source.history_buffer.data_ptr() for source in mem_managers], dtype=torch.uint64, device="cuda"
            )
        assert move_token_indexes.numel() == mem_indexes.numel() == token_dp_indexes.numel()
        assert mem_indexes.numel() % DSV4_CACHE_BLOCK_SIZE == 0
        copy_dsv4_dp_history(
            self.dp_history_ptrs,
            self.mem_manager,
            move_token_indexes,
            token_dp_indexes,
            mem_indexes,
            len(mem_managers) // dp_size_in_node,
            rank_in_dp,
        )

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        """Copy packed history pages; continuation is restored by the request manager."""
        manager = self.mem_manager
        # Each allocation page contains one or more independently packed history blocks.
        src_pages = (src_mem_index[::DSV4_CACHE_BLOCK_SIZE] // DSV4_CACHE_BLOCK_SIZE).to(
            device="cuda", dtype=torch.int64
        )
        dst_pages = (dst_mem_index[::DSV4_CACHE_BLOCK_SIZE] // DSV4_CACHE_BLOCK_SIZE).to(
            device="cuda", dtype=torch.int64
        )
        for buffer in manager.kv_buffer:
            if buffer.numel():
                buffer.index_copy_(1, dst_pages, buffer.index_select(1, src_pages))

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        raise NotImplementedError("DeepSeek-V4 writes packed KV using request-owned SWA slots")

    def offload_gpu_kv_to_cpu_cache(self, mem_indexes, page_indexes, page_readies, cpu_cache_client, req):
        from lightllm.utils.dist_utils import get_current_rank_in_dp
        from lightllm.server.router.model_infer.infer_batch import g_infer_context
        from lightllm.models.deepseek_v4.triton_kernel.cpu_cache_io import copy_history_pages, copy_checkpoints_to_pages

        # All ranks retain their checkpoints until the common completion event.
        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        # MLA KV and compression states are replicated across TP ranks.
        if get_current_rank_in_dp() != 0:
            return
        manager = self.mem_manager
        requests = g_infer_context.req_manager
        radix = g_infer_context.radix_cache
        page_tokens = manager.cache_config.cpu_cache_token_page_size
        shared_ids = radix.get_big_page_ids_by_node(req.shared_kv_node)
        checkpoints = []
        for page in range(page_indexes.numel()):
            end = (page + 1) * page_tokens
            if end > mem_indexes.numel():
                checkpoint = requests.small_page_buffers.buffer[req.tail_small_page_buffer_id]
            else:
                slot = shared_ids[page] if page < len(shared_ids) else req.hybrid_len_to_big_page_id[end]
                checkpoint = manager.big_page_buffers.buffer[slot]
            checkpoints.append(checkpoint.data_ptr())
        pointers = torch.tensor(checkpoints, dtype=torch.uint64, device="cpu", pin_memory=True).cuda(non_blocking=True)
        pages = cpu_cache_client.cpu_kv_cache_tensor.view(cpu_cache_client.cpu_kv_cache_tensor.shape[0], -1)
        copy_history_pages(manager, mem_indexes, page_indexes, pages, to_pages=True, readies=page_readies)
        copy_checkpoints_to_pages(manager, page_indexes, page_readies, pointers, pages)

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, cpu_cache_client, req):
        from lightllm.server.router.model_infer.infer_batch import g_infer_context
        from lightllm.models.deepseek_v4.triton_kernel.cpu_cache_io import copy_history_pages

        manager = self.mem_manager
        requests = g_infer_context.req_manager
        config = manager.cache_config
        page_tokens = config.cpu_cache_token_page_size
        checkpoint_offset = config.get_cpu_cache_full_att_bytes()
        pages = cpu_cache_client.cpu_kv_cache_tensor.view(cpu_cache_client.cpu_kv_cache_tensor.shape[0], -1)
        copy_history_pages(manager, mem_indexes, page_indexes, pages, to_pages=False)
        page_ids = page_indexes.tolist()
        full_pages = mem_indexes.numel() // page_tokens
        first_end = req.cur_kv_len // page_tokens * page_tokens - (full_pages - 1) * page_tokens
        slots = manager.big_page_buffers.alloc_state_cache(full_pages)
        if slots is None:
            raise RuntimeError("No hybrid checkpoint slots available for CPU cache load")
        for i, slot in enumerate(slots):
            manager.big_page_buffers.buffer[slot].copy_(pages[page_ids[i], checkpoint_offset:])
            req.hybrid_len_to_big_page_id[first_end + i * page_tokens] = slot
        if mem_indexes.numel() % page_tokens:
            if req.cur_kv_len == req.hybrid_cache_len:
                g_infer_context.radix_cache.free_one_small_page_buffer()
                tail = requests.small_page_buffers.alloc_one_state_cache()
                if tail is not None:
                    req.tail_small_page_buffer_id = tail
                    requests.small_page_buffers.buffer[tail].copy_(pages[page_ids[-1], checkpoint_offset:])
        # The common loader keeps these CPU pages referenced until its stream sync.
        checkpoint = manager.big_page_buffers.view(pages[page_ids[-1], checkpoint_offset:])
        requests.restore_state_from_tensors(req, checkpoint, req.cur_kv_len)
