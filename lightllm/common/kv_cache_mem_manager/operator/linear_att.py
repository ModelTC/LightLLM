import torch
from typing import List
from typing import TYPE_CHECKING
from .base import BaseMemManagerOperator
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.utils.log_utils import init_logger
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

if TYPE_CHECKING:
    from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
    from lightllm.server.router.model_infer.infer_batch import InferReq

logger = init_logger(__name__)


class LinearAttMemOperator(BaseMemManagerOperator):
    """
    只用于非量化的linear att 混合 full att的模型，列入 qwen3.5
    """

    def __init__(self, mem_manager):
        super().__init__(mem_manager)
        self.linear_config = LinearAttCacheConfig.load_from_args()

    def load_cpu_cache_to_gpu(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
        req: "InferReq",
    ):
        assert mem_indexes.is_cuda and page_indexes.is_cuda
        args = get_env_start_args()
        assert len(mem_indexes) % args.cpu_cache_token_page_size == 0
        assert args.cpu_cache_token_page_size == args.linear_att_hash_page_size * args.linear_att_page_block_num
        from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager

        mem_manager: Qwen3NextMemManager = self.mem_manager

        big_page_num = len(mem_indexes) // args.cpu_cache_token_page_size
        max_kv_len = req.cur_kv_len
        assert max_kv_len % args.cpu_cache_token_page_size == 0

        big_page_buffer_ids_cpu = []
        for i in range(big_page_num):
            page_id = mem_manager.linear_att_big_page_buffers.alloc_one_state_cache()
            assert page_id is not None
            req.linear_att_len_to_big_page_id[max_kv_len] = page_id
            big_page_buffer_ids_cpu.append(page_id)
            max_kv_len -= args.cpu_cache_token_page_size

        big_page_buffer_ids_cpu.reverse()
        big_page_buffer_ids_gpu = torch.tensor(big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu").cuda(
            non_blocking=True
        )

        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_cpu_cache_to_kv_buffer,
        )

        copy_cpu_cache_to_kv_buffer(
            mem_indexes=mem_indexes,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            page_indexes=page_indexes,
            gpu_full_att_kv_state=mem_manager.kv_buffer,
            cpu_kv_conv_state=mem_manager.linear_att_big_page_buffers.conv_state_cache.buffer,
            cpu_kv_ssm_state=mem_manager.linear_att_big_page_buffers.ssm_state_cache.buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        g_infer_context.req_manager.copy_big_page_buffer_to_linear_att_state(
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
        if not hasattr(self, "big_page_ids_buffer_store"):
            self.big_page_ids_buffer_store = torch.empty((1024 * 1024 * 4,), dtype=torch.int64, device="cuda")

        assert mem_indexes.is_cuda and page_indexes.is_cuda and page_readies.is_cuda
        args = get_env_start_args()
        assert len(mem_indexes) % args.cpu_cache_token_page_size == 0
        assert len(mem_indexes) // args.cpu_cache_token_page_size == len(page_indexes)
        assert len(mem_indexes) == len(page_indexes) * args.cpu_cache_token_page_size
        from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager

        mem_manager: Qwen3NextMemManager = self.mem_manager

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        big_page_buffer_ids_cpu = g_infer_context.radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
        max_kv_len = len(mem_indexes)
        start_kv_len = (len(big_page_buffer_ids_cpu) + 1) * args.cpu_cache_token_page_size
        for seq_len in range(start_kv_len, max_kv_len + 1, args.cpu_cache_token_page_size):
            page_id = req.linear_att_len_to_big_page_id[seq_len]
            big_page_buffer_ids_cpu.append(page_id)

        assert len(big_page_buffer_ids_cpu) == len(page_indexes) == len(page_readies)

        big_page_buffer_ids_cpu = torch.tensor(
            big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu", pin_memory=True
        )
        assert len(big_page_buffer_ids_cpu) <= self.big_page_ids_buffer_store.shape[0]
        big_page_buffer_ids_gpu = self.big_page_ids_buffer_store[0 : len(big_page_buffer_ids_cpu)]
        big_page_buffer_ids_gpu.copy_(big_page_buffer_ids_cpu, non_blocking=True)

        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_kv_buffer_to_cpu_cache,
        )

        copy_kv_buffer_to_cpu_cache(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            page_readies=page_readies,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            gpu_kv_full_att_state=mem_manager.kv_buffer,
            cpu_kv_conv_state=mem_manager.linear_att_big_page_buffers.conv_state_cache.buffer,
            cpu_kv_ssm_state=mem_manager.linear_att_big_page_buffers.ssm_state_cache.buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )
        return

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        # Qwen3Next 需要调整 layer_index
        layer_index = layer_index // self.linear_config.full_attention_interval
        from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

        mem_manager: MemoryManager = self.mem_manager
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import (
            destindex_copy_kv,
        )

        destindex_copy_kv(kv, mem_index, mem_manager.kv_buffer[layer_index])
        return
