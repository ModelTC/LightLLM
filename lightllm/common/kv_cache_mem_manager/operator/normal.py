import torch
from typing import TYPE_CHECKING, List
from .base import BaseMemManagerOperator
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient

logger = init_logger(__name__)


class NormalMemOperator(BaseMemManagerOperator):
    def load_cpu_kv_to_gpu(
        self, mem_indexes: torch.Tensor, page_indexes: torch.Tensor, cpu_cache_client: "CpuKvCacheClient"
    ):
        assert mem_indexes.is_cuda and page_indexes.is_cuda
        args = get_env_start_args()
        assert len(mem_indexes) % args.cpu_cache_token_page_size == 0
        from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

        mem_manager: MemoryManager = self.mem_manager
        from lightllm.common.basemodel.triton_kernel.kv_cache_offload import load_cpu_kv_to_gpu

        load_cpu_kv_to_gpu(
            gpu_mem_indexes=mem_indexes,
            gpu_kv_cache=mem_manager.kv_buffer,
            gpu_kv_cache_scale=None,
            cpu_kv_cache=cpu_cache_client.cpu_kv_cache_tensor,
            cpu_kv_cache_scale=None,
            tp_index=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            grid_num=16,
        )
        return

    def offload_gpu_kv_to_cpu(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        page_readies: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
    ):
        assert mem_indexes.is_cuda and page_indexes.is_cuda
        args = get_env_start_args()
        assert len(mem_indexes) % args.cpu_cache_token_page_size == 0
        assert len(mem_indexes) // args.cpu_cache_token_page_size == len(page_indexes)
        from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

        mem_manager: MemoryManager = self.mem_manager

        from lightllm.common.basemodel.triton_kernel.kv_cache_offload import offload_gpu_kv_to_cpu

        offload_gpu_kv_to_cpu(
            token_indexes=mem_indexes,
            gpu_kv_cache=mem_manager.kv_buffer,
            gpu_kv_cache_scale=None,
            cpu_kv_cache=cpu_cache_client.cpu_kv_cache_tensor,
            cpu_kv_cache_scale=None,
            page_indexes=page_indexes,
            page_readies=page_readies,
            tp_index=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            grid_num=16,
        )
        return

    def copy_kv_from_other_dp_ranks(
        self,
        mem_managers: List,
        move_token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        mem_indexes: torch.Tensor,
        dp_size_in_node: int,
        rank_in_dp: int,
    ):
        if not hasattr(self, "mem_ptrs_tensor"):
            # 构建一个2D tensor，shape为(layer_num, mem_num)
            mems_ptr_list = []
            for i in range(0, len(mem_managers)):
                mems_ptr_list.append(mem_managers[i].kv_buffer.data_ptr())
            self.mem_ptrs_tensor = torch.tensor(mems_ptr_list, dtype=torch.uint64, device="cpu", pin_memory=True)

        from lightllm.common.kv_trans_kernel.kv_trans_v2 import kv_trans_for_dp

        # 一次性传输所有层
        kv_trans_for_dp(
            input_mems=self.mem_ptrs_tensor.cuda(non_blocking=True),
            input_idx=move_token_indexes,
            input_dp_idx=token_dp_indexes,
            output=self.mem_manager.kv_buffer,
            output_idx=mem_indexes,
            dp_size_in_node=dp_size_in_node,
            rank_in_dp=rank_in_dp,
        )
        return
