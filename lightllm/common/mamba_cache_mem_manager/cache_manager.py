from typing import List, Tuple, Union

import torch
import numpy as np

from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from lightllm.common.basemodel.triton_kernel.mamba_buffer_copy import copy_mamba_buffer, fork_mamba_buffer
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt

logger = init_logger(__name__)


class LayerCache:
    def __init__(self, size: int, dtype: torch.dtype, shape: Tuple[int, ...], layer_num: int):
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.layer_num = layer_num

        self.buffer = torch.zeros((self.layer_num, size + 1, *shape), dtype=dtype, device="cuda")

    def get_cell_size(self):
        return np.prod(self.shape) * self.layer_num * torch._utils._element_size(self.dtype)


class MambaCacheManager:
    def __init__(
        self,
        size: int,
        layer_num: int,
        conv_state_dtype: torch.dtype,
        conv_state_shape: Tuple[int, ...],
        ssm_state_dtype: torch.dtype,
        ssm_state_shape: Tuple[int, ...],
    ):
        # init the mem state
        self.size = size
        self.mem_state = torch.arange(
            0, self.size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self._mem_state_return = torch.arange(
            0, self.size * 3, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self._return_start = 0
        self.mark_start = 0
        self.mark_end = self.size
        self.can_use_mem_size = self.size
        self.shared_can_use_token_num = SharedInt(
            f"{get_unique_server_name()}_mamba_cache_can_use_num_{get_current_rank_in_node()}"
        )
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self.HOLD_TOKEN_MEMINDEX = self.size

        # init the layer cache
        self.conv_state_cache = LayerCache(size, conv_state_dtype, conv_state_shape, layer_num)
        self.ssm_state_cache = LayerCache(size, ssm_state_dtype, ssm_state_shape, layer_num)
        self.HOLD_BUFFER_INDEX = size

        logger.warning(
            f"Linear attention state cache size: {size}\n"
            f"Conv state use : "
            f"{self.conv_state_cache.get_cell_size() * size / 1024 ** 3} GB Memory.\n"
            f"Ssm state use : "
            f"{self.ssm_state_cache.get_cell_size() * size / 1024 ** 3} GB Memory.\n"
        )

    def get_mamba_cache(self, layer_idx: int):
        conv_state = self.conv_state_cache.buffer[layer_idx]
        ssm_state = self.ssm_state_cache.buffer[layer_idx]
        return conv_state, ssm_state

    def copy_state_buffers(self, src_buffer_indexes: torch.Tensor, dst_buffer_indexes: torch.Tensor):
        copy_mamba_buffer(
            self.conv_state_cache.buffer, self.conv_state_cache.buffer, src_buffer_indexes, dst_buffer_indexes
        )
        copy_mamba_buffer(
            self.ssm_state_cache.buffer, self.ssm_state_cache.buffer, src_buffer_indexes, dst_buffer_indexes
        )

    def fork_state_buffers(self, src_buffer_index: torch.Tensor, dst_buffer_indexes: torch.Tensor):
        fork_mamba_buffer(
            self.conv_state_cache.buffer, self.conv_state_cache.buffer, src_buffer_index, dst_buffer_indexes
        )
        fork_mamba_buffer(
            self.ssm_state_cache.buffer, self.ssm_state_cache.buffer, src_buffer_index, dst_buffer_indexes
        )

    def fork_ssm_buffers(self, src_buffer_index: torch.Tensor, dst_buffer_indexes: torch.Tensor):
        """
        Fork ONLY SSM states (not conv states) from source indices to destination indices.

        This is used for MTP mode where each buffer maintains its own independent conv state,
        but SSM states need to be synchronized.
        """
        fork_mamba_buffer(
            self.ssm_state_cache.buffer, self.ssm_state_cache.buffer, src_buffer_index, dst_buffer_indexes
        )

    def alloc(self, need_size) -> torch.Tensor:
        if need_size > self.mark_end - self.mark_start:
            logger.error(f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}")
            assert False, "error alloc state"

        start = self.mark_start
        end = self.mark_start + need_size
        self.mark_start += need_size

        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        # 利用缓冲区返回，避免异步情况下的内存竞争
        if self._return_start + need_size > self._mem_state_return.shape[0]:
            self._return_start = 0
        ans = self._mem_state_return[self._return_start : self._return_start + need_size]
        ans.copy_(self.mem_state[start:end])
        self._return_start += need_size
        return ans

    def free(self, free_index: Union[torch.Tensor, List[int]]):
        """
        Free the allocated cache buffers and clear them.

        Args:
            free_index: Buffer indices to free (tensor or list of ints)
        """
        # Convert to tensor if needed for indexing
        if isinstance(free_index, list):
            free_index_tensor = torch.tensor(free_index, dtype=torch.long, device="cuda")
        else:
            free_index_tensor = free_index.to(device="cuda", dtype=torch.long)

        # Clear the buffers for the freed indices
        # Shape: [layer_num, buffer_index, *shape]
        self.conv_state_cache.buffer[:, free_index_tensor, ...] = 0
        self.ssm_state_cache.buffer[:, free_index_tensor, ...] = 0

        # update the mem state
        end = self.mark_start
        start = self.mark_start - len(free_index)
        assert start >= 0, f"error free state start: {self.mark_start} free len {len(free_index)}"

        if isinstance(free_index, list):
            free_index_tensor = torch.tensor(free_index, dtype=self.mem_state.dtype, device=self.mem_state.device)
            self.mem_state[start:end] = free_index_tensor
        else:
            # 从 gpu 到 cpu 的拷贝操作是流内阻塞操作
            self.mem_state[start:end] = free_index

        self.mark_start -= len(free_index)

        self.can_use_mem_size += len(free_index)
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        if self.can_use_mem_size == len(self.mem_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")

        return

    def free_all(self):
        self.conv_state_cache.buffer[:] = 0
        self.ssm_state_cache.buffer.fill_(0)
        self.can_use_mem_size = len(self.mem_state)
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self.mem_state.numpy()[:] = list(range(0, len(self.mem_state)))
        self.mark_start = 0
        self.mark_end = len(self.mem_state)

        return

    def resize_mem(self, new_size):
        """
        just for test code
        """
        self.size = new_size
        self.mem_state = torch.arange(
            0, self.size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_start = 0
        self.mark_end = self.size
        self.can_use_mem_size = self.size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return


class ReadOnlyStaticsMambaCacheManager:
    """
    读取一些统计信息
    """

    def __init__(self) -> None:
        args = get_env_start_args()
        self.global_world_size = args.tp
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = self.global_world_size // args.dp
        # 兼容多机 dp size=1 纯 tp 模式的情况
        self.is_multinode_tp = args.dp == 1 and args.nnodes > 1
        self.shared_tp_can_use_token_nums = [
            SharedInt(f"{get_unique_server_name()}_mamba_cache_can_use_num_{rank_in_node}")
            for rank_in_node in range(0, self.node_world_size, self.dp_world_size)
        ]

    def get_unrefed_token_num(self, dp_rank_in_node: int):
        if self.is_multinode_tp:
            return self.shared_tp_can_use_token_nums[0].get_value()
        return self.shared_tp_can_use_token_nums[dp_rank_in_node].get_value()
