from typing import List, Tuple, Union
import threading

import torch
import numpy as np

from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from lightllm.common.allocator_utils import TokenAllocator
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.common.basemodel.triton_kernel.mamba_buffer_copy import (
    copy_buffer_p2p as triton_copy_buffer_p2p,
    copy_buffer_broadcast as triton_copy_buffer_broadcast,
)

logger = init_logger(__name__)

MAMBA_CACHE_CAN_USE_NUM_SHM_NAME = f"{get_unique_server_name()}_mamba_cache_can_use_num"


class LayerCache:
    def __init__(self, size: int, dtype: torch.dtype, shape: Tuple[int, ...], layer_num: int):
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.layer_num = layer_num

        self.buffer = torch.zeros((self.layer_num, size + 1, *shape), dtype=dtype, device="cuda")

    def get_cell_size(self):
        return np.prod(self.shape) * self.layer_num * torch._utils._element_size(self.dtype)


class MambaCacheManager(TokenAllocator):
    def __init__(
        self,
        size: int,
        layer_num: int,
        conv_state_dtype: torch.dtype,
        conv_state_shape: Tuple[int, ...],
        ssm_state_dtype: torch.dtype,
        ssm_state_shape: Tuple[int, ...],
    ):
        super().__init__(size, f"{MAMBA_CACHE_CAN_USE_NUM_SHM_NAME}_{get_current_rank_in_node()}")
        self.conv_state_cache = LayerCache(size, conv_state_dtype, conv_state_shape, layer_num)
        self.ssm_state_cache = LayerCache(size, ssm_state_dtype, ssm_state_shape, layer_num)
        self.HOLD_BUFFER_INDEX = size

        logger.info(
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

    def copy_buffer_p2p(self, src_buffer_indexes: torch.Tensor, dst_buffer_indexes: torch.Tensor):
        """
        Copy buffers from source indices to destination indices using optimized Triton kernel.

        Args:
            src_buffer_indexes: Source buffer indices (1D tensor)
            dst_buffer_indexes: Destination buffer indices (1D tensor)
        """
        assert src_buffer_indexes.dim() == 1
        assert dst_buffer_indexes.dim() == 1
        assert src_buffer_indexes.shape[0] == dst_buffer_indexes.shape[0]

        # Use Triton kernel for efficient parallel copy
        triton_copy_buffer_p2p(
            self.conv_state_cache.buffer,
            self.conv_state_cache.buffer,
            src_buffer_indexes,
            dst_buffer_indexes,
        )
        triton_copy_buffer_p2p(
            self.ssm_state_cache.buffer,
            self.ssm_state_cache.buffer,
            src_buffer_indexes,
            dst_buffer_indexes,
        )
        return

    def copy_buffer_broadcast(self, src_buffer_index: torch.Tensor, dst_buffer_indexes: torch.Tensor):
        assert src_buffer_index.dim() == 1
        assert dst_buffer_indexes.dim() == 2
        assert src_buffer_index.shape[0] == dst_buffer_indexes.shape[0]

        # Use Triton kernel for efficient broadcast copy
        triton_copy_buffer_broadcast(
            self.conv_state_cache.buffer,
            self.conv_state_cache.buffer,
            src_buffer_index,
            dst_buffer_indexes,
        )
        triton_copy_buffer_broadcast(
            self.ssm_state_cache.buffer,
            self.ssm_state_cache.buffer,
            src_buffer_index,
            dst_buffer_indexes,
        )
        return

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
        self.conv_state_cache.buffer[:, free_index_tensor].zero_()
        self.ssm_state_cache.buffer[:, free_index_tensor].zero_()

        # Call parent's free method to update allocator state
        super().free(free_index)
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
            SharedInt(f"{MAMBA_CACHE_CAN_USE_NUM_SHM_NAME}_{rank_in_node}")
            for rank_in_node in range(0, self.node_world_size, self.dp_world_size)
        ]

    def get_unrefed_token_num(self, dp_rank_in_node: int):
        if self.is_multinode_tp:
            return self.shared_tp_can_use_token_nums[0].get_value()
        return self.shared_tp_can_use_token_nums[dp_rank_in_node].get_value()
