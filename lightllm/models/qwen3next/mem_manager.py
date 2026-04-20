import torch
import torch.distributed as dist
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.kv_buffer.hybrid_kv_buffer import HybridKvBuffer
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.profile_max_tokens import get_available_gpu_memory, get_total_gpu_memory
from lightllm.common.mamba_cache_mem_manager.config_objs import LinearAttCacheConfig

logger = init_logger(__name__)


class Qwen3NextHybridMemManager(MemoryManager):
    def __init__(
        self,
        size,
        dtype,
        num_kv_heads,
        head_dim,
        layer_num,
        linear_config: LinearAttCacheConfig,
        always_copy=False,
        mem_fraction=0.9,
    ):
        self.linear_config = linear_config

        super().__init__(size, dtype, num_kv_heads, head_dim, layer_num, always_copy, mem_fraction)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        # TODO 初始化线性 att 对应的部分 buffer.
        return
