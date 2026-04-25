import torch
import triton
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.linear_att_cache_manager import LinearAttCacheConfig, LinearAttCacheManager
from .operator import LinearAttMemOperator
from typing import Tuple, Any

logger = init_logger(__name__)


class Qwen3NextMemManager(MemoryManager):
    operator_class = LinearAttMemOperator

    def __init__(
        self,
        size,
        dtype,
        num_kv_heads,
        head_dim,
        full_att_layer_num,
        linear_config: LinearAttCacheConfig,
        always_copy=False,
        mem_fraction=0.9,
    ):
        self.linear_config = linear_config

        super().__init__(size, dtype, num_kv_heads, head_dim, full_att_layer_num, always_copy, mem_fraction)

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        layer_index = layer_index // self.linear_config.full_attention_interval
        return super().get_att_input_params(layer_index)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        # TODO 初始化线性 att 对应的部分 buffer.
        self._init_linear_att_buffers()
        return

    def _init_linear_att_buffers(self):
        big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        # 申请大页可能需要对应的资源
        self.linear_att_big_page_buffers = LinearAttCacheManager(
            size=triton.cdiv(self.size, big_page_token_num),
            linear_config=self.linear_config,
        )
        return

    def _free_buffers(self):
        super()._free_buffers()
        self._free_linear_att_buffers()
        return

    def _free_linear_att_buffers(self):
        self.linear_att_big_page_buffers = None
        return
