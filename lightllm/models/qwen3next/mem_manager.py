import torch
from typing import Dict, List, Protocol, Set, Union, Tuple
from typing_extensions import override
from lightllm.utils.log_utils import init_logger
from lightllm.common.mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)


class MambaStateBufferConfig:
    def __init__(
        self,
        conv_state_dtype: torch.dtype,
        conv_state_shape: torch.Size,
        ssm_state_dtype: torch.dtype,
        ssm_state_shape: torch.Size,
    ):

        self.conv_state_dtype = conv_state_dtype
        self.conv_state_shape = conv_state_shape
        self.ssm_state_dtype = ssm_state_dtype
        self.ssm_state_shape = ssm_state_shape


class Qwen3NextMemoryManager(MemoryManager):
    def __init__(
        self,
        size,
        dtype,
        num_kv_heads,
        head_dim,
        layer_num,
        full_attention_interval: int,
        max_req_num: int,
        mamba_state_buffer_config: MambaStateBufferConfig,
        always_copy=False,
        mem_fraction=0.9,
    ):
        self.full_attention_interval = full_attention_interval
        self.max_req_num = max_req_num

        assert layer_num % full_attention_interval == 0
        self.layer_num_wo_mtp = layer_num
        self.full_attn_layer_num = layer_num // full_attention_interval
        self.linear_attn_layer_num = layer_num - self.full_attn_layer_num

        self.conv_state_dtype = mamba_state_buffer_config.conv_state_dtype
        self.conv_state_shape = mamba_state_buffer_config.conv_state_shape
        self.ssm_state_dtype = mamba_state_buffer_config.ssm_state_dtype
        self.ssm_state_shape = mamba_state_buffer_config.ssm_state_shape

        self.init_mamba_state_buffer()

        # allocate kv buffer pool secondly.
        super().__init__(size, dtype, num_kv_heads, head_dim, self.full_attn_layer_num, always_copy, mem_fraction)

    def init_mamba_state_buffer(self):
        self.conv_state_buffers = torch.empty(
            (self.linear_attn_layer_num, self.max_req_num + 1, *self.conv_state_shape),
            dtype=self.conv_state_dtype,
            device="cuda",
        )
        self.ssm_state_buffers = torch.empty(
            (self.linear_attn_layer_num, self.max_req_num + 1, *self.ssm_state_shape),
            dtype=self.ssm_state_dtype,
            device="cuda",
        )

    @override
    def get_kv_buffer(self, layer_index):
        assert (layer_index + 1) % self.full_attention_interval == 0, "layer_index is not full attention layer"
        return self.kv_buffer[layer_index // self.full_attention_interval]

    def get_mamba_state_buffer(self, layer_index) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (layer_index + 1) % self.full_attention_interval != 0, "layer_index is not linear attention layer"
        real_layer_index = layer_index - layer_index // self.full_attention_interval
        conv_states = self.conv_state_buffers[real_layer_index]
        ssm_states = self.ssm_state_buffers[real_layer_index]
        return conv_states, ssm_states

    @override
    def _free_buffers(self):
        super()._free_buffers()
        self.conv_state_buffers = None
        self.ssm_state_buffers = None

    @override
    def init_buffers(self):
        super().init_buffers()
        if self.conv_state_buffers is None and self.ssm_state_buffers is None:
            self.init_mamba_state_buffer()
        return
