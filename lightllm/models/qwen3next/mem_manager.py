import torch
import torch.distributed as dist
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.kv_buffer.hybrid_kv_buffer import HybridKvBuffer
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.profile_max_tokens import get_available_gpu_memory, get_total_gpu_memory

logger = init_logger(__name__)


class Qwen3NextHybridMemManager(MemoryManager):
    def __init__(
        self,
        full_attn_cache_size,
        linear_attn_cache_size,
        dtype,
        num_kv_heads,
        head_dim,
        layer_num,
        full_attention_interval: int,
        conv_state_dtype: torch.dtype,
        ssm_state_dtype: torch.dtype,
        conv_kernel_size: int,
        num_linear_k_heads: int,
        num_linear_v_heads: int,
        head_linear_k_dim: int,
        head_linear_v_dim: int,
        max_req_num: int,
        always_copy=False,
        mem_fraction=0.9,
        network_config: dict = None,
    ):

        self.full_attention_interval = full_attention_interval
        assert layer_num % full_attention_interval == 0
        self.transformer_layer_num = layer_num
        self.full_attn_layer_num = layer_num // full_attention_interval
        self.linear_attn_layer_num = layer_num - self.full_attn_layer_num
        self.linear_attn_cache_size = linear_attn_cache_size
        self.conv_state_dtype = conv_state_dtype
        self.ssm_state_dtype = ssm_state_dtype
        self.conv_kernel_size = conv_kernel_size
        self.num_linear_k_heads = num_linear_k_heads
        self.num_linear_v_heads = num_linear_v_heads
        self.head_linear_k_dim = head_linear_k_dim
        self.head_linear_v_dim = head_linear_v_dim

        super().__init__(
            full_attn_cache_size, dtype, num_kv_heads, head_dim, self.full_attn_layer_num, always_copy, mem_fraction
        )

    def profile_size(self, mem_fraction):
        if self.size is not None:
            return

        world_size = dist.get_world_size()
        total_memory = get_total_gpu_memory()
        available_memory = get_available_gpu_memory(world_size) - total_memory * (1 - mem_fraction)

        conv_dim = (
            self.head_linear_k_dim * self.num_linear_k_heads * 2 + self.head_linear_v_dim * self.num_linear_v_heads
        )
        mamba_cell_size = (
            self.linear_attn_layer_num
            * conv_dim
            * (self.conv_kernel_size - 1)
            * torch._utils._element_size(self.conv_state_dtype)
        ) + (
            self.linear_attn_layer_num
            * self.num_linear_v_heads
            * self.head_linear_k_dim
            * self.head_linear_v_dim
            * torch._utils._element_size(self.ssm_state_dtype)
        )

        if self.linear_attn_cache_size is None:
            start_args = get_env_start_args()
            mamba_cache_ratio = start_args.mamba_cache_ratio if start_args.mamba_cache_ratio is not None else 0.5
            self.linear_attn_cache_size = int(available_memory * mamba_cache_ratio * 1024 ** 3 / mamba_cell_size)
        reserved_mamba_memory = self.linear_attn_cache_size * mamba_cell_size / (1024 ** 3)
        available_memory -= reserved_mamba_memory

        cell_size = self.get_cell_size()
        self.size = int(available_memory * 1024 ** 3 / cell_size)
        if world_size > 1:
            tensor = torch.tensor(self.size, dtype=torch.int64, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            self.size = tensor.item()

        logger.info(
            f"{available_memory} GB space is available for full attention kv cache after reserving "
            f"{reserved_mamba_memory} GB for mamba cache\n"
            f"{cell_size / 1024 ** 2} MB is the size of one token kv cache\n"
            f"{self.size} is the profiled max_total_token_num with the mem_fraction {mem_fraction}\n"
        )
        return

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = HybridKvBuffer(
            torch.empty((layer_num, size + 1, 2 * head_num, head_dim), dtype=dtype, device="cuda"),
            head_num=head_num,
            transformer_layer_num=self.transformer_layer_num,
            full_attention_interval=self.full_attention_interval,
            mamba_cache_size=self.linear_attn_cache_size,
            linear_attn_layer_num=self.linear_attn_layer_num,
            conv_state_dtype=self.conv_state_dtype,
            ssm_state_dtype=self.ssm_state_dtype,
            conv_kernel_size=self.conv_kernel_size,
            num_linear_k_heads=self.num_linear_k_heads,
            num_linear_v_heads=self.num_linear_v_heads,
            head_linear_k_dim=self.head_linear_k_dim,
            head_linear_v_dim=self.head_linear_v_dim,
        )

    def free_all(self):
        super().free_all()
        self.kv_buffer.mamba_cache_manager.free_all()
        return
