import torch
from typing import Tuple
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.mamba_cache_mem_manager.cache_manager import MambaCacheManager
from lightllm.server.core.objs.start_args_type import StartArgs

logger = init_logger(__name__)


class Qwen3NextHybridMemManager(MemoryManager):
    @staticmethod
    def calculate_mamba_cache_size(
        start_args: StartArgs,
        max_total_token_num: int,
        mem_fraction: float,
        config: dict,
        head_linear_k_dim: int,
        num_linear_k_heads: int,
        head_linear_v_dim: int,
        num_linear_v_heads: int,
        tp_world_size: int,
        data_type: torch.dtype,
    ) -> int:
        """Calculate mamba cache size based on available memory and mamba_cache_ratio."""
        from lightllm.utils.profile_max_tokens import get_available_gpu_memory, get_total_gpu_memory
        import torch.distributed as dist

        use_ratio = max_total_token_num is None and start_args.mamba_cache_size is None
        print(f"mem_fraction ", mem_fraction, flush=True)

        world_size = dist.get_world_size()
        total_memory = get_total_gpu_memory()
        available_memory = get_available_gpu_memory(world_size) - total_memory * (1 - mem_fraction)

        conv_kernel_size = config["linear_conv_kernel_dim"]
        conv_dim = (
            head_linear_k_dim * num_linear_k_heads * 2 + head_linear_v_dim * num_linear_v_heads
        ) // tp_world_size

        num_linear_layers = config["n_layer"] - (config["n_layer"] // config["full_attention_interval"])

        conv_cell_size = num_linear_layers * conv_dim * (conv_kernel_size - 1) * torch._utils._element_size(data_type)

        ssm_dtype = torch.bfloat16 if start_args.mamba_ssm_data_type == "bfloat16" else torch.float32
        ssm_cell_size = (
            num_linear_layers
            * (num_linear_v_heads // tp_world_size)
            * head_linear_k_dim
            * head_linear_v_dim
            * torch._utils._element_size(ssm_dtype)
        )

        total_cell_size = conv_cell_size + ssm_cell_size

        if use_ratio:
            # mamba_cache_ratio = mamba_memory / total_cache_memory
            mamba_cache_ratio = start_args.mamba_cache_ratio if start_args.mamba_cache_ratio is not None else 0.5
            mamba_memory_gb = available_memory * mamba_cache_ratio
        else:
            mamba_memory_gb = available_memory
            mamba_cache_ratio = None

        mamba_cache_size = int(mamba_memory_gb * 1024 ** 3 / total_cell_size)

        if mamba_cache_size < start_args.running_max_req_size * 2:
            ratio = mamba_cache_ratio if mamba_cache_ratio is not None else 0.5
            raise ValueError(
                f"Insufficient memory for mamba cache allocation!\n\n"
                f"mamba_cache_size should be at least running_max_req_size * 2\n"
                f"Calculated mamba_cache_size ({mamba_cache_size}) < "
                f"running_max_req_size * 2 ({start_args.running_max_req_size * 2})\n\n"
                f"Memory budget:\n"
                f"  Available for mamba cache: {mamba_memory_gb:.2f} GB\n"
                f"  Memory per buffer: {total_cell_size / 1024 ** 2:.2f} MB\n"
                f"  Calculated buffers: {mamba_cache_size}\n"
                f"  Required buffers: {start_args.running_max_req_size}\n\n"
                f"Solutions:\n"
                f"  1. Reduce --running_max_req_size to {mamba_cache_size} or lower\n"
                f"  2. Increase --mamba_cache_ratio from {ratio} to "
                f"{start_args.running_max_req_size / mamba_cache_size * ratio:.3f} or higher\n"
                f"  3. Increase --mem_fraction to leave more memory for caches\n"
            )

        logger.info(
            f"Mamba cache allocation:\n"
            f"  Available memory: {mamba_memory_gb:.2f} GB\n"
            f"  Memory per buffer: {total_cell_size / 1024 ** 2:.2f} MB\n"
            f"  Calculated mamba_cache_size: {mamba_cache_size}"
        )

        return mamba_cache_size

    def __init__(
        self,
        full_attn_cache_size,
        linear_attn_cache_size,
        dtype,
        num_kv_heads,
        head_dim,
        layer_num,
        mtp_layer_num,
        full_attention_interval: int,
        conv_state_dtype: torch.dtype,
        conv_state_shape: Tuple[int, ...],
        ssm_state_dtype: torch.dtype,
        ssm_state_shape: Tuple[int, ...],
        max_req_num: int,
        always_copy=False,
        mem_fraction=0.9,
    ):

        self.full_attention_interval = full_attention_interval
        assert layer_num % full_attention_interval == 0
        self.layer_num = layer_num
        self.mtp_layer_num = mtp_layer_num
        self.full_attn_layer_num = layer_num // full_attention_interval
        self.linear_attn_layer_num = layer_num - self.full_attn_layer_num

        self.mamba_cache_mem_manager = MambaCacheManager(
            linear_attn_cache_size,
            self.linear_attn_layer_num,
            conv_state_dtype,
            conv_state_shape,
            ssm_state_dtype,
            ssm_state_shape,
        )

        super().__init__(full_attn_cache_size, dtype, num_kv_heads, head_dim, layer_num, always_copy, mem_fraction)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        # KV buffer layout: [None, None, None, kv_cache, None, None, None, kv_cache, ...,
        #                    None, kv_cache, mtp_kv_cache, mtp_kv_cache]
        # Only full attention layers and MTP layers have KV cache.
        self.kv_buffer = [None for _ in range(self.layer_num)]
        for layer_id in range(self.full_attn_layer_num):
            self.kv_buffer[(layer_id + 1) * self.full_attention_interval - 1] = torch.empty(
                (size + 1, 2 * head_num, head_dim), dtype=dtype, device="cuda"
            )

    def free_all(self):
        super().free_all()
        self.mamba_cache_mem_manager.free_all()
        return

    def get_cell_size(self):
        # Only full attention layers and MTP layers have KV cache
        kv_cache_layer_num = self.full_attn_layer_num + self.mtp_layer_num
        return 2 * self.head_num * self.head_dim * kv_cache_layer_num * torch._utils._element_size(self.dtype)

    def get_mamba_cache(self, layer_idx: int):
        layer_idx_in_linear = layer_idx - (layer_idx // self.full_attention_interval)
        return self.mamba_cache_mem_manager.get_mamba_cache(layer_idx_in_linear)
