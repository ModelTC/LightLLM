import torch
import torch.distributed as dist
import triton

from lightllm.common.sliding_window_cache_manager import SlidingWindowStateCacheManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.profile_max_tokens import get_available_gpu_memory, get_total_gpu_memory

from .mem_manager import MemoryManager
from .operator.hybrid_sliding import HybridSlidingMemOperator

logger = init_logger(__name__)


class HybridSlidingMemoryManager(MemoryManager):
    """Token-granular full KV plus request-granular sliding-window KV."""

    operator_class = HybridSlidingMemOperator

    def __init__(self, size, sliding_config, always_copy=False, mem_fraction=0.9):
        self.sliding_config = sliding_config
        args = get_env_start_args()
        self.enable_prompt_cache = args.use_dynamic_prompt_cache
        self.small_page_num = args.linear_att_cache_size if self.enable_prompt_cache else 0
        self.big_page_token_num = args.linear_att_page_block_num * args.linear_att_hash_page_size
        super().__init__(
            size=size,
            dtype=sliding_config.dtype,
            head_num=sliding_config.full_head_num,
            head_dim=sliding_config.full_head_dim,
            layer_num=sliding_config.full_layer_num,
            always_copy=always_copy,
            mem_fraction=mem_fraction,
        )

    def _big_page_num(self, token_num):
        return max(1, triton.cdiv(token_num, self.big_page_token_num)) if self.enable_prompt_cache else 0

    def _cache_nbytes(self, token_num):
        # Runtime windows already exist when profiling. Reserve BOTH GPU page
        # pools here, plus the full-KV hold token and the final partial big page.
        return (token_num + 1) * self.get_cell_size() + (
            self.small_page_num + self._big_page_num(token_num)
        ) * self.sliding_config.get_state_nbytes()

    def _profile_token_num(self, available_bytes):
        if self._cache_nbytes(1) > available_bytes:
            raise ValueError(
                "Insufficient GPU memory for sliding-window checkpoints and full KV: "
                f"{available_bytes / 1024 ** 3:.2f} GiB available, "
                f"{self.small_page_num} small pages at "
                f"{self.sliding_config.get_state_nbytes() / 1024 ** 2:.2f} MiB/page. "
                "Reduce --linear_att_cache_size or --running_max_req_size."
            )
        low, high = 1, available_bytes // self.get_cell_size()
        while low < high:
            mid = (low + high + 1) // 2
            if self._cache_nbytes(mid) <= available_bytes:
                low = mid
            else:
                high = mid - 1
        return low

    def profile_size(self, mem_fraction):
        torch.cuda.empty_cache()
        world_size = dist.get_world_size()
        available_memory = get_available_gpu_memory(world_size)
        if self.size is None:
            available_memory -= get_total_gpu_memory() * (1 - mem_fraction)
            self.size = self._profile_token_num(int(available_memory * 1024 ** 3))
            if world_size > 1:
                size_tensor = torch.tensor(self.size, dtype=torch.int64, device="cuda")
                dist.all_reduce(size_tensor, op=dist.ReduceOp.MIN)
                self.size = size_tensor.item()
        elif self._cache_nbytes(self.size) > int(available_memory * 1024 ** 3):
            raise ValueError(
                "Requested full KV and sliding-window checkpoints exceed available GPU memory; "
                "reduce --max_total_token_num, --linear_att_cache_size or --running_max_req_size."
            )
        logger.info(
            f"Sliding-window cache budget: {self.size} full-KV tokens, "
            f"{self._big_page_num(self.size)} big pages, {self.small_page_num} small pages, "
            f"{self._cache_nbytes(self.size) / 1024 ** 3:.2f} GiB (runtime windows already allocated)"
        )

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        # Keep the existing radix-cache contract; no second alias is needed.
        self.linear_att_big_page_buffers = SlidingWindowStateCacheManager(
            size=self._big_page_num(size),
            sliding_config=self.sliding_config,
        )
        self.sliding_small_page_buffers = SlidingWindowStateCacheManager(
            size=self.small_page_num,
            sliding_config=self.sliding_config,
        )

    def get_att_input_params(self, layer_index: int):
        return super().get_att_input_params(self.sliding_config.get_full_layer_index(layer_index))

    def get_full_cache_layer_index(self, layer_index: int):
        return self.sliding_config.get_full_layer_index(layer_index)

    def _free_buffers(self):
        super()._free_buffers()
        self.linear_att_big_page_buffers = None
        self.sliding_small_page_buffers = None
