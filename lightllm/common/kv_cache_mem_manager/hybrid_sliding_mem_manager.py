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
        self.enable_prompt_cache = not args.disable_dynamic_prompt_cache
        self.small_page_num = args.linear_att_cache_size if self.enable_prompt_cache else 0
        self.cpu_cache_temp_page_num = 2 if args.enable_cpu_cache else 0
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

    def profile_size(self, mem_fraction):
        torch.cuda.empty_cache()
        world_size = dist.get_world_size()
        available_memory = get_available_gpu_memory(world_size)
        if self.size is None:
            available_memory -= get_total_gpu_memory() * (1 - mem_fraction)
        available_bytes = int(available_memory * 1024 ** 3)
        cell_size = self.get_cell_size()
        state_bytes = self.sliding_config.get_state_nbytes()
        # Runtime windows already exist. Reserve the hold token, small pages
        # and CPU-transfer slots before sizing full KV and big checkpoints.
        fixed_bytes = cell_size + (self.small_page_num + self.cpu_cache_temp_page_num) * state_bytes
        big_page_state_bytes = state_bytes if self.enable_prompt_cache else 0
        if self.size is None:
            if available_bytes < fixed_bytes + cell_size + big_page_state_bytes:
                raise ValueError(
                    "Insufficient GPU memory for sliding-window checkpoints and full KV; "
                    "reduce --linear_att_cache_size or --running_max_req_size."
                )
            # Each complete page costs B full-KV tokens plus one checkpoint.
            # A partial page also needs one checkpoint before it can hold tokens.
            page_bytes = self.big_page_token_num * cell_size + big_page_state_bytes
            page_num, tail_bytes = divmod(available_bytes - fixed_bytes, page_bytes)
            self.size = page_num * self.big_page_token_num + max(0, (tail_bytes - big_page_state_bytes) // cell_size)
            if world_size > 1:
                size_tensor = torch.tensor(self.size, dtype=torch.int64, device="cuda")
                dist.all_reduce(size_tensor, op=dist.ReduceOp.MIN)
                self.size = size_tensor.item()

        big_page_num = triton.cdiv(self.size, self.big_page_token_num) if self.enable_prompt_cache else 0
        cache_bytes = fixed_bytes + self.size * cell_size + big_page_num * state_bytes
        if cache_bytes > available_bytes:
            raise ValueError(
                "Requested full KV and sliding-window checkpoints exceed available GPU memory; "
                "reduce --max_total_token_num, --linear_att_cache_size or --running_max_req_size."
            )
        logger.info(
            f"Sliding-window cache budget: {self.size} full-KV tokens, "
            f"{big_page_num} big pages, {self.small_page_num} small pages, "
            f"{self.cpu_cache_temp_page_num} CPU-cache staging states, "
            f"{cache_bytes / 1024 ** 3:.2f} GiB (runtime windows already allocated)"
        )

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        big_page_num = triton.cdiv(size, self.big_page_token_num) if self.enable_prompt_cache else 0
        # Keep the existing radix-cache contract; no second alias is needed.
        self.linear_att_big_page_buffers = SlidingWindowStateCacheManager(
            size=big_page_num + self.cpu_cache_temp_page_num,
            sliding_config=self.sliding_config,
            keep_num=self.cpu_cache_temp_page_num,
        )
        if self.cpu_cache_temp_page_num:
            self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 2
            self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 1
        self.sliding_small_page_buffers = SlidingWindowStateCacheManager(
            size=self.small_page_num,
            sliding_config=self.sliding_config,
        )

    def get_att_input_params(self, layer_index: int):
        return super().get_att_input_params(self.sliding_config.get_full_layer_index(layer_index))

    def _free_buffers(self):
        super()._free_buffers()
        self.linear_att_big_page_buffers = None
        self.sliding_small_page_buffers = None
