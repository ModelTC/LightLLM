import atexit
import torch
import numpy as np
from multiprocessing import shared_memory
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_rank_in_dp
from lightllm.server.router.dynamic_prompt.shared_arr import SharedArray
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)


def routing_dtype_id_to_np(dtype_id: int):
    if dtype_id == 1:
        return np.int8
    elif dtype_id == 2:
        return np.int16
    return np.int32


def get_routing_config_shm() -> SharedArray:
    service_name = get_unique_server_name()
    return SharedArray(f"{service_name}_routing_config", shape=(4,), dtype=np.int32)


class RoutingCaptureManager:
    def __init__(
        self,
        num_moe_layers: int,
        topk: int,
        num_experts: int,
        kv_cache_size: int,
        max_capture_tokens: int,
    ):
        self.num_moe_layers = num_moe_layers
        self.topk = topk
        self.num_experts = num_experts
        self.kv_cache_size = kv_cache_size

        self.dtype = torch.int8 if num_experts <= 127 else torch.int16
        dtype_bytes = 1 if self.dtype == torch.int8 else 2

        # Shape: (num_moe_layers, kv_cache_size, topk) — on CPU to save GPU memory.
        # Written after forward() via flush_to_routing_buffer(), read on request finish.
        routing_buffer_size = num_moe_layers * kv_cache_size * topk * dtype_bytes
        self.routing_buffer = torch.zeros(
            (num_moe_layers, kv_cache_size, topk),
            dtype=self.dtype,
            device="cpu",
        )

        # Capture buffers: simple contiguous tensors written to during forward().
        capture_buf_size = max_capture_tokens * num_moe_layers * topk * dtype_bytes
        self._capture_buffer = [
            torch.zeros((max_capture_tokens, num_moe_layers, topk), dtype=self.dtype, device="cuda") for _ in range(2)
        ]

        dtype_name = "int8" if self.dtype == torch.int8 else "int16"
        logger.info(
            f"RoutingCaptureManager initialized: {num_moe_layers} MoE layers, topk={topk}, "
            f"routing_buffer(cpu)={routing_buffer_size / 1024 / 1024:.2f}MB, "
            f"capture_buffer={capture_buf_size / 1024 / 1024:.2f}MB x2, dtype={dtype_name}"
        )

    @property
    def np_dtype(self):
        return np.int8 if self.dtype == torch.int8 else np.int16

    @property
    def dtype_id(self) -> int:
        return 1 if self.dtype == torch.int8 else 2

    def capture(self, moe_layer_index: int, topk_ids: torch.Tensor, microbatch_index: int = 0) -> None:
        num_tokens = topk_ids.shape[0]
        self._capture_buffer[microbatch_index][:num_tokens, moe_layer_index, :] = topk_ids.to(self.dtype)

    def flush_to_routing_buffer(self, mem_indexes: torch.Tensor, num_tokens: int, microbatch_index: int = 0) -> None:
        buf = self._capture_buffer[microbatch_index][:num_tokens]  # (num_tokens, num_moe_layers, topk)
        buf_t = buf.permute(1, 0, 2).cpu()
        self.routing_buffer[:, mem_indexes[:num_tokens].cpu(), :] = buf_t

    def extract_routing_data(self, mem_indexes: torch.Tensor) -> np.ndarray:
        cpu_indexes = mem_indexes.cpu() if mem_indexes.is_cuda else mem_indexes
        return self.routing_buffer[:, cpu_indexes, :].numpy()


g_routing_capture_manager: Optional[RoutingCaptureManager] = None


def create_routing_capture_manager(
    num_moe_layers: int,
    topk: int,
    num_experts: int,
    kv_cache_size: int,
    max_capture_tokens: int,
) -> None:
    global g_routing_capture_manager
    assert g_routing_capture_manager is None, "RoutingCaptureManager already exists"
    g_routing_capture_manager = RoutingCaptureManager(
        num_moe_layers=num_moe_layers,
        topk=topk,
        num_experts=num_experts,
        kv_cache_size=kv_cache_size,
        max_capture_tokens=max_capture_tokens,
    )


def preallocate_routing_shm_pool(max_req_num: int, num_moe_layers: int, max_tokens: int, topk: int, np_dtype) -> None:
    """Pre-allocate POSIX SHM segments for all request slots.

    Each segment is sized for the maximum possible routing data so it can be
    reused across requests without create/destroy overhead.
    """
    dtype_bytes = np.dtype(np_dtype).itemsize
    segment_size = num_moe_layers * max_tokens * topk * dtype_bytes
    service_name = get_unique_server_name()

    for i in range(max_req_num):
        name = f"{service_name}_shm_routing_{i}"
        shm = create_or_link_shm(name, segment_size, auto_cleanup=True)
        shm.close()  # close handle; SHM persists in /dev/shm

    logger.info(
        f"Pre-allocated {max_req_num} routing SHM segments, "
        f"each {segment_size / 1024:.1f} KB (total {max_req_num * segment_size / 1024 / 1024:.1f} MB)"
    )


def cleanup_routing_shm_pool() -> None:
    """Unlink all pre-allocated routing SHM segments. Called at server shutdown."""
    try:
        from lightllm.utils.envs_utils import get_env_start_args

        args = get_env_start_args()
    except Exception:
        return

    service_name = get_unique_server_name()

    for i in range(args.running_max_req_size):
        name = f"{service_name}_shm_routing_{i}"
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except Exception:
            pass

    config_name = f"{service_name}_routing_config"
    try:
        shm = shared_memory.SharedMemory(name=config_name)
        shm.close()
        shm.unlink()
    except Exception:
        pass


def init_routing_capture(model, num_moe_layers: int) -> None:
    dp_rank = get_current_rank_in_dp()
    logger.info(f"init_routing_capture called: num_moe_layers={num_moe_layers}, dp_rank={dp_rank}")
    if dp_rank != 0:
        logger.info(f"Skipping routing capture initialization on dp_rank={dp_rank}")
        return

    if num_moe_layers == 0:
        logger.warning(
            "enable_return_routed_experts is set but no MoE layers found. Routing capture will not be enabled."
        )
        return

    num_experts = model.config.get("n_routed_experts", model.config.get("num_experts", 0))
    topk = model.config.get("num_experts_per_tok", 0)
    assert num_experts > 0 and topk > 0

    from lightllm.utils.envs_utils import get_env_start_args

    args = get_env_start_args()

    # Capture buffer must fit the max tokens in any single forward call.
    # For prefill that's batch_max_tokens; for decode it's graph_max_batch_size.
    batch_max_tokens = args.batch_max_tokens or args.max_req_total_len or 8192
    max_capture_tokens = max(batch_max_tokens, args.graph_max_batch_size)

    logger.info(
        f"Initializing routing capture: num_moe_layers={num_moe_layers}, "
        f"topk={topk}, num_experts={num_experts}, max_capture_tokens={max_capture_tokens}"
    )

    create_routing_capture_manager(
        num_moe_layers=num_moe_layers,
        topk=topk,
        num_experts=num_experts,
        kv_cache_size=model.mem_manager.size + 1,
        max_capture_tokens=max_capture_tokens,
    )

    mgr = g_routing_capture_manager
    np_dtype = mgr.np_dtype
    dtype_id = mgr.dtype_id

    max_req_total_len = args.max_req_total_len

    # Write config to cross-process SHM
    shm = get_routing_config_shm()
    shm.arr[0] = num_moe_layers
    shm.arr[1] = topk
    shm.arr[2] = dtype_id
    shm.arr[3] = max_req_total_len
    logger.info(
        f"Shared routing config set: num_moe_layers={num_moe_layers}, topk={topk}, "
        f"dtype_id={dtype_id}, max_tokens={max_req_total_len}"
    )

    preallocate_routing_shm_pool(
        max_req_num=args.running_max_req_size,
        num_moe_layers=num_moe_layers,
        max_tokens=max_req_total_len,
        topk=topk,
        np_dtype=np_dtype,
    )

    atexit.register(cleanup_routing_shm_pool)
