import torch
import numpy as np
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_rank_in_dp
from lightllm.server.router.dynamic_prompt.shared_arr import SharedArray
from lightllm.utils.envs_utils import get_unique_server_name

logger = init_logger(__name__)


class SharedRoutingConfig:
    """Shared MoE routing configuration across processes."""

    def __init__(self):
        service_name = get_unique_server_name()
        self._shm = SharedArray(f"{service_name}_routing_config", shape=(2,), dtype=np.int32)

    @property
    def num_moe_layers(self) -> int:
        return int(self._shm.arr[0])

    @num_moe_layers.setter
    def num_moe_layers(self, value: int):
        self._shm.arr[0] = value

    @property
    def topk(self) -> int:
        return int(self._shm.arr[1])

    @topk.setter
    def topk(self, value: int):
        self._shm.arr[1] = value

    def is_initialized(self) -> bool:
        return self.num_moe_layers > 0 and self.topk > 0


_shared_routing_config: Optional[SharedRoutingConfig] = None


def get_shared_routing_config() -> SharedRoutingConfig:
    """Get or create the shared routing config."""
    global _shared_routing_config
    if _shared_routing_config is None:
        _shared_routing_config = SharedRoutingConfig()
    return _shared_routing_config


_moe_layer_counter: int = 0


def reset_moe_layer_counter() -> None:
    global _moe_layer_counter
    _moe_layer_counter = 0


def get_next_moe_layer_index() -> int:
    global _moe_layer_counter
    idx = _moe_layer_counter
    _moe_layer_counter += 1
    return idx


def get_moe_layer_count() -> int:
    return _moe_layer_counter


class RoutingCaptureManager:
    """Captures MoE routing decisions"""

    def __init__(
        self,
        num_moe_layers: int,
        topk: int,
        num_experts: int,
        batch_max_tokens: int,
        kv_cache_size: int,
        enable_overlap: bool = False,
    ):
        self.num_moe_layers = num_moe_layers
        self.topk = topk
        self.num_experts = num_experts
        self.batch_max_tokens = batch_max_tokens
        self.kv_cache_size = kv_cache_size

        self.dtype = torch.int8 if num_experts <= 127 else torch.int16
        dtype_bytes = 1 if self.dtype == torch.int8 else 2

        self.num_slots = 2 if enable_overlap else 1

        gpu_buffer_size = self.num_slots * num_moe_layers * batch_max_tokens * topk * dtype_bytes
        self.gpu_buffer = torch.zeros(
            (self.num_slots, num_moe_layers, batch_max_tokens, topk),
            dtype=self.dtype,
            device="cuda",
        )

        cpu_buffer_size = num_moe_layers * kv_cache_size * topk * dtype_bytes
        self.cpu_buffer = torch.zeros(
            (num_moe_layers, kv_cache_size, topk),
            dtype=self.dtype,
            device="cpu",
            pin_memory=True,
        )

        self.flush_streams = [torch.cuda.Stream() for _ in range(self.num_slots)]
        self.flush_events = [torch.cuda.Event() for _ in range(self.num_slots)]

        dtype_name = "int8" if self.dtype == torch.int8 else "int16"
        logger.info(
            f"RoutingCaptureManager initialized: {num_moe_layers} MoE layers, topk={topk}, "
            f"slots={self.num_slots}, GPU={gpu_buffer_size / 1024 / 1024:.2f}MB, "
            f"CPU={cpu_buffer_size / 1024 / 1024:.2f}MB, dtype={dtype_name}"
        )

    def capture(self, moe_layer_index: int, topk_ids: torch.Tensor, microbatch_index: int = 0) -> None:
        num_tokens = topk_ids.shape[0]
        self.gpu_buffer[microbatch_index, moe_layer_index, :num_tokens, :] = topk_ids.to(self.dtype)

    def flush_to_cpu_async(self, mem_indexes: torch.Tensor, microbatch_index: int) -> None:
        num_tokens = mem_indexes.shape[0]
        if num_tokens == 0:
            return

        slot = microbatch_index % self.num_slots
        stream = self.flush_streams[slot]
        event = self.flush_events[slot]

        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            cpu_indexes = mem_indexes.cpu()
            self.cpu_buffer[:, cpu_indexes, :] = self.gpu_buffer[slot, :, :num_tokens, :].cpu()
            event.record()

    def sync_events(self) -> None:
        """Synchronize all flush events. Call once before batch extraction."""
        for event in self.flush_events:
            event.synchronize()

    def extract_for_request(self, mem_indexes: torch.Tensor) -> np.ndarray:
        self.sync_events()
        return self.cpu_buffer[:, mem_indexes, :].numpy()

    def extract_for_request_no_sync(self, mem_indexes: torch.Tensor) -> np.ndarray:
        return self.cpu_buffer[:, mem_indexes, :].numpy()


g_routing_capture_manager: Optional[RoutingCaptureManager] = None


def create_routing_capture_manager(
    num_moe_layers: int,
    topk: int,
    num_experts: int,
    batch_max_tokens: int,
    kv_cache_size: int,
    enable_overlap: bool = False,
) -> None:
    global g_routing_capture_manager
    assert g_routing_capture_manager is None, "RoutingCaptureManager already exists"
    g_routing_capture_manager = RoutingCaptureManager(
        num_moe_layers=num_moe_layers,
        topk=topk,
        num_experts=num_experts,
        batch_max_tokens=batch_max_tokens,
        kv_cache_size=kv_cache_size,
        enable_overlap=enable_overlap,
    )


def init_routing_capture(model) -> None:
    if not getattr(model.args, "enable_return_routed_experts", False):
        return

    if get_current_rank_in_dp() != 0:
        logger.info("Skipping routing capture initialization on non-zero rank")
        return

    num_moe_layers = get_moe_layer_count()
    if num_moe_layers == 0:
        logger.warning(
            "enable_return_routed_experts is set but no MoE layers found. " "Routing capture will not be enabled."
        )
        return

    num_experts = model.config.get("n_routed_experts", model.config.get("num_experts", 0))
    topk = model.config.get("num_experts_per_tok", 0)
    assert num_experts > 0 and topk > 0
    enable_overlap = getattr(model.args, "enable_decode_microbatch_overlap", False)

    logger.info(
        f"Initializing routing capture: num_moe_layers={num_moe_layers}, "
        f"topk={topk}, num_experts={num_experts}, enable_overlap={enable_overlap}"
    )

    create_routing_capture_manager(
        num_moe_layers=num_moe_layers,
        topk=topk,
        num_experts=num_experts,
        batch_max_tokens=model.max_total_token_num,
        kv_cache_size=model.mem_manager.size + 1,
        enable_overlap=enable_overlap,
    )

    shared_config = get_shared_routing_config()
    shared_config.num_moe_layers = num_moe_layers
    shared_config.topk = topk
    logger.info(f"Shared routing config set: num_moe_layers={num_moe_layers}, topk={topk}")


def flush_routing_capture(mem_indexes: torch.Tensor, microbatch_index: int = 0) -> None:
    if g_routing_capture_manager is not None:
        g_routing_capture_manager.flush_to_cpu_async(mem_indexes, microbatch_index)
