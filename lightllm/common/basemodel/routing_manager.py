import atexit
import json
import os
import torch
import numpy as np
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple
from lightllm.common.basemodel.triton_kernel.routing_capture import scatter_routing_topk_to_cpu
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_rank_in_dp
from lightllm.utils.envs_utils import get_unique_server_name

logger = init_logger(__name__)


def routing_dtype_id_to_np(dtype_id: int):
    if dtype_id == 1:
        return np.uint8
    elif dtype_id == 2:
        return np.int16
    return np.int32


def _get_model_text_config(config: dict) -> dict:
    return config.get("text_config", config)


def _get_num_moe_layers_from_config(config: dict) -> int:
    num_layers = config.get("num_hidden_layers", config.get("n_layer", config.get("num_layers", 0)))
    num_experts = config.get("n_routed_experts", config.get("num_experts", config.get("num_local_experts", 0)))
    if num_layers <= 0 or not num_experts:
        return 0

    if "first_k_dense_replace" in config:
        first_k_dense_replace = config.get("first_k_dense_replace", 0)
        moe_layer_freq = config.get("moe_layer_freq", 1)
        return sum(
            1
            for layer_num in range(num_layers)
            if layer_num >= first_k_dense_replace and layer_num % moe_layer_freq == 0
        )

    if "mlp_only_layers" in config or "decoder_sparse_step" in config:
        mlp_only_layers = set(config.get("mlp_only_layers", []))
        decoder_sparse_step = config.get("decoder_sparse_step", 1)
        return sum(
            1
            for layer_num in range(num_layers)
            if layer_num not in mlp_only_layers and (layer_num + 1) % decoder_sparse_step == 0
        )

    if config.get("enable_moe_block", False):
        return num_layers

    return num_layers


def get_routing_config_from_model_dir(model_dir: str) -> Optional[Tuple[int, int, int]]:
    with open(os.path.join(model_dir, "config.json"), "r") as json_file:
        config = _get_model_text_config(json.load(json_file))

    num_moe_layers = _get_num_moe_layers_from_config(config)
    topk = config.get("num_experts_per_tok", config.get("top_k_experts", 0))
    num_experts = config.get("n_routed_experts", config.get("num_experts", config.get("num_local_experts", 0)))
    if num_moe_layers <= 0 or topk <= 0 or not num_experts:
        return None

    dtype_id = 1 if num_experts <= 256 else 2
    return num_moe_layers, topk, dtype_id


class RoutingCaptureManager:
    def __init__(
        self,
        num_moe_layers: int,
        topk: int,
        num_experts: int,
        kv_cache_size: int,
        max_capture_tokens: int,
        layer_num_to_moe_index: Optional[Dict[int, int]] = None,
    ):
        self.num_moe_layers = num_moe_layers
        self.topk = topk
        self.num_experts = num_experts
        self.kv_cache_size = kv_cache_size
        self.max_capture_tokens = max_capture_tokens
        self.layer_num_to_moe_index = layer_num_to_moe_index or {i: i for i in range(num_moe_layers)}

        self.dtype = torch.uint8 if num_experts <= 256 else torch.int16
        dtype_bytes = 1 if self.dtype == torch.uint8 else 2

        # Shape: (kv_cache_size, num_moe_layers, topk). Pinned CPU memory saves GPU memory
        # while allowing the Triton scatter kernel to write without a synchronous D2H copy.
        routing_buffer_size = num_moe_layers * kv_cache_size * topk * dtype_bytes
        self.routing_buffer = torch.zeros(
            (kv_cache_size, num_moe_layers, topk),
            dtype=self.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.routing_buffer_ptr = torch.tensor([self.routing_buffer.data_ptr()], dtype=torch.uint64, device="cuda")

        dtype_name = "uint8" if self.dtype == torch.uint8 else "int16"
        logger.info(
            f"RoutingCaptureManager initialized: {num_moe_layers} MoE layers, topk={topk}, "
            f"routing_buffer(cpu)={routing_buffer_size / 1024 / 1024:.2f}MB, "
            f"dtype={dtype_name}"
        )

    @property
    def np_dtype(self):
        return np.uint8 if self.dtype == torch.uint8 else np.int16

    @property
    def dtype_id(self) -> int:
        return 1 if self.dtype == torch.uint8 else 2

    def make_capture_callback_factory(self, mem_indexes: torch.Tensor):
        if not mem_indexes.is_cuda:
            mem_indexes = mem_indexes.cuda(non_blocking=True)

        def make_capture_callback(layer_num: int):
            routing_layer_index = self.layer_num_to_moe_index.get(layer_num)
            if routing_layer_index is None:
                return None

            def capture_callback(topk_ids: torch.Tensor) -> None:
                self.capture(routing_layer_index=routing_layer_index, topk_ids=topk_ids, mem_indexes=mem_indexes)

            return capture_callback

        return make_capture_callback

    def capture(self, routing_layer_index: int, topk_ids: torch.Tensor, mem_indexes: torch.Tensor) -> None:
        assert topk_ids.dim() == 2
        assert topk_ids.shape[1] == self.topk
        assert mem_indexes.shape[0] >= topk_ids.shape[0]
        scatter_routing_topk_to_cpu(
            topk_ids=topk_ids,
            mem_indexes=mem_indexes,
            routing_buffer_ptr=self.routing_buffer_ptr,
            moe_layer_index=routing_layer_index,
            num_moe_layers=self.num_moe_layers,
            topk=self.topk,
            dtype_id=self.dtype_id,
        )

    def extract_routing_data(self, mem_indexes: torch.Tensor) -> np.ndarray:
        torch.cuda.synchronize()
        cpu_indexes = mem_indexes.cpu() if mem_indexes.is_cuda else mem_indexes
        return self.routing_buffer[cpu_indexes, :, :].numpy()


g_routing_capture_manager: Optional[RoutingCaptureManager] = None


def create_routing_capture_manager(
    num_moe_layers: int,
    topk: int,
    num_experts: int,
    kv_cache_size: int,
    max_capture_tokens: int,
    layer_num_to_moe_index: Optional[Dict[int, int]] = None,
) -> None:
    global g_routing_capture_manager
    assert g_routing_capture_manager is None, "RoutingCaptureManager already exists"
    g_routing_capture_manager = RoutingCaptureManager(
        num_moe_layers=num_moe_layers,
        topk=topk,
        num_experts=num_experts,
        kv_cache_size=kv_cache_size,
        max_capture_tokens=max_capture_tokens,
        layer_num_to_moe_index=layer_num_to_moe_index,
    )


def _get_moe_layer_nums(model) -> List[int]:
    moe_layer_nums = []
    for layer_weight in getattr(model, "trans_layers_weight", []):
        is_moe = getattr(layer_weight, "is_moe", None)
        if is_moe is None:
            is_moe = hasattr(layer_weight, "experts")
        if is_moe:
            moe_layer_nums.append(layer_weight.layer_num_)
    return moe_layer_nums


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


def init_routing_capture(model, num_moe_layers: Optional[int] = None) -> None:
    moe_layer_nums = _get_moe_layer_nums(model)
    if num_moe_layers is None:
        num_moe_layers = len(moe_layer_nums)
    elif moe_layer_nums:
        assert num_moe_layers == len(moe_layer_nums)
    else:
        moe_layer_nums = list(range(num_moe_layers))
    layer_num_to_moe_index = {layer_num: moe_index for moe_index, layer_num in enumerate(moe_layer_nums)}

    dp_rank = get_current_rank_in_dp()
    logger.info(
        f"init_routing_capture called: num_moe_layers={num_moe_layers}, "
        f"moe_layer_nums={moe_layer_nums}, dp_rank={dp_rank}"
    )
    if dp_rank != 0:
        logger.info(f"Skipping routing capture initialization on dp_rank={dp_rank}")
        return

    if num_moe_layers == 0:
        logger.warning(
            "enable_return_routed_experts is set but no MoE layers found. Routing capture will not be enabled."
        )
        return

    num_experts = model.config.get(
        "n_routed_experts",
        model.config.get("num_experts", model.config.get("num_local_experts", 0)),
    )
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
        layer_num_to_moe_index=layer_num_to_moe_index,
    )

    logger.info(
        f"Routing capture config set: num_moe_layers={num_moe_layers}, topk={topk}, "
        f"dtype_id={g_routing_capture_manager.dtype_id}, max_tokens={args.max_req_total_len}"
    )
    atexit.register(cleanup_routing_shm_pool)
