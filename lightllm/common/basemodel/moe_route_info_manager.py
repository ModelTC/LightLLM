"""MoE routed-experts 捕获与配置管理。

本模块为 ``--enable_return_routed_experts`` 提供端到端支持：在 MoE 推理过程中记录
每个 token、每一层选中的 top-k expert id，并在请求结束时随响应返回
``routed_experts`` 元数据。

背景与目标
----------
MoE 模型每个 token 只会激活少量专家。训练、评测、路由分析等场景常需要知道
「某个 token 在各 MoE 层实际走到了哪些 expert」。LightLLM 在开启
``enable_return_routed_experts`` 后，由本模块在推理路径上零拷贝地收集这些
topk ids，最终写入请求的 final token metadata shm，供 HTTP 进程读取并编码进
API 返回。

非 MoE 模型不应开启该功能；配置解析假定模型具备合法的 MoE 字段
（专家数、topk、MoE 层分布等）。

核心职责
--------
1. **路由配置（phase-1）**
   从 ``config.json`` 解析：
   - ``num_moe_layers`` / ``topk`` / ``dtype_id``
   - ``layer_index_to_moe_index``：transformer layer index → 稠密 MoE 槽位 index
     （部分模型前若干层是 dense MLP，或按 ``moe_layer_freq`` /
     ``decoder_sparse_step`` 稀疏分布 MoE 层）。

2. **捕获缓冲（phase-2，可选）**
   在 infer 进程按 KV cache 槽位分配 pinned CPU buffer：
   ``routing_buffer[kv_slot, moe_layer_slot, topk]``。
   Triton kernel 将 GPU 上的 topk ids scatter 写入该 buffer，避免同步 D2H。

3. **捕获回调**
   fused MoE 前向在选出 topk ids 后调用 ``moe_capture_callback``，按当前
   ``mem_indexes``（token 对应的 KV 槽位）写入对应层的路由信息。

4. **导出**
   请求结束时按该请求占用的 mem indexes ``extract`` 出
   ``(num_tokens, num_moe_layers, topk)`` 数组，写入 final token metadata，
   HTTP 侧再编码为响应中的 ``routed_experts``。

进程与初始化
------------
- 进程内单例：``MoeRouteInfoManager.get_instance()``。
  仅在 ``enable_return_routed_experts`` 时创建，且只做 phase-1 配置初始化。
- **HTTP 进程**：只需配置元数据（层数 / topk / dtype），用于解析 shm 中的
  routed experts 布局，不分配 capture buffer。
- **Infer 进程（通常 dp_rank==0）**：在 phase-1 之后调用
  ``init_capture_buffer(kv_cache_size)``，再参与 capture / extract。

数据流简图::

    MoE forward (topk_ids)
           │
           ▼
    moe_capture_callback  ──scatter──►  routing_buffer[mem_index, moe_slot, :]
           │
           ▼  (request finished)
        extract(mem_indexes)
           │
           ▼
    final_token_metadata shm  ──HTTP──►  response["routed_experts"]
"""

import json
import os
import torch
import numpy as np
from typing import ClassVar, Dict, Optional, Tuple
from lightllm.common.basemodel.triton_kernel.routing_capture import scatter_routing_topk_to_cpu
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MoeRouteInfoManager:
    """管理 MoE topk expert id 的配置、捕获缓冲与导出。

    详见模块文档字符串。
    """

    _instance: ClassVar[Optional["MoeRouteInfoManager"]] = None

    @classmethod
    def get_instance(cls) -> Optional["MoeRouteInfoManager"]:
        """Return the process singleton with phase-1 (route config) init only.

        Capture buffer is optional and must be allocated separately via
        ``init_capture_buffer`` when needed (infer process).
        """
        if cls._instance is not None:
            return cls._instance

        from lightllm.utils.envs_utils import get_env_start_args

        args = get_env_start_args()
        if not args.enable_return_routed_experts:
            return None

        num_moe_layers, topk, dtype_id, layer_index_to_moe_index = cls.get_route_config_from_model_dir(args.model_dir)
        cls._instance = cls(
            num_moe_layers=num_moe_layers,
            topk=topk,
            dtype_id=dtype_id,
            layer_index_to_moe_index=layer_index_to_moe_index,
        )
        return cls._instance

    @staticmethod
    def _get_layer_index_to_moe_index_from_config(config: dict) -> Dict[int, int]:
        """Build layer_index -> dense moe-slot index from model config."""
        num_layers = config.get("num_hidden_layers", config.get("n_layer", config.get("num_layers", 0)))
        num_experts = config.get("n_routed_experts", config.get("num_experts", config.get("num_local_experts", 0)))
        assert num_layers > 0 and num_experts > 0

        if "first_k_dense_replace" in config:
            first_k_dense_replace = config.get("first_k_dense_replace", 0)
            moe_layer_freq = config.get("moe_layer_freq", 1)
            moe_layer_indexes = [
                layer_index
                for layer_index in range(num_layers)
                if layer_index >= first_k_dense_replace and layer_index % moe_layer_freq == 0
            ]
        elif "mlp_only_layers" in config or "decoder_sparse_step" in config:
            mlp_only_layers = set(config.get("mlp_only_layers", []))
            decoder_sparse_step = config.get("decoder_sparse_step", 1)
            moe_layer_indexes = [
                layer_index
                for layer_index in range(num_layers)
                if layer_index not in mlp_only_layers and (layer_index + 1) % decoder_sparse_step == 0
            ]
        elif config.get("enable_moe_block", False):
            moe_layer_indexes = list(range(num_layers))
        else:
            moe_layer_indexes = list(range(num_layers))

        assert len(moe_layer_indexes) > 0
        return {layer_index: moe_index for moe_index, layer_index in enumerate(moe_layer_indexes)}

    @staticmethod
    def get_route_config_from_model_dir(model_dir: str) -> Tuple[int, int, int, Dict[int, int]]:
        """Return (num_moe_layers, topk, dtype_id, layer_index_to_moe_index) from model config.

        Caller must only use this when --enable_return_routed_experts is set on a MoE model.
        """
        with open(os.path.join(model_dir, "config.json"), "r") as json_file:
            config = json.load(json_file)
            config = config.get("text_config", config)

        layer_index_to_moe_index = MoeRouteInfoManager._get_layer_index_to_moe_index_from_config(config)
        num_moe_layers = len(layer_index_to_moe_index)
        topk = config.get("num_experts_per_tok", config.get("top_k_experts", 0))
        num_experts = config.get("n_routed_experts", config.get("num_experts", config.get("num_local_experts", 0)))
        assert topk > 0 and num_experts > 0

        dtype_id = 1 if num_experts <= 256 else 2
        return num_moe_layers, topk, dtype_id, layer_index_to_moe_index

    def __init__(
        self,
        num_moe_layers: int,
        topk: int,
        dtype_id: int,
        layer_index_to_moe_index: Optional[Dict[int, int]] = None,
    ):
        """Phase-1 init: route config metadata only. Call init_capture_buffer() when capture is needed."""
        self.num_moe_layers = num_moe_layers
        self.topk = topk
        self.dtype_id = dtype_id
        self.layer_index_to_moe_index = layer_index_to_moe_index or {i: i for i in range(num_moe_layers)}

        self.kv_cache_size: Optional[int] = None
        self.routing_buffer: Optional[torch.Tensor] = None
        self.routing_buffer_ptr: Optional[torch.Tensor] = None

        logger.info(
            f"MoeRouteInfoManager created: num_moe_layers={num_moe_layers}, topk={topk}, " f"dtype_id={dtype_id}"
        )

    def get_np_dtype(self):
        if self.dtype_id == 1:
            return np.uint8
        elif self.dtype_id == 2:
            return np.int16
        return np.int32

    def get_torch_dtype(self):
        if self.dtype_id == 1:
            return torch.uint8
        elif self.dtype_id == 2:
            return torch.int16
        return torch.int32

    def init_capture_buffer(self, kv_cache_size: int) -> None:
        """Phase-2 init: allocate pinned CPU routing buffer for capture/extract."""
        if self.routing_buffer is not None:
            return

        torch_dtype = self.get_torch_dtype()
        dtype_bytes = torch_dtype.itemsize
        # Shape: (kv_cache_size, num_moe_layers, topk). Pinned CPU memory saves GPU memory
        # while allowing the Triton scatter kernel to write without a synchronous D2H copy.
        buffer_bytes = self.num_moe_layers * kv_cache_size * self.topk * dtype_bytes
        self.kv_cache_size = kv_cache_size
        self.routing_buffer = torch.zeros(
            (kv_cache_size, self.num_moe_layers, self.topk),
            dtype=torch_dtype,
            device="cpu",
            pin_memory=True,
        )
        self.routing_buffer_ptr = torch.tensor([self.routing_buffer.data_ptr()], dtype=torch.uint64, device="cuda")

        logger.info(
            f"MoeRouteInfoManager capture buffer ready: kv_cache_size={kv_cache_size}, "
            f"routing_buffer(cpu)={buffer_bytes / 1024 / 1024:.2f}MB, dtype={torch_dtype}"
        )

    def get_moe_capture_callback(self, layer_index: int, mem_indexes: torch.Tensor):
        """Return a callback that captures MoE topk expert ids into the routing buffer."""
        assert self.routing_buffer is not None, "call init_capture_buffer() before capture"
        moe_layer_index = self.layer_index_to_moe_index.get(layer_index)
        if moe_layer_index is None:
            return None
        if not mem_indexes.is_cuda:
            mem_indexes = mem_indexes.cuda(non_blocking=True)

        # Captures MoE topk ids for this layer and scatters them into the pinned CPU buffer.
        def moe_capture_callback(topk_ids: torch.Tensor) -> None:
            self.capture(moe_layer_index=moe_layer_index, topk_ids=topk_ids, mem_indexes=mem_indexes)

        return moe_capture_callback

    def capture(self, moe_layer_index: int, topk_ids: torch.Tensor, mem_indexes: torch.Tensor) -> None:
        assert self.routing_buffer_ptr is not None, "call init_capture_buffer() before capture"
        assert topk_ids.dim() == 2
        assert topk_ids.shape[1] == self.topk
        assert mem_indexes.shape[0] >= topk_ids.shape[0]
        scatter_routing_topk_to_cpu(
            topk_ids=topk_ids,
            mem_indexes=mem_indexes,
            routing_buffer_ptr=self.routing_buffer_ptr,
            moe_layer_index=moe_layer_index,
            num_moe_layers=self.num_moe_layers,
            topk=self.topk,
            dtype_id=self.dtype_id,
        )

    def extract(self, mem_indexes: torch.Tensor) -> np.ndarray:
        assert self.routing_buffer is not None, "call init_capture_buffer() before extract"
        cpu_indexes = mem_indexes.cpu() if mem_indexes.is_cuda else mem_indexes
        return self.routing_buffer[cpu_indexes, :, :].numpy()


def get_moe_capture_callback(infer_state, layer_index: int):
    """Return a callback that captures MoE topk expert ids, or None if capture is disabled."""
    mgr = MoeRouteInfoManager.get_instance()
    if mgr is None or mgr.routing_buffer is None:
        return None
    return mgr.get_moe_capture_callback(layer_index=layer_index, mem_indexes=infer_state.mem_index)
