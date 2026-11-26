"""定义各推理阶段的 attention 后端、模型资源和单步输入。

后端持有静态配置和可复用的私有缓冲区。配置相同的层共享实例，
每个微批次显式保存 create_att_state 返回的状态。
不同配置各自持有准备结果和用于规划的私有缓冲区。
"""
from collections.abc import Callable

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from lightllm.utils.dist_utils import get_current_device_id

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo

AttConfig = TypeVar(name="AttConfig")
AttState = TypeVar(name="AttState")
AttInputs = TypeVar(name="AttInputs")


@dataclass(frozen=True)
class AttRuntimeConfig:
    """创建后端时读取一次的模型配置。"""

    logical_layer_ids: tuple[int, ...] = field(default=(), kw_only=True)
    # Decode 每个请求的最大查询行数，至少为 1；动态验证的实际分组在 att_state 中生成。
    decode_rows_per_request: int = field(default=1, kw_only=True)
    # 主模型是否压缩投机验证行，草稿模型使用固定布局。
    dynamic_decode_rows: bool = field(default=False, kw_only=True)
    # Decode CUDA Graph 的 KV 长度容量；不运行该类 Graph 的后端可省略。
    cuda_graph_max_kv_seq_len: int | None = field(default=None, kw_only=True)

    @classmethod
    def from_model(cls, model: "TpPartBaseModel") -> "AttRuntimeConfig":
        """读取模型固定的解码布局和 Graph 容量，单步派生张量由后端准备。"""
        mtp_manager = getattr(model, "mtp_manager", None)
        rows = (
            mtp_manager.get_decode_draft_step(is_draft_model=model.is_mtp_draft_model) + 1
            if mtp_manager is not None
            else 1
        )
        return cls(
            decode_rows_per_request=rows,
            dynamic_decode_rows=rows > 1
            and not model.is_mtp_draft_model
            and getattr(model.args, "mtp_dynamic_verify", False),
            cuda_graph_max_kv_seq_len=getattr(model, "graph_max_len_in_batch", None),
        )


class _AttBackend(ABC, Generic[AttConfig, AttState, AttInputs]):
    """保存公共配置和缓冲区，各阶段接口负责执行计算。"""

    # ========================================================================
    # 一、构造与运行时配置
    # ========================================================================

    def __init__(
        self, att_config: AttConfig, *, role: str = "main", runtime_config: AttRuntimeConfig = AttRuntimeConfig()
    ) -> None:
        self.att_config = att_config
        self.role = role
        self.runtime_config = runtime_config

    @classmethod
    def create_runtime_config(cls, model: "TpPartBaseModel") -> AttRuntimeConfig:
        """在模型初始化时读取该实现所需的运行时配置。"""
        return AttRuntimeConfig.from_model(model=model)

    # ========================================================================
    # 二、注册信息与能力查询
    # ========================================================================

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """返回实现名，供后端选择和注册使用。"""
        pass

    @property
    def logical_layer_ids(self) -> tuple[int, ...]:
        """返回该实例服务的逻辑层号。"""
        return self.runtime_config.logical_layer_ids

    def get_registration_name(self) -> str:
        """实现名已包含阶段，按角色和最小逻辑层号生成注册名称。"""
        return f"{self.role}_{self.get_name()}_{min(self.logical_layer_ids)}_backend"

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        """声明计算是否需要与 GPU 上的 CPU cache 拷贝同步。"""
        return False

    # ========================================================================
    # 三、临时工作区
    # ========================================================================

    @staticmethod
    def get_workspace(name: str, role: str, numel: int, dtype: torch.dtype = torch.int8) -> torch.Tensor:
        """获取当前 CUDA 设备上含 numel 个元素的一维临时工作区。

        role 使用模型的 att_state_role。同设备、同名、同 dtype 的请求共用一组缓存。
        优先按 (role, numel) 返回已缓存的张量。首次请求若能找到其他角色容量足够的工作区，
        就取其前缀视图，否则分配新缓冲区。两种结果都在返回前缓存，以保持 CUDA Graph
        使用的地址稳定。name 必须非空，numel 是元素数，必须大于零。

        复用的视图共享底层存储，内容可被下一次计算覆盖。调用方须保证使用
        同一存储的计算按顺序执行，例如在同一 CUDA stream 上排队或用 event
        同步不同 stream。需要并发读写或保留内容的调用方应使用独立名称。
        缓存锁只保护 CPU 侧的查找和分配。
        """
        if not name:
            raise ValueError("workspace name must not be empty")
        if numel <= 0:
            raise ValueError(f"workspace numel must be positive, got {numel}")

        device_id = get_current_device_id()
        key = (role, numel)
        attrs = _AttBackend.get_workspace.__dict__
        if "_cache" not in attrs:
            # 并发调用共用同一把锁和同一份缓存。
            attrs.setdefault("_cache", (Lock(), {}))
        lock, buffers = attrs["_cache"]
        with lock:
            pool = buffers.setdefault((device_id, name, dtype), {})
            if key in pool:
                return pool[key]

            # 复用其他角色的存储时，先把视图登记到当前角色，以便下次直接命中。
            for (other_role, capacity), candidate in pool.items():
                if other_role != role and capacity >= numel:
                    buffer = candidate if capacity == numel else candidate[:numel]
                    pool[key] = buffer
                    return buffer

            buffer = torch.empty(size=(numel,), dtype=dtype, device=device_id)
            pool[key] = buffer
            return buffer

    # ========================================================================
    # 四、当前微批次的状态准备
    # ========================================================================

    @abstractmethod
    def create_att_state(self, infer_state: "InferStateInfo") -> AttState:
        """在第一层执行前为该实例准备状态，包含 CUDA Graph 重放步骤。

        动态张量由本步状态持有，CUDA Graph replay 前复制到捕获状态。
        只读常量可共享；第三方 wrapper 和调度资源由具体后端管理。
        根据 Graph 容量确定 kernel 启动规模。
        """


class PrefillAttBackend(_AttBackend[AttConfig, AttState, AttInputs]):
    """准备并执行 prefill attention。"""

    @abstractmethod
    def prefill(
        self,
        att_state: AttState,
        layer_inputs: AttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """写入并返回 out。线性 attention 的 prefill 实现也接受 out=None。"""


class DecodeAttBackend(_AttBackend[AttConfig, AttState, AttInputs]):
    """准备并执行 decode attention。"""

    @abstractmethod
    def decode(
        self,
        att_state: AttState,
        layer_inputs: AttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """写入并返回调用方传入的 out 张量。"""


AttentionBackend = PrefillAttBackend | DecodeAttBackend
