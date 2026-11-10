from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import torch

from .batch import AttentionBatch, PlanCapacity
from .capability import CapabilityResult
from .inputs import VarlenAttentionInputs
from .spec import LayerSpec

Plan = TypeVar("Plan")
Inputs = TypeVar("Inputs")
Output = TypeVar("Output")
Params = TypeVar("Params")


@dataclass(frozen=True)
class LayerBinding(Generic[Params]):
    """一个逻辑层在一个 phase 下的固定绑定，权重和存储准备完成后构造。

    group_id 用于取得当前批次的共享 plan；logical_layer_id 是模型层号，
    不一定等于物理存储下标。params 借用该层的存储视图，存储重分配后须重新绑定。
    """

    group_id: int
    logical_layer_id: int
    spec: LayerSpec
    params: Params
    forward: Callable
    alloc_tensor: Callable

    def __call__(self, plan, inputs, *, out=None):
        return self.forward(self, plan, inputs, out=out)


class AttentionBackend(ABC, Generic[Plan, Inputs, Output]):
    """同组层共用的执行实现；每批次变化的 metadata 和 wrapper 放在 plan 中。

    Plan 的具体类型由 backend 定义，实例由 InferStateInfo 保存，不能跨 backend
    或并发 microbatch 共用。调用顺序为 allocate_plan -> prepare -> update，
    随后同组各层用各自的 LayerBinding / Inputs 调用 forward。
    """

    @abstractmethod
    def allocate_plan(self, capacity: PlanCapacity, *, capture: bool, workspace) -> Plan:
        """创建归属此 backend 的 plan，workspace 由模型按 group / microbatch slot 提供。

        capture=True 的 plan 随 decode CUDA Graph 保存，在后续 replay 中复用。
        """
        pass

    @abstractmethod
    def prepare(self, plan: Plan, batch: AttentionBatch) -> None:
        """在图外绑定批次输入并完成 provider planning，这里允许 CPU 侧准备工作。

        decode replay 时仍会调用；图中引用的 buffer 必须保持地址，只更新内容。
        """
        pass

    @abstractmethod
    def update(self, plan: Plan) -> None:
        """在各层 forward 前生成 GPU 侧派生 metadata。

        decode Graph 会捕获这段 GPU 工作；不能依赖 replay 时重新执行 Python，
        也不能把当前批次的 GPU 值读回 CPU 来决定执行路径。
        """
        pass

    @abstractmethod
    def forward(self, layer: LayerBinding, plan: Plan, inputs: Inputs, *, out: torch.Tensor | None = None) -> Output:
        """使用已准备的 plan 执行一层，允许写入 params 借用的 KV / Conv / SSM 存储。

        out 指定 attention 结果的目标缓冲区；返回 tuple 时只对应首项，
        其余项保留模型所需的附加输出，例如 GDN 的 gate 输入。
        """
        pass

    def resource_requirements(self, capacity: PlanCapacity, *, capture: bool):
        """返回分配前可计算的资源上界，供共享容量规划和捕获前预留使用。"""
        from .execution import BackendResources

        return BackendResources()

    def bind_graph(self, plan: Plan):
        """为此捕获实例创建可选的 GraphBinding；普通原生 Graph 无需额外绑定。"""
        return None


class VarlenAttentionBackend(ABC):
    """无持久 KV 的双向 dense attention，供视觉和音频编码器使用。

    Q/K/V 与变长边界由调用方直接传入，不需要文本推理的请求表或 cache plan。
    实现不保留批次 tensor；调用方持有 out，返回值必须引用同一块输出存储。
    SDPA / xFormers 需要在 CPU 读取序列边界；只有声明 supports_graph 的实现可整段捕获。
    """

    supports_graph = False

    @classmethod
    @abstractmethod
    def check_environment(cls, caps) -> CapabilityResult:
        pass

    @classmethod
    @abstractmethod
    def check_support(cls, inputs: VarlenAttentionInputs, caps=None) -> CapabilityResult:
        """初始化或独立探针使用；forward 依赖输入契约，避免逐层重复能力检查。"""
        pass

    @abstractmethod
    def forward(self, inputs: VarlenAttentionInputs, *, out: torch.Tensor) -> torch.Tensor:
        pass
