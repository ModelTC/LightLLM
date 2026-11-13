from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar

import torch

from lightllm.platform import DeviceCaps

from .batch import AttentionBatch, ContextCapacity
from .execution import BackendResources, ExecutionRequirements
from .spec import LayerSpec
from .workspace import AttentionWorkspace


@dataclass(slots=True)
class AttentionExecutionContext:
    capacity: ContextCapacity
    capture: bool
    workspace: AttentionWorkspace
    batch: AttentionBatch | None = None
    current_max_kv_seq_len: int = 0
    backend_identity: AttentionBackend | None = None


Context = TypeVar("Context", bound=AttentionExecutionContext)
Inputs = TypeVar("Inputs")
Output = TypeVar("Output")
Params = TypeVar("Params")


class UnsupportedAttention(ValueError):
    """当前环境或输入不满足此 attention 实现的要求。"""


class AttentionExecution(Protocol):
    def attention_context(self, binding: "LayerAttentionBinding[Params, Context, Inputs, Output]") -> Context:
        ...


@dataclass(frozen=True)
class LayerAttentionBinding(Generic[Params, Context, Inputs, Output]):
    """一个逻辑层在 prefill 或 decode 阶段的 attention 执行入口。

    权重和存储准备完成后构造；backend 是同组层共享的实现，params 借用本层的
    权重和存储视图，存储重分配后须重新构造。group_id 用于取得当前批次的 context；
    logical_layer_id 是模型层号，不一定等于物理存储下标。
    forward 从当前 execution 查找 context，不保存批次 context；prepare/update 由 Runtime 管理。
    """

    group_id: int
    logical_layer_id: int
    spec: LayerSpec
    params: Params
    backend: "AttentionBackend[Context, Inputs, Output]"
    alloc_tensor: Callable

    def context(self, execution: AttentionExecution) -> Context:
        return execution.attention_context(self)

    def forward(self, execution: AttentionExecution, inputs: Inputs, *, out: torch.Tensor | None = None) -> Output:
        context = self.context(execution)
        return self.backend.forward(self, context, inputs, out=out)


class AttentionBackend(ABC, Generic[Context, Inputs, Output]):
    """同组层共用的执行实现；每批次变化的 metadata 和 wrapper 放在 context 中。

    Context 的具体类型由 backend 定义，实例由 InferStateInfo.attention_runtime 保存，不能跨 backend
    或并发 microbatch 共用。调用顺序为 allocate_context -> prepare -> update，
    随后同组各层用各自的 LayerAttentionBinding / Inputs 调用 forward。
    """

    @classmethod
    @abstractmethod
    def check_environment(cls, caps: DeviceCaps) -> None:
        """Reject unavailable platforms or provider dependencies with UnsupportedAttention."""
        pass

    @classmethod
    @abstractmethod
    def check_support(
        cls, spec: LayerSpec, caps: DeviceCaps | None = None, execution: ExecutionRequirements | None = None
    ) -> None:
        """Reject unsupported static semantics or execution requirements."""
        pass

    @classmethod
    @abstractmethod
    def create(cls, model, spec: LayerSpec) -> AttentionBackend[Context, Inputs, Output]:
        """Construct a backend for this spec; model supplies runtime and layer resources."""
        pass

    @abstractmethod
    def allocate_context(self, capacity: ContextCapacity, *, capture: bool, workspace: AttentionWorkspace) -> Context:
        """创建归属此 backend 的 context，workspace 由模型按 group / microbatch slot 提供。

        capture=True 的 context 随 decode CUDA Graph 保存，在后续 replay 中复用。
        """
        pass

    @abstractmethod
    def prepare(self, context: Context, batch: AttentionBatch) -> None:
        """在图外绑定批次输入并完成 provider planning，这里允许 CPU 侧准备工作。

        decode replay 时仍会调用；图中引用的 buffer 必须保持地址，只更新内容。
        """
        pass

    @abstractmethod
    def update(self, context: Context) -> None:
        """在各层 forward 前生成 GPU 侧派生 metadata。

        decode Graph 会捕获这段 GPU 工作；不能依赖 replay 时重新执行 Python，
        也不能把当前批次的 GPU 值读回 CPU 来决定执行路径。
        """
        pass

    @abstractmethod
    def forward(
        self,
        layer: LayerAttentionBinding[Any, Context, Inputs, Output],
        context: Context,
        inputs: Inputs,
        *,
        out: torch.Tensor | None = None,
    ) -> Output:
        """使用已准备的 context 执行一层，允许写入 params 借用的 KV / Conv / SSM 存储。

        out 是调用方拥有的目标缓冲区，dtype、device 必须与输出一致。
        provider 直接写入时可返回其中有效部分的 view（例如去掉 head padding）；
        返回独立 tensor 时元素数必须一致，由基类按 out 的形状复制。
        provider 可直接写入，或返回临时结果由基类复制；不承诺零分配。返回 tuple 时只对应首项，
        其余项保留模型所需的附加输出，例如 GDN 的 gate 输入。
        """
        pass

    def resource_requirements(self, capacity: ContextCapacity, *, capture: bool):
        """返回分配前可计算的资源上界，供共享容量规划和捕获前预留使用。"""
        return BackendResources()

    def bind_graph(self, context: Context):
        """为此捕获实例创建可选的 GraphBinding；普通原生 Graph 无需额外绑定。"""
        return None
