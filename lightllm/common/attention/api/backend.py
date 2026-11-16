from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, TYPE_CHECKING, TypeVar

import torch

from lightllm.platform import DeviceCaps, Platform

from .batch import AttentionBatch, ContextCapacity
from .environment import BackendEnv
from .execution import ServingConstraints, WorkspaceBudget
from .spec import AttentionFamily, AttentionPhase, CacheEncoding, LayerSpec
from .workspace import AttentionWorkspace


if TYPE_CHECKING:
    from lightllm.common.attention.session import AttentionSession


@dataclass(slots=True)
class AttentionContext:
    capacity: ContextCapacity
    capture: bool
    workspace: AttentionWorkspace
    batch: AttentionBatch | None = None
    current_max_kv_seq_len: int = 0
    backend_identity: AttentionBackend | None = None


Context = TypeVar("Context", bound=AttentionContext)
Inputs = TypeVar("Inputs")
Output = TypeVar("Output")
Resources = TypeVar("Resources")


class UnsupportedAttention(ValueError):
    pass


@dataclass(frozen=True)
class LayerAttention(Generic[Resources, Context, Inputs, Output]):
    group_id: int
    logical_layer_id: int
    resources: Resources
    backend: AttentionBackend[Context, Inputs, Output]
    output_spec: Callable[[LayerSpec, Inputs], tuple]

    def context(self, session: AttentionSession) -> Context:
        return session.context(self)

    def forward(self, session: AttentionSession, inputs: Inputs, *, out: torch.Tensor | None = None) -> Output:
        return session.forward(self, inputs, out=out)

    def sparse_index_metadata(self, session: AttentionSession):
        return self.backend.sparse_index_metadata(session.context(self))


class AttentionBackend(ABC, Generic[Context, Inputs, Output]):
    zero_initialize_cache = False
    requires_cpu_cache_sync = False
    capture_field_policies = {}

    family: AttentionFamily
    provider: str
    phase: AttentionPhase
    cache_encodings = frozenset({CacheEncoding.PLAIN})
    state_mode = "native"
    variant = ""
    supported_features = frozenset()
    supported_platforms = frozenset({Platform.CUDA})
    supports_graph = True
    supported_graph_kinds = frozenset({"decode", "prefill_segments"})
    activation_dtypes = frozenset({torch.float16, torch.bfloat16, torch.float32})
    dependency = "triton"
    min_device_capability = None
    supports_expert_parallel = True
    max_head_size = None

    def __init__(self, env: BackendEnv, spec: LayerSpec):
        self.runtime = env.runtime
        self.spec = spec
        self.draft_step = env.draft_step if spec.use_mtp else 0
        self.hold_req_id = env.hold_req_id
        self.alloc_tensor = env.alloc_tensor

    @classmethod
    def check_environment(cls, caps: DeviceCaps) -> None:
        from lightllm.common.attention._environment import check_attention_environment

        check_attention_environment(
            caps,
            provider=cls.provider,
            supported_platforms=cls.supported_platforms,
            min_device_capability=cls.min_device_capability,
            dependency=cls.dependency,
        )

    @classmethod
    def check_support(cls, spec: LayerSpec, caps=None, constraints=None) -> None:
        if (spec.family, spec.phase, spec.variant) != (cls.family, cls.phase, cls.variant):
            raise UnsupportedAttention(f"{cls.__name__} does not implement {spec.family.value} {spec.phase.value}")
        if spec.cache.encoding not in cls.cache_encodings or spec.cache.state_mode != cls.state_mode:
            raise UnsupportedAttention(
                f"cache layout {spec.cache.encoding.value}/{spec.cache.state_mode} is unsupported"
            )
        if spec.dtype not in cls.activation_dtypes:
            raise UnsupportedAttention(f"activation dtype {spec.dtype} is unsupported")
        missing = spec.features - cls.supported_features
        if missing:
            raise UnsupportedAttention("unsupported attention features: " + ", ".join(sorted(f.value for f in missing)))
        if cls.max_head_size is not None and spec.head_size > cls.max_head_size:
            raise UnsupportedAttention(f"head size exceeds {cls.max_head_size}")
        if constraints is not None:
            if constraints.expert_parallel and not cls.supports_expert_parallel:
                raise UnsupportedAttention("this backend is incompatible with expert-parallel MoE execution")
            if spec.phase in constraints.graph_phases and (
                not cls.supports_graph
                or (caps is not None and not caps.supports_graph)
                or ("decode" if spec.phase is AttentionPhase.DECODE else "prefill_segments")
                not in cls.supported_graph_kinds
            ):
                raise UnsupportedAttention(f"{spec.phase.value} Graph is unsupported")

    @property
    def kernel_page_size(self) -> int:
        return self.spec.cache.page_size

    @abstractmethod
    def allocate_context(self, capacity: ContextCapacity, *, capture: bool, workspace: AttentionWorkspace) -> Context:
        pass

    def plan_host(self, context: Context) -> None:
        pass

    def build_device_metadata(self, context: Context) -> None:
        pass

    @abstractmethod
    def forward(
        self, layer: LayerAttention[Any, Context, Inputs, Output], context: Context, inputs: Inputs, *, out=None
    ):
        pass

    def resource_requirements(self, capacity: ContextCapacity, *, capture: bool):
        return WorkspaceBudget()

    def capacity_memory_bytes(self, capacity: ContextCapacity, *, cache_tokens: int) -> int:
        return 0

    def sparse_index_metadata(self, context):
        raise UnsupportedAttention("this backend does not provide sparse index metadata")
