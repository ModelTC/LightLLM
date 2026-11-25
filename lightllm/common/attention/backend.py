"""Attention backends and the inputs to build and forward.

Each type is named for how long it lives:

- LayerConfig: one layer's static attention configuration, fixed when the model loads. An
  AttentionBackend instance holds this config.
- InferStateInfo: request layout and KV cache indices for one forward step.
- AttState: the attention state shared by compatible layers in one microbatch. Backends that
  prepare page tables, recurrent indices or wrappers keep them on their state.
- LayerInputs: one layer call's tensors.

A model passes AttentionBuildResources to create_build_context on first use. Layers whose
backend class and build_context_key match share a context and one state per microbatch.

Attention states are built every step, including CUDA graph replay steps. The caller writes the KV cache
before attention and passes the attention state explicitly rather than through a global forward context.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar

import torch

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo

LayerConfig = TypeVar("LayerConfig")
AttState = TypeVar("AttState")
LayerInputs = TypeVar("LayerInputs")


class TensorAllocator(Protocol):
    """Allocate contiguous scratch with the requested shape, dtype and device.

    Calls pass a tuple/list shape positionally and dtype/device by keyword,
    as supported by torch.empty and the serving allocator. Returned storage
    must not overlap live operands, outputs or other scratch. The caller's
    allocator manages reuse and CUDA graph lifetimes; the backend retains none.
    """

    def __call__(
        self, shape: tuple[int, ...] | list[int], /, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class AttentionBuildResources:
    """Model-wide capacities and workspace access used to initialize a backend's storage."""

    # CUDA device index of the step's tensors.
    device: int
    # Cache slots per page of the KV cache; slot s lies on page s // page_size.
    page_size: int
    # The longest request in tokens, including any speculative-decoding headroom.
    max_seq_len: int
    # Decode rows of the largest CUDA graph step.
    max_cuda_graph_batch_size: int
    # Decode rows any step may have (running requests times rows per request, padded as the
    # model pads its batches); at least max_cuda_graph_batch_size.
    max_decode_rows: int
    # Concurrent steps of microbatch overlap; InferStateInfo.microbatch_index is below it.
    microbatch_count: int
    # Size of the scratch workspace that implementations needing one share by name.
    workspace_bytes: int
    # get_workspace(name, numel, dtype) returns process-wide storage on device, the same
    # tensor for the same arguments, so builders of one implementation share it.
    get_workspace: Callable[[str, int, torch.dtype], torch.Tensor]
    # Decode batches carry the shared-prefix groups of diverse sampling
    # (InferStateInfo.b_shared_seq_len and b_shared_radix_node_id).
    diverse_decode: bool = False


class AttentionBackend(ABC, Generic[LayerConfig, AttState, LayerInputs]):
    """Prepare shared state once per step and run attention with each layer's operands."""

    def __init__(self, layer_config: LayerConfig) -> None:
        self.layer_config = layer_config

    @cached_property
    def state_key(self) -> tuple:
        return type(self), self.build_context_key(self.layer_config)

    @abstractmethod
    def forward(
        self,
        att_state: AttState,
        layer_inputs: LayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out itself; concrete impls define tensor layouts.

        Kernels write into out directly. Compute elsewhere and copy only where a kernel cannot: it has
        no out parameter, it needs padded heads, or partial results merge at higher precision first.
        """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @classmethod
    @abstractmethod
    def build(cls, infer_state: "InferStateInfo", context: Any) -> AttState:
        """Prepare state once per compatible set of layers before capture or replay.

        Replay reads captured addresses and launch parameters. Keep graph storage for each
        (microbatch, batch size), update its contents in place and size launches for graph capacity.
        """
        pass

    @classmethod
    def create_build_context(cls, build_key: Any, ctx: AttentionBuildResources) -> Any:
        """Create shared storage using the exact key returned by build_context_key and model-wide resources."""
        return None

    @classmethod
    def build_context_key(cls, layer_config: Any) -> Any:
        """Return the layer fields that affect build, or None for configuration-independent builds."""
        return None

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        """Whether its kernels must not overlap CPU cache copies on the GPU."""
        return False
