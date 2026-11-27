"""Attention backends and the inputs to create_att_metadata and forward.

Each type is named for how long it lives:

- LayerConfig: one layer's static attention configuration, fixed when the model loads. An
  AttentionBackend instance holds this config.
- InferStateInfo: request layout and KV cache indices for one forward step.
- AttentionMetadata: the attention metadata shared by compatible layers in one microbatch. Backends that
  prepare page tables, recurrent indices or wrappers keep them in their metadata.
- LayerInputs: one layer call's tensors.

In each phase, one backend instance creates metadata for the layers whose metadata_group_key matches: the
backend class plus metadata_config_key(layer_config). That instance sets up its storage on first use and
keeps the storage CUDA graphs read.

Attention metadata is built every step, including CUDA graph replay steps. The caller writes the KV cache
before attention and passes the attention metadata explicitly rather than through a global forward context.
"""

import enum
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

import torch

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo

LayerConfig = TypeVar("LayerConfig")
AttentionMetadata = TypeVar("AttentionMetadata")
LayerInputs = TypeVar("LayerInputs")


class GraphMode(enum.Enum):
    """How a step runs: eagerly, captured into a decode CUDA graph, or replaying one."""

    EAGER = "eager"
    CAPTURE = "capture"
    REPLAY = "replay"


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


class AttentionBackend(ABC, Generic[LayerConfig, AttentionMetadata, LayerInputs]):
    """Prepare shared metadata once per step and run attention with each layer's operands."""

    # The backend's name in logs and selection errors.
    name: ClassVar[str]
    # The phase the backend runs in, "prefill" or "decode"; the selector binds it only to that phase.
    phase: ClassVar[str]
    # Whether its kernels must not overlap CPU cache copies on the GPU.
    requires_cpu_cache_sync: ClassVar[bool] = False

    def __init__(self, layer_config: LayerConfig) -> None:
        self._check_supported(layer_config)
        self.layer_config = layer_config
        self._runtime_initialized = False
        # Tensors CUDA graphs read, kept at fixed addresses per (microbatch, batch size, name).
        self._graph_tensors: dict[tuple, torch.Tensor] = {}

    @classmethod
    def _check_supported(cls, layer_config: LayerConfig) -> None:
        """Raise NotImplementedError if this backend cannot run the layer config."""

    @classmethod
    def metadata_config_key(cls, layer_config: LayerConfig) -> Any:
        """Config fields read by init_runtime and build_att_metadata; equal keys share metadata."""
        return None

    @cached_property
    def metadata_group_key(self) -> tuple:
        return type(self), self.metadata_config_key(self.layer_config)

    def create_att_metadata(self, infer_state: "InferStateInfo") -> AttentionMetadata:
        """Create this phase's shared metadata for one microbatch, setting up storage on first use."""
        if not self._runtime_initialized:
            self.init_runtime(infer_state)
            self._runtime_initialized = True
        return self.build_att_metadata(infer_state)

    def init_runtime(self, infer_state: "InferStateInfo") -> None:
        """Initialize storage on first use; retain needed values and buffers, never the step itself."""

    @abstractmethod
    def build_att_metadata(self, infer_state: "InferStateInfo") -> AttentionMetadata:
        """Prepare metadata once per compatible set of layers before capture or replay.

        Read only the layer config fields in metadata_config_key. Replay reads captured addresses and launch
        parameters: keep graph storage for each (microbatch, batch size), update its contents in
        place and size launches for graph capacity.
        """

    @abstractmethod
    def forward(
        self,
        att_metadata: AttentionMetadata,
        layer_inputs: LayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out itself; concrete backends define tensor layouts.

        Kernels write into out directly. Compute elsewhere and copy only where a kernel cannot: it has
        no out parameter, it needs padded heads, or partial results merge at higher precision first.
        """

    def update_graph_tensor(self, infer_state: "InferStateInfo", name: str, value: torch.Tensor) -> torch.Tensor:
        """Return value, or for a CUDA graph step its copy at a fixed address per (microbatch, batch size).

        The first call for a key keeps a clone; later calls copy value into it in place, so tensors a
        captured CUDA graph reads see each replayed step's values.
        """
        if not infer_state.uses_cuda_graph:
            return value
        key = (infer_state.microbatch_index, infer_state.batch_size, name)
        kept = self._graph_tensors.get(key)
        if kept is None:
            kept = self._graph_tensors[key] = value.clone()
        else:
            kept.copy_(value)
        return kept
