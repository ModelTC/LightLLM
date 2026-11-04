from __future__ import annotations

from typing import Hashable, Mapping, Protocol, TYPE_CHECKING, TypeVar, runtime_checkable
from lightllm.core.contracts.device import DeviceCaps, DeviceRuntime, ExecutionLimits, GraphBinding
from lightllm.core.contracts.provider import AttentionOffer, Phase, ProbeResult
from lightllm.core.contracts.state import FormatSpec, StateView

if TYPE_CHECKING:
    from lightllm.core.compute.attention.batch import AttentionBatch
    from lightllm.core.planner.plan import AttentionRequest


class AttentionSpec(Protocol):
    def validate_state(self, request: AttentionRequest) -> None:
        ...


class AttentionParams(Protocol):
    @property
    def spec(self) -> AttentionSpec:
        ...


class AttentionMetadata(Protocol):
    def copy_for_graph(self, new: AttentionMetadata) -> None:
        ...


InputT = TypeVar("InputT", contravariant=True)
MetadataT = TypeVar("MetadataT", bound=AttentionMetadata)
OutputT = TypeVar("OutputT", covariant=True)


class BoundAttention(Protocol[InputT, MetadataT, OutputT]):
    phase: Phase
    transfer_requires_compute_sync: bool

    @property
    def metadata_key(self) -> Hashable:
        """Equal keys must allow either execution's metadata to serve both for the same batch.

        Include every difference in layout, algorithm configuration or referenced
        state identity that prevents sharing. Metadata cannot retain layer-local scratch.
        """
        ...

    def create_metadata(self, batch: AttentionBatch) -> MetadataT:
        ...

    def run(self, inputs: InputT, metadata: MetadataT) -> OutputT:
        ...


@runtime_checkable
class GraphAttention(Protocol[MetadataT]):
    def bind_graph(self, metadata: MetadataT, runtime: DeviceRuntime) -> GraphBinding:
        ...


class AttentionBackend(Protocol):
    name: str
    implementation: str
    phase: Phase
    supports_auxiliary_states: bool

    def probe(
        self,
        attention_spec: AttentionSpec,
        format_spec: FormatSpec,
        device_caps: DeviceCaps,
        limits: ExecutionLimits,
        *,
        auxiliary_formats: Mapping[str, FormatSpec] | None = None,
    ) -> ProbeResult:
        ...

    def bind(
        self,
        offer: AttentionOffer,
        state_view: StateView,
        static_params: AttentionParams,
        workspace: object,
        *,
        auxiliary_views: Mapping[str, StateView] | None = None,
    ) -> BoundAttention:
        ...
