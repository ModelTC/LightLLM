from dataclasses import dataclass
from typing import Any, Hashable, Mapping
from lightllm.common.execution.contracts.provider import AttentionOffer, Phase
from lightllm.common.execution.contracts.state import StateSpec, FormatSpec
from lightllm.common.execution.compute.attention.backend import AttentionBackend, AttentionSpec, AttentionParams, BoundAttention


@dataclass(frozen=True)
class AttentionBinding:
    layer_id: int
    phase: Phase
    backend: str
    execution: BoundAttention


@dataclass(frozen=True)
class BoundAttentionPlan:
    bindings: tuple[AttentionBinding, ...]
    workspace_bytes: int


@dataclass(frozen=True)
class LinearPlan:
    bindings: tuple[tuple[str, Any], ...]
    legacy: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MoePlan:
    bindings: tuple[tuple[int, Any], ...]
    legacy: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class AttentionRequest:
    layer_id: int
    phase: Phase
    spec: AttentionSpec
    state: StateSpec
    format: FormatSpec
    params: AttentionParams
    requested: str = "auto"
    auxiliary_states: tuple[tuple[str, StateSpec, FormatSpec], ...] = ()


@dataclass(frozen=True)
class SelectedAttention:
    request: AttentionRequest
    backend: AttentionBackend
    offer: AttentionOffer
    workspace_key: Hashable


@dataclass(frozen=True)
class AttentionSelectionPlan:
    selections: tuple[SelectedAttention, ...]
    workspace_sizes: Mapping[Hashable, int]

    @property
    def workspace_bytes(self):
        return sum(self.workspace_sizes.values())

    def reserve(self, allocate):
        return {key: allocate(size) for key, size in self.workspace_sizes.items() if size}

    def bind(self, views, workspaces):
        bindings = []
        for selection in self.selections:
            request, backend, offer = selection.request, selection.backend, selection.offer
            view = views[request.state.key]
            if view.state != request.state or view.format != request.format:
                raise ValueError("allocated state does not match the prepared plan")
            auxiliary = {}
            for name, state, format in request.auxiliary_states:
                native = views[state.key]
                if native.state != state or native.format != format:
                    raise ValueError("allocated auxiliary state does not match the prepared plan")
                auxiliary[name] = native
            execution = backend.bind(
                offer, view, request.params, workspaces.get(selection.workspace_key), auxiliary_views=auxiliary
            )
            bindings.append(AttentionBinding(request.layer_id, request.phase, backend.name, execution))
        return BoundAttentionPlan(tuple(bindings), self.workspace_bytes)
