from dataclasses import dataclass, field
from typing import Mapping, Hashable

from lightllm.common.execution.contracts.model import ModelAdapter
from lightllm.common.execution.contracts.device import DeviceRuntime, ExecutionLimits
from lightllm.common.execution.contracts.provider import Phase
from lightllm.common.execution.contracts.state import FormatSpec, StateManager, StateSpec, StateStore, StateView
from lightllm.common.execution.compute.attention.backend import AttentionParams, AttentionSpec
from lightllm.common.execution.planner import AttentionRequest, Planner, AttentionSelectionPlan, BoundAttentionPlan
from lightllm.common.execution.registry import ProviderRegistry, StateProvider
from lightllm.common.execution.runner.attention import AttentionMetadataRunner


@dataclass(frozen=True)
class BuildConfig:
    limits: ExecutionLimits
    formats: Mapping[str, FormatSpec]
    requested: Mapping[str, Mapping[Phase, str]]
    static_params: Mapping[int, AttentionParams]


@dataclass(frozen=True)
class StateAllocation:
    retention: str
    declarations: tuple[StateSpec, ...]
    format: FormatSpec
    provider: StateProvider


@dataclass(frozen=True)
class AttentionBuild:
    """Selection only: no workspace, storage or model recipe has been mutated."""

    selection: AttentionSelectionPlan
    allocations: tuple[StateAllocation, ...]
    specs: Mapping[int, AttentionSpec]
    decode_phase: Phase

    def allocate(self, *, manager: StateManager | None, device: DeviceRuntime) -> "AttentionResources":
        workspaces = self.selection.reserve(device.allocate_workspace)
        stores = {
            item.retention: item.provider.create_store(item.declarations, item.format, manager, device, self.specs)
            for item in self.allocations
        }
        return AttentionResources(self, stores, workspaces)


@dataclass
class AttentionResources:
    """Own reservations; shared managers must finalize storage before bind reads native views."""

    build: AttentionBuild
    stores: Mapping[str, StateStore]
    workspaces: Mapping[Hashable, object]
    _bound: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)

    def bind(self) -> "AttentionExecution":
        if self._closed:
            raise RuntimeError("closed attention resources cannot be bound")
        if self._bound:
            raise RuntimeError("bound execution plans cannot be replaced while resources or Graphs may be in use")
        views = {}
        for store in self.stores.values():
            for key, view in store.views().items():
                if key in views:
                    raise ValueError("a resource must resolve to exactly one native view")
                views[key] = view
        plan = self.build.selection.bind(views, self.workspaces)
        writers = {}
        providers = {item.retention: item.provider for item in self.build.allocations}
        for key, view in views.items():
            create_writer = providers[view.state.retention].create_writer
            if create_writer is not None:
                writers[key] = create_writer(view)
        metadata = AttentionMetadataRunner(plan, self.build.decode_phase)
        self._bound = True
        return AttentionExecution(plan, metadata, views, writers, self)

    def close(self) -> None:
        """Release provider-owned resources after all bound users have drained."""

        if self._closed:
            return
        for store in self.stores.values():
            close = getattr(store, "close", None)
            if close is not None:
                close()
        self._closed = True


@dataclass(frozen=True)
class AttentionExecution:
    """Fully bound execution. Resources stay owned for the lifetime of execution and Graphs."""

    plan: BoundAttentionPlan
    metadata: AttentionMetadataRunner
    views: Mapping[tuple[str, str], StateView]
    writers: Mapping[tuple[str, str], object]
    resources: AttentionResources

    def close(self) -> None:
        """Close the execution owner after graphs and asynchronous work drain."""

        self.resources.close()


def prepare(
    *, model_adapter: ModelAdapter, providers: ProviderRegistry, device: DeviceRuntime, config: BuildConfig
) -> AttentionBuild:
    specs = model_adapter.attention_specs()
    declarations = model_adapter.state_specs()
    resources = model_adapter.attention_states()
    declared = {state.key: state for state in declarations}
    if len(declared) != len(declarations):
        raise ValueError("state resource identities must be unique")
    if set(resources) != set(specs):
        raise ValueError("state inputs must describe exactly the declared attention layers")
    for inputs in resources.values():
        for state in (inputs.primary,) + tuple(state for _, state in inputs.auxiliary):
            if declared.get(state.key) != state:
                raise ValueError("attention input refers to an undeclared state resource")
    requests = []
    for i, spec in specs.items():
        inputs = resources[i]
        state = inputs.primary
        if state.retention not in config.formats or not config.requested.get(state.retention):
            raise ValueError(f"missing format or execution phases for {state.retention} state")
        auxiliary = []
        for name, resource in inputs.auxiliary:
            if resource.retention not in config.formats:
                raise ValueError(f"missing format for auxiliary {name}: {resource.retention}")
            auxiliary.append((name, resource, config.formats[resource.retention]))
        for phase, name in config.requested[state.retention].items():
            requests.append(
                AttentionRequest(
                    i,
                    phase,
                    spec,
                    state,
                    config.formats[state.retention],
                    config.static_params[i],
                    name,
                    tuple(auxiliary),
                )
            )
    backends = providers.attention_backends(tuple(dict.fromkeys(r.phase for r in requests)))
    providers.validate_attention_requests(requests, backends)
    selection = Planner(backends, device.caps, config.limits).prepare(requests)
    groups = {}
    for state in declarations:
        groups.setdefault(state.retention, []).append(state)
    allocations = []
    for retention, group in groups.items():
        if retention not in config.formats:
            raise ValueError(f"missing format for {retention} state")
        fmt = config.formats[retention]
        allocations.append(StateAllocation(retention, tuple(group), fmt, providers.state_provider(fmt, retention)))
    decode_phase = "verify" if any(r.phase == "verify" for r in requests) else "decode"
    return AttentionBuild(selection, tuple(allocations), specs, decode_phase)


def build(
    *,
    model_adapter: ModelAdapter,
    providers: ProviderRegistry,
    device: DeviceRuntime,
    config: BuildConfig,
    manager=None,
) -> AttentionExecution:
    """Build with self-allocating stores; shared allocation uses prepare, allocate, then bind."""
    prepared = prepare(model_adapter=model_adapter, providers=providers, device=device, config=config)
    return prepared.allocate(manager=manager, device=device).bind()


@dataclass(frozen=True)
class ExpertResources:
    workspaces: tuple
    prefill_buffer: object
    decode_buffer: object
    prefill_limit: int
    decode_limit: int
    num_sms: int
    round_scales: bool


def build_experts(spec, view, device, resources, providers, transfers=None):
    from lightllm.common.execution.planner.weights import bind_compute
    from lightllm.common.execution.runner.expert import ExpertRunner
    from lightllm.common.execution.transfer.expert import ExpertTransfer

    compute = bind_compute(
        spec,
        view,
        device,
        providers.expert,
        bind_kwargs={"round_scales": resources.round_scales, "workspaces": resources.workspaces},
    )
    key = (compute.total_experts, resources.round_scales, spec.round_intermediates, view.fields["gate_up"].device)
    transfer = None if transfers is None else transfers.get(key)
    if transfer is None:
        transfer = ExpertTransfer(
            resources.prefill_buffer,
            resources.decode_buffer,
            experts=compute.total_experts,
            prefill_limit=resources.prefill_limit,
            decode_limit=resources.decode_limit,
            num_sms=resources.num_sms,
            round_scales=resources.round_scales,
            separate_contributions=spec.round_intermediates,
        )
        if transfers is not None:
            transfers[key] = transfer
    return ExpertRunner(compute, transfer)
