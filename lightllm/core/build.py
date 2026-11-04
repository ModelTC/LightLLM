from dataclasses import dataclass
from typing import Callable

from lightllm.core.contracts.model import ModelAdapter
from lightllm.core.contracts.device import ExecutionLimits
from lightllm.core.contracts.state import FormatSpec
from lightllm.core.planner import AttentionRequest, Planner
from lightllm.core.registry import ProviderRegistry, default_providers


@dataclass(frozen=True)
class BuildConfig:
    layers: object
    manager: object
    limits: ExecutionLimits
    format: FormatSpec
    requested: dict
    recurrent_format: FormatSpec | None = None
    recurrent_requested: dict | None = None
    static_params: dict | None = None
    window_format: FormatSpec | None = None
    allocate_storage: Callable | None = None


class PreparedExecution:
    def __init__(
        self,
        adapter,
        layers,
        manager,
        runtime,
        limits,
        format,
        requested,
        recurrent_format=None,
        recurrent_requested=None,
        static_params=None,
        window_format=None,
        *,
        providers=None,
    ):
        self.providers = providers if providers is not None else default_providers()
        self.layers = tuple(layers)
        self.specs = specs = adapter.attention_specs()
        declarations = adapter.state_specs()
        primary = [state for state in declarations if state.retention != "window"]
        windows = [state for state in declarations if state.retention == "window"]
        states = {consumer: state for state in primary for consumer in state.consumers}
        tails = {consumer: state for state in windows for consumer in state.consumers}
        if len(states) != sum(len(state.consumers) for state in primary):
            raise ValueError("a layer must have exactly one primary attention resource")
        if len(tails) != sum(len(state.consumers) for state in windows):
            raise ValueError("a pooled layer must have one indexer tail")
        formats = {"history": format, "recurrent": recurrent_format, "window": window_format}
        selections = {"history": requested, "recurrent": recurrent_requested}
        requests = []
        for layer in self.layers:
            i = layer.layer_num_
            state = states[i]
            params = static_params[i] if static_params is not None else self.providers.static_params(specs[i], runtime)
            for phase, name in selections[state.retention].items():
                requests.append(
                    AttentionRequest(
                        i,
                        phase,
                        specs[i],
                        state,
                        formats[state.retention],
                        params,
                        name,
                        (("indexer_tail", tails[i], window_format),) if i in tails else (),
                    )
                )
        phases = tuple(dict.fromkeys(phase for selection in selections.values() if selection for phase in selection))
        planner = Planner(self.providers.attention_backends(phases), runtime.caps, limits)
        self.prepared = planner.prepare(requests)
        self.workspaces = self.prepared.reserve(runtime.allocate_workspace)
        self.stores = {}
        for retention, fmt in formats.items():
            group = [state for state in declarations if state.retention == retention]
            if not group and retention != "history":
                continue
            provider = self.providers.state_provider(fmt, retention)
            self.stores[retention] = provider.create_store(group, fmt, manager, runtime, specs)
        self.store = self.stores.get("history")
        self.recurrent_store = self.stores.get("recurrent")
        self.window_store = self.stores.get("window")
        self.plan = None
        self.runner = None
        self.decode_phase = "verify" if "verify" in requested else "decode"

    def bind(self, layers=None):
        if self.plan is not None:
            raise RuntimeError("bound execution plans cannot be replaced while resources or Graphs may be in use")
        layers = self.layers if layers is None else tuple(layers)
        if tuple(layer.layer_num_ for layer in layers) != tuple(layer.layer_num_ for layer in self.layers):
            raise ValueError("binding must use the planned layer identities and order")
        views = {}
        for store in self.stores.values():
            views.update(store.views())
        plan = self.prepared.bind(views, self.workspaces)
        writers, requires_writer = {}, set()
        for view in views.values():
            provider = self.providers.state_provider(view.format, view.state.retention)
            if provider.create_writer is not None:
                writers[view.state.producer] = provider.create_writer(view)
                requires_writer.update(view.state.consumers)
        bindings = {(item.layer_id, item.phase): item.execution for item in plan.bindings}
        for layer in layers:
            if layer.layer_num_ in requires_writer and layer.layer_num_ not in writers:
                raise ValueError("this model recipe does not support a read-only shared KV consumer")
        for layer in layers:
            layer.bound_prefill_attention = bindings.get((layer.layer_num_, "prefill"))
            layer.bound_decode_attention = bindings.get((layer.layer_num_, self.decode_phase))
            if layer.layer_num_ in writers:
                layer.bound_kv_writer = writers[layer.layer_num_]
        from lightllm.core.runner.attention import Runner

        self.plan = plan
        self.runner = Runner(plan, self.decode_phase)
        return plan

    @property
    def metadata_sources(self):
        return self.runner.metadata_sources

    def prepare_batch(self, batch):
        if self.runner is None:
            raise RuntimeError("execution must be bound before preparing a batch")
        return self.runner.prepare_batch(batch)


def prepare(*, model_adapter: ModelAdapter, providers: ProviderRegistry, device, config: BuildConfig):
    return PreparedExecution(
        model_adapter,
        config.layers,
        config.manager,
        device,
        config.limits,
        config.format,
        config.requested,
        config.recurrent_format,
        config.recurrent_requested,
        config.static_params,
        config.window_format,
        providers=providers,
    )


def build(*, model_adapter: ModelAdapter, providers: ProviderRegistry, device, config: BuildConfig):
    execution = prepare(model_adapter=model_adapter, providers=providers, device=device, config=config)
    if config.allocate_storage is not None:
        config.allocate_storage(execution)
    execution.bind()
    return execution
