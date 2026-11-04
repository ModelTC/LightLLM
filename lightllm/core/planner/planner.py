from lightllm.core.compute.attention.backend import AttentionBackend
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from .plan import (
    AttentionBinding,
    AttentionRequest,
    ExecutionPlan,
    LinearPlan,
    MoePlan,
    PreparedPlan,
    SelectedAttention,
)


class Planner:
    def __init__(self, backends: tuple[AttentionBackend, ...], device: DeviceCaps, limits: ExecutionLimits):
        if len({b.name for b in backends}) != len(backends):
            raise ValueError("backend names must be unique")
        self.backends = backends
        self.device = device
        self.limits = limits

    def select(self, spec, format, phase: str, requested: str = "auto", auxiliary_formats=None):
        failures = []
        for backend in self.backends:
            if requested not in ("auto", backend.name, getattr(backend, "implementation", None)):
                continue
            if auxiliary_formats and not getattr(backend, "supports_auxiliary_states", False):
                failures.append(f"{backend.name}: does not consume auxiliary state")
                continue
            kwargs = {"auxiliary_formats": auxiliary_formats} if auxiliary_formats else {}
            result = backend.probe(spec, format, self.device, self.limits, **kwargs)
            for offer in result.offers:
                if offer.backend != backend.name:
                    raise ValueError("backend returned an offer with another backend's identity")
                if offer.phase != phase or offer.format != format:
                    continue
                if offer.workspace_bytes <= self.limits.workspace_bytes:
                    return backend, offer
            failures.append(f"{backend.name}: {result.reason or 'no compatible phase/format within workspace budget'}")
        raise ValueError(f"{phase} attention ({requested}): " + ("; ".join(failures) or "backend is not registered"))

    def prepare(self, requests) -> PreparedPlan:
        selections, sizes, identities, states = [], {}, set(), {}
        for request in requests:
            identity = request.layer_id, request.phase
            if identity in identities:
                raise ValueError(f"duplicate layer/phase: {identity}")
            identities.add(identity)
            if request.layer_id not in request.state.consumers:
                raise ValueError(f"layer {request.layer_id} is not a declared consumer of {request.state.key}")
            validate = getattr(request.spec, "validate_state", None)
            if validate is not None:
                validate(request)
            names, keys = set(), {request.state.key}
            for name, state, _ in request.auxiliary_states:
                if not name or name in names or state.key in keys:
                    raise ValueError("auxiliary state names and resources must be unique")
                if state.role != request.state.role or request.layer_id not in state.consumers:
                    raise ValueError("layer is not an auxiliary state consumer in the same model role")
                names.add(name)
                keys.add(state.key)
            for state, format in ((request.state, request.format),) + tuple(
                (s, f) for _, s, f in request.auxiliary_states
            ):
                previous = states.setdefault(state.key, (state, format))
                if previous != (state, format):
                    raise ValueError("one resource cannot have conflicting state or format declarations")
            try:
                backend, offer = self.select(
                    request.spec,
                    request.format,
                    request.phase,
                    request.requested,
                    {name: format for name, _, format in request.auxiliary_states},
                )
            except ValueError as exc:
                raise ValueError(f"layer {request.layer_id}: {exc}") from exc
            key = (request.state.role, backend.name, offer.workspace_group or identity)
            sizes[key] = max(sizes.get(key, 0), offer.workspace_bytes)
            selections.append(SelectedAttention(request, backend, offer, key))
        total = sum(sizes.values())
        if total > self.limits.workspace_bytes:
            raise ValueError(f"plan workspace {total} exceeds budget {self.limits.workspace_bytes}")
        return PreparedPlan(tuple(selections), sizes)

    def bind(self, layers, allocate_workspace) -> ExecutionPlan:
        requests, views = [], {}
        for layer, phase, spec, view, params, requested in layers:
            if view.state.key in views and views[view.state.key] is not view:
                raise ValueError("a resource must resolve to exactly one native view")
            views[view.state.key] = view
            requests.append(AttentionRequest(layer, phase, spec, view.state, view.format, params, requested))
        prepared = self.prepare(requests)
        return prepared.bind(views, prepared.reserve(allocate_workspace))

    @staticmethod
    def bind_linears(layers, allocate):
        from lightllm.common.basemodel.weight_binding import bind_linears

        return bind_linears(layers, allocate)

    @staticmethod
    def bind_moe(specs, layers, recipes):
        from lightllm.common.basemodel.weight_binding import bind_moe

        return bind_moe(specs, layers, recipes)

    @staticmethod
    def bind_experts(spec, weight, transfers=None):
        from lightllm.common.basemodel.weight_binding import bind_experts

        return bind_experts(spec, weight, transfers)
