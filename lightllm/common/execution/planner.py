"""Select implementations and reserve scratch before allocating state storage."""

from dataclasses import dataclass
from typing import Any, Protocol

from .specs import AttentionOffer, DenseAttentionSpec, GatedDeltaNetSpec, MLAAttentionSpec, SparseMLAAttentionSpec
from .specs import DeviceCaps, ExecutionLimits, ProbeResult
from .specs import KimiDeltaAttentionSpec, PooledNoPEAttentionSpec, StateSpec, FormatSpec, LinearSpec


class AttentionBackend(Protocol):
    name: str

    def probe(self, attention_spec, format_spec, device_caps, limits) -> ProbeResult:
        ...

    def bind(self, offer, state_view, static_params, workspace) -> Any:
        ...


@dataclass(frozen=True)
class AttentionBinding:
    layer_id: int
    phase: str
    backend: str
    execution: Any


@dataclass(frozen=True)
class ExecutionPlan:
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
    phase: str
    spec: Any
    state: Any
    format: Any
    params: Any
    requested: str = "auto"
    auxiliary_states: tuple[tuple[str, StateSpec, FormatSpec], ...] = ()


@dataclass(frozen=True)
class SelectedAttention:
    request: AttentionRequest
    backend: AttentionBackend
    offer: AttentionOffer
    workspace_key: Any


@dataclass(frozen=True)
class PreparedPlan:
    selections: tuple[SelectedAttention, ...]
    workspace_sizes: dict

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
            kwargs = {"auxiliary_views": auxiliary} if auxiliary else {}
            execution = backend.bind(offer, view, request.params, workspaces.get(selection.workspace_key), **kwargs)
            bindings.append(AttentionBinding(request.layer_id, request.phase, backend.name, execution))
        return ExecutionPlan(tuple(bindings), self.workspace_bytes)


class Planner:
    @staticmethod
    def bind_moe(specs, layers, recipes):
        import torch
        from .moe import MoeBackend
        from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight

        recipes = tuple(recipes)
        weights = {recipe.layer_num_: weight for recipe, weight in zip(recipes, layers, strict=True)}
        recipes = {layer.layer_num_: layer for layer in recipes}
        pending, legacy = [], []
        expert_transfers = {}
        if set(specs) != {
            i
            for i, layer in weights.items()
            if isinstance(getattr(layer, "experts", None), FusedMoeWeight)
            or callable(getattr(getattr(layer, "experts", None), "native_moe_view", None))
        }:
            raise ValueError("model MoE declarations must cover exactly the loaded expert layers")
        for layer_id, spec in specs.items():
            weight = getattr(weights[layer_id], "experts", None)
            if not getattr(recipes[layer_id], "is_moe", False):
                raise ValueError(f"layer {layer_id}: recipe disagrees with the declared MoE semantics")
            native_view = getattr(weight, "native_moe_view", None)
            if callable(native_view):
                fmt, fields = native_view()
                device = fields["gate_up"].device
                caps = DeviceCaps(device.type, device.index, torch.cuda.get_device_capability(device))
                backend = MoeBackend()
                result = backend.probe(spec, fmt, caps)
                if not result.offers:
                    raise ValueError(f"layer {layer_id}: {result.reason}")
                pending.append((layer_id, backend.bind(result.offers[0], **fields)))
                continue
            if type(weight) is not FusedMoeWeight:
                raise ValueError(f"layer {layer_id}: model declared MoE without matching expert weights")
            method = weight.quant_method.method_name
            if method not in ("none", "fp8w8a8-b128-triton", "fp8w8a8-b128-deepgemm"):
                legacy.append((layer_id, method))
                continue
            if weight.w13.weight_scale is not None and weight.w13.weight_scale.dtype == torch.int32:
                legacy.append((layer_id, "packed ue8m0 scales"))
                continue
            if not weight.verify_load():
                raise ValueError(f"layer {layer_id}: MoE weights must be fully loaded before binding")
            if (
                weight.hidden_size,
                weight.moe_intermediate_size,
                weight.n_routed_experts,
                weight.num_experts_per_tok,
                weight.norm_topk_prob,
                weight.scoring_func,
                weight.routed_scaling_factor,
                weight.n_group or 1,
                weight.topk_group if weight.n_group else 1,
            ) != (
                spec.hidden_size,
                spec.intermediate_size,
                spec.experts,
                spec.top_k,
                spec.renormalize,
                spec.scoring,
                spec.routed_scale,
                spec.groups,
                spec.selected_groups,
            ):
                raise ValueError(f"layer {layer_id}: loaded MoE dimensions/routing disagree with model semantics")
            if weight.enable_ep_moe:
                if (
                    method != "fp8w8a8-b128-deepgemm"
                    or spec.dtype != torch.bfloat16
                    or torch.cuda.get_device_capability(weight.w13.weight.device) != (9, 0)
                ):
                    legacy.append((layer_id, "expert parallel format/device"))
                    continue
                pending.append((layer_id, Planner.bind_experts(spec, weight, expert_transfers)))
                continue
            quantized = method != "none"
            fmt = FormatSpec(
                "fp8_block128" if quantized else "plain",
                weight.w13.weight.dtype,
                "expert_output_input",
                ("gate_up", "down", "gate_up_scale", "down_scale") if quantized else ("gate_up", "down"),
                1,
                (weight.tp_rank_, weight.tp_world_size_),
                group_size=128 if quantized else None,
                scale_dtype=torch.float32 if quantized else None,
            )
            device = weight.w13.weight.device
            caps = DeviceCaps(device.type, device.index, torch.cuda.get_device_capability(device))
            backend = MoeBackend()
            result = backend.probe(spec, fmt, caps, fused_shared_experts=weight.num_fused_shared_experts)
            if not result.offers:
                raise ValueError(f"layer {layer_id}: {result.reason}")
            bound = backend.bind(
                result.offers[0],
                weight.w13.weight,
                weight.w2.weight,
                gate_up_scale=weight.w13.weight_scale,
                down_scale=weight.w2.weight_scale,
                correction_bias=weight.e_score_correction_bias,
                expert_scale=weight.per_expert_scale,
            )
            pending.append((layer_id, bound))
        for layer_id, bound in pending:
            if weights[layer_id].experts.enable_ep_moe:
                recipes[layer_id].bound_ep = bound
            else:
                recipes[layer_id].bound_moe = bound
        return MoePlan(tuple(pending), tuple(legacy))

    @staticmethod
    def bind_experts(spec, weight, transfers=None):
        import torch
        from lightllm.distributed import dist_group_manager
        from lightllm.utils.envs_utils import get_env_use_ue8m0_scales
        from .expert_compute import ExpertComputeBackend
        from .expert_runner import ExpertRunner
        from lightllm.common.transfer.expert import ExpertTransfer

        fmt = FormatSpec(
            "fp8_block128",
            weight.w13.weight.dtype,
            "local_expert_output_input",
            ("gate_up", "down", "gate_up_scale", "down_scale"),
            1,
            (weight.global_rank_, weight.global_world_size),
            group_size=128,
            scale_dtype=torch.float32,
        )
        device = weight.w13.weight.device
        backend = ExpertComputeBackend()
        result = backend.probe(
            spec, fmt, DeviceCaps(device.type, device.index, torch.cuda.get_device_capability(device))
        )
        if not result.offers:
            raise ValueError(result.reason)
        round_scales = get_env_use_ue8m0_scales()
        compute = backend.bind(
            result.offers[0],
            weight.w13.weight,
            weight.w2.weight,
            gate_up_scale=weight.w13.weight_scale,
            down_scale=weight.w2.weight_scale,
            correction_bias=weight.e_score_correction_bias,
            expert_scale=weight.per_expert_scale,
            redundancy_ids=weight.redundancy_expert_ids_tensor,
            expert_counter=weight.routed_expert_counter_tensor,
            count_routes=weight.auto_update_redundancy_expert,
            round_scales=round_scales,
            workspaces=tuple(
                dist_group_manager.get_deep_ep_prefill_moe_workspace(i) for i in range(len(dist_group_manager.groups))
            ),
        )
        key = (compute.total_experts, round_scales, spec.round_intermediates, device)
        transfer = None if transfers is None else transfers.get(key)
        if transfer is None:
            transfer = ExpertTransfer(
                dist_group_manager.ep_buffer,
                dist_group_manager.ep_low_latency_buffer,
                experts=compute.total_experts,
                prefill_limit=dist_group_manager.ll_num_tokens,
                decode_limit=dist_group_manager.ll_decode_num_tokens,
                num_sms=dist_group_manager.ep_num_sms or 0,
                round_scales=round_scales,
                separate_contributions=spec.round_intermediates,
            )
            if transfers is not None:
                transfers[key] = transfer
        return ExpertRunner(compute, transfer)

    @staticmethod
    def bind_linears(layers, allocate):
        import torch
        from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl, BMMWeightTpl
        from .linear import LinearBackend

        implementations = {
            "none": "torch.linear",
            "fp8w8a8-pt-triton": "triton.linear.fp8_tensor",
            "fp8w8a8-pt-sgl": "sgl.linear.fp8_tensor",
            "fp8w8a8g64-triton": "triton.linear.fp8_channel_g64",
            "fp8w8a8g128-triton": "triton.linear.fp8_channel_g128",
            "fp8w8a8-b128-triton": "triton.linear.fp8_block128",
        }
        pending, legacy, seen = [], [], set()
        for layer in layers:
            for name, weight in vars(layer).items():
                if not isinstance(weight, (MMWeightTpl, BMMWeightTpl)) or id(weight) in seen:
                    continue
                seen.add(id(weight))
                label = f"{getattr(layer, 'layer_num_', 'pre_post')}.{name}"
                batched = isinstance(weight, BMMWeightTpl)
                method = "none" if batched else weight.quant_method.method_name
                if method not in implementations:
                    legacy.append((label, method))
                    continue
                if not weight.verify_load():
                    raise ValueError(f"{label}: linear weights must be fully loaded before binding")
                tensor = weight.weight if batched else weight.mm_param.weight
                scale = None if batched else weight.mm_param.weight_scale
                bias = None if batched else weight.bias
                spec = LinearSpec(
                    weight.dim1 if batched else weight.in_dim,
                    weight.dim2 if batched else sum(weight.out_dims),
                    weight.data_type_,
                    weight.dim0 if batched else None,
                )
                backend = LinearBackend(implementations[method])
                encoding = backend.encodings[backend.name]
                fmt = FormatSpec(
                    encoding,
                    tensor.dtype,
                    "batch_input_output" if batched else "output_input",
                    ("weight",) if scale is None else ("weight", "scale"),
                    1,
                    (weight.tp_rank_, weight.tp_world_size_),
                    group_size=128 if encoding == "fp8_block128" else None,
                    scale_dtype=None if scale is None else scale.dtype,
                )
                caps = DeviceCaps(
                    tensor.device.type,
                    tensor.device.index,
                    torch.cuda.get_device_capability(tensor.device) if tensor.is_cuda else None,
                )
                result = backend.probe(spec, fmt, caps)
                if not result.offers:
                    raise ValueError(f"{label}: {result.reason}")
                bound = backend.bind(result.offers[0], tensor, scale, bias, allocate=allocate)
                pending.append((label, weight, "bmm" if batched else "mm", bound))
        # Install only after every selected format and native view has been checked.
        for _, weight, entry, bound in pending:
            setattr(weight, entry, bound.run)
        return LinearPlan(tuple((label, bound) for label, _, _, bound in pending), tuple(legacy))

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
            if isinstance(request.spec, DenseAttentionSpec):
                shape = request.state.shape
                local_heads = max(1, shape[0] // request.format.shard[1])
                if len(shape) != 2 or (local_heads, shape[-1]) != (request.spec.kv_heads, request.spec.head_dim):
                    raise ValueError(f"layer {request.layer_id}: state shape disagrees with attention semantics")
                if getattr(request.params, "spec", request.spec) != request.spec:
                    raise ValueError("static attention parameters differ from the probed semantics")
            elif isinstance(request.spec, MLAAttentionSpec):
                shape = (request.spec.latent_dim, request.spec.rope_dim)
                if isinstance(request.spec, SparseMLAAttentionSpec):
                    shape += (request.spec.index_dim,)
                if request.state.shape != shape:
                    raise ValueError("MLA latent state shape disagrees with attention semantics")
                if request.params.spec != request.spec:
                    raise ValueError("static MLA parameters differ from the probed semantics")
            elif isinstance(request.spec, GatedDeltaNetSpec):
                shape, spec = request.state.shape, request.spec
                tp = request.format.shard[1]
                expected = (spec.k_heads, spec.v_heads, spec.key_dim, spec.value_dim, spec.conv_size - 1)
                if len(shape) != 5 or (shape[0] // tp, shape[1] // tp, *shape[2:]) != expected:
                    raise ValueError(f"layer {request.layer_id}: recurrent state shape disagrees with semantics")
                if getattr(request.params, "spec", request.spec) != request.spec:
                    raise ValueError("static recurrent parameters differ from the probed semantics")
            elif isinstance(request.spec, KimiDeltaAttentionSpec):
                spec, tp = request.spec, request.format.shard[1]
                expected = (spec.heads * tp, spec.heads * tp, spec.head_dim, spec.head_dim, spec.conv_size - 1)
                if request.state.shape != expected or request.params.spec != spec:
                    raise ValueError("KDA state or static parameters disagree with attention semantics")
            elif isinstance(request.spec, PooledNoPEAttentionSpec):
                if request.state.shape != (request.spec.latent_dim, request.spec.index_dim):
                    raise ValueError("pooled latent state shape disagrees with attention semantics")
                if request.params.spec != request.spec:
                    raise ValueError("static pooled parameters differ from the probed semantics")
                if len(request.auxiliary_states) != 1:
                    raise ValueError("pooled attention requires its separately declared indexer tail")
                name, tail, _ = request.auxiliary_states[0]
                if (
                    name != "indexer_tail"
                    or tail.retention != "window"
                    or tail.shape != (request.spec.pool_size, request.spec.pool_size, request.spec.index_dim)
                ):
                    raise ValueError("indexer tail window disagrees with pooled semantics")
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
