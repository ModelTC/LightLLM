from lightllm.common.execution.contracts.state import FormatSpec
from lightllm.common.execution.contracts.device import DeviceCaps
from lightllm.common.execution.compute.linear.spec import LinearSpec
from lightllm.common.execution.planner.plan import LinearPlan, MoePlan
from lightllm.common.execution.formats.weights.view import WeightView
from lightllm.common.execution.planner.weights import bind_compute
from lightllm.common.execution.registry import default_providers


def weight_device_caps(device, runtime=None):
    if runtime is not None:
        if device != runtime.device:
            raise ValueError("weight view and execution device differ")
        return runtime.caps
    import torch

    return DeviceCaps(
        device.type, device.index, torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    )


def finalize_weights(layers, runtime, providers):
    layers = tuple(layers)
    for prepare in providers.weight_finalizers:
        prepare(layers, runtime)
    seen = set()
    for layer in layers:
        for weight in vars(layer).values():
            finalize = getattr(weight, "finalize_weight", None)
            if id(weight) not in seen and callable(finalize):
                seen.add(id(weight))
                finalize(runtime)


def bind_moe(specs, layers, recipes, *, providers=None, runtime=None, requested="auto"):
    import torch

    providers = providers if providers is not None else default_providers()
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
            view = native_view()
            if not isinstance(view, WeightView):
                view = WeightView(*view)
            caps = weight_device_caps(view.fields["gate_up"].device, runtime)
            pending.append((layer_id, bind_compute(spec, view, caps, providers.moe, requested=requested)))
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
            if requested != "auto":
                raise ValueError(f"layer {layer_id}: {requested} does not provide expert-parallel execution")
            if (
                method != "fp8w8a8-b128-deepgemm"
                or spec.dtype != torch.bfloat16
                or torch.cuda.get_device_capability(weight.w13.weight.device) != (9, 0)
            ):
                legacy.append((layer_id, "expert parallel format/device"))
                continue
            pending.append((layer_id, bind_experts(spec, weight, expert_transfers, providers=providers)))
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
        caps = weight_device_caps(device, runtime)
        view = WeightView(
            fmt,
            dict(
                gate_up=weight.w13.weight,
                down=weight.w2.weight,
                gate_up_scale=weight.w13.weight_scale,
                down_scale=weight.w2.weight_scale,
                correction_bias=weight.e_score_correction_bias,
                expert_scale=weight.per_expert_scale,
            ),
        )
        bound = bind_compute(
            spec,
            view,
            caps,
            providers.moe,
            requested=requested,
            probe_kwargs={"fused_shared_experts": weight.num_fused_shared_experts},
        )
        pending.append((layer_id, bound))
    if legacy and requested != "auto":
        raise ValueError(f"{requested} cannot bind legacy MoE weights: {legacy}")
    for layer_id, bound in pending:
        if getattr(weights[layer_id].experts, "enable_ep_moe", False):
            recipes[layer_id].bound_ep = bound
        else:
            recipes[layer_id].bound_moe = bound
    return MoePlan(tuple(pending), tuple(legacy))


def bind_experts(spec, weight, transfers=None, *, providers=None):
    import torch
    from lightllm.distributed import dist_group_manager
    from lightllm.utils.envs_utils import get_env_use_ue8m0_scales
    from lightllm.common.execution.build import ExpertResources, build_experts

    providers = providers if providers is not None else default_providers()
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
    view = WeightView(
        fmt,
        dict(
            gate_up=weight.w13.weight,
            down=weight.w2.weight,
            gate_up_scale=weight.w13.weight_scale,
            down_scale=weight.w2.weight_scale,
            correction_bias=weight.e_score_correction_bias,
            expert_scale=weight.per_expert_scale,
            redundancy_ids=weight.redundancy_expert_ids_tensor,
            expert_counter=weight.routed_expert_counter_tensor,
            count_routes=weight.auto_update_redundancy_expert,
        ),
    )
    resources = ExpertResources(
        tuple(dist_group_manager.get_deep_ep_prefill_moe_workspace(i) for i in range(len(dist_group_manager.groups))),
        dist_group_manager.ep_buffer,
        dist_group_manager.ep_low_latency_buffer,
        dist_group_manager.ll_num_tokens,
        dist_group_manager.ll_decode_num_tokens,
        dist_group_manager.ep_num_sms or 0,
        get_env_use_ue8m0_scales(),
    )
    caps = DeviceCaps(device.type, device.index, torch.cuda.get_device_capability(device))
    return build_experts(spec, view, caps, resources, providers, transfers)


def bind_linears(layers, allocate, *, providers=None, runtime=None):
    import torch
    from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl, BMMWeightTpl

    providers = providers if providers is not None else default_providers()

    implementations = {
        "none": ("torch.linear", "plain"),
        "fp8w8a8-pt-triton": ("triton.linear.fp8_tensor", "fp8_per_tensor"),
        "fp8w8a8-pt-sgl": ("sgl.linear.fp8_tensor", "fp8_per_tensor"),
        "fp8w8a8g64-triton": ("triton.linear.fp8_channel_g64", "fp8_per_channel"),
        "fp8w8a8g128-triton": ("triton.linear.fp8_channel_g128", "fp8_per_channel"),
        "fp8w8a8-b128-triton": ("triton.linear.fp8_block128", "fp8_block128"),
    }
    pending, legacy, seen = [], [], set()
    for layer in layers:
        for name, weight in vars(layer).items():
            if id(weight) in seen:
                continue
            native_view = getattr(weight, "native_linear_view", None)
            if not isinstance(weight, (MMWeightTpl, BMMWeightTpl)) and not callable(native_view):
                continue
            seen.add(id(weight))
            label = f"{getattr(layer, 'layer_num_', 'pre_post')}.{name}"
            if callable(native_view):
                spec, view, entry = native_view()
                if entry not in ("mm", "bmm"):
                    raise ValueError(f"{label}: unknown linear entry point {entry}")
                caps = weight_device_caps(view.fields["weight"].device, runtime)
                bound = bind_compute(spec, view, caps, providers.linear, bind_kwargs={"allocate": allocate})
                pending.append((label, weight, entry, bound))
                continue
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
            requested, encoding = implementations[method]
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
            caps = weight_device_caps(tensor.device, runtime)
            view = WeightView(fmt, {"weight": tensor, "scale": scale, "bias": bias})
            bound = bind_compute(
                spec, view, caps, providers.linear, requested=requested, bind_kwargs={"allocate": allocate}
            )
            pending.append((label, weight, "bmm" if batched else "mm", bound))
    # Install only after every selected format and native view has been checked.
    for _, weight, entry, bound in pending:
        setattr(weight, entry, bound.run)
        setattr(weight, "bound_" + entry, bound)
    return LinearPlan(tuple((label, bound) for label, _, _, bound in pending), tuple(legacy))
