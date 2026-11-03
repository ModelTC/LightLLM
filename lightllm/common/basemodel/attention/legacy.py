from .state_writers import bind_kv_writer
from lightllm.common.attention_state.full_att import (
    FullAttentionState,
    StaticFP8FullAttentionState,
    MLAFullAttentionState,
    PackedMLAFullAttentionState,
    FP8E8M0PackedMLAState,
)
from lightllm.common.attention_state.bounded_att import (
    GDNState,
    KDAState,
    FP8E8M0SlidingWindowState,
    WindowCompressorState,
)
from .base_att import AttControl
from .binding import BoundAttention
from .contracts import KVFormat, MlaFormat, MlaSpec, ResourceBinding, ResourceId
from .planner import select_attention
from .recurrent import BoundRecurrent
from .mla import BoundMla
from .sparse import BoundSparse


def explain_attention_plans(model):

    choices = {spec.logical_layer: choice for spec, choice in model.attention_choices}
    plans = tuple(
        dict(
            role=spec.model_role,
            layer=spec.logical_layer,
            resource=spec.resource,
            representation=format,
            storage=state.state_name,
            storage_row=state.get_storage_layer_index(spec.logical_layer),
            writer=getattr(spec, "cache_writer", "upstream"),
            prefill=type(model.attention_prefill_backends[slots[0]]).__name__,
            decode=type(model.attention_decode_backends[slots[1]]).__name__,
            metadata_slots=slots,
            rejected=choices[spec.logical_layer].rejected,
        )
        for layer, state, spec, format, slots in (*model.attention_plans, *model.mla_attention_plans)
    )
    return plans + tuple(
        dict(
            role=model.att_state_role,
            layer=layer,
            resource=ResourceId(model.att_state_role, layer, state.algorithm),
            representation=state.config,
            storage=state.state_name,
            storage_row=row,
            writer="recurrent_update",
            prefill=type(model.attention_prefill_backends[slots[0]]).__name__,
            decode=type(model.attention_decode_backends[slots[1]]).__name__,
            metadata_slots=slots,
        )
        for state, slots in model.recurrent_attention_plans
        for row, layer in enumerate(state.config.logical_layer_ids)
    )


def describe_gqa_format(state):
    format = state.get_attention_format()
    if not isinstance(format, KVFormat):
        raise TypeError(f"{type(state).__name__} is not GQA storage")
    return format


def _shared_write(*, token_indices, kv):
    if kv is not None:
        raise ValueError("a shared KV consumer must not produce a second cache write")


def prepare_model_attention(model):
    manager = model.att_state_manager
    states = manager.get_full_attention_states(model.att_state_role)
    if not states:
        model.prefill_att_backend = model.decode_att_backend = None
    prefill = [model.prefill_att_backend, model.prefill_att_backend1]
    decode = [model.decode_att_backend, model.decode_att_backend1]
    slots = {}
    plans = []
    mla_plans = []
    resources = []
    choices = []
    for layer in model.layers_infer:
        matches = [state for state in states if state.contains_logical_layer(layer.layer_num_)]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"ambiguous full-attention resource for layer {layer.layer_num_}")
        state: FullAttentionState = matches[0]
        resources.append((layer, state))
        format = state.get_attention_format()
        if format is None and not isinstance(state, MLAFullAttentionState):
            continue
        declare = getattr(layer, "get_attention_spec", None)
        if declare is None:
            continue
        row = state.get_storage_layer_index(layer.layer_num_)
        producer = state.config.logical_layer_groups[row][0]
        resource = ResourceId(model.att_state_role, producer)
        if isinstance(state, MLAFullAttentionState):
            spec = MlaSpec(
                model.att_state_role,
                layer.layer_num_,
                resource,
                layer.tp_q_head_num_,
                layer.qk_nope_head_dim,
                layer.qk_rope_head_dim,
                layer.kv_lora_rank,
                layer.v_head_dim,
                model.data_type,
                layer.softmax_scale,
            )
            format = MlaFormat(state.config.latent_dtype, state.config.latent_dim, state.page_size)
        else:
            spec = declare(resource, model.data_type)
        index = layer.attention_backend_index()

        def candidates(format, phase):
            backend = (prefill if phase == "prefill" else decode)[index]
            if backend is None:
                return
            if getattr(model, "_gqa_auto_candidates", False) and index == 0:
                from .create_utils import gqa_backend_candidates

                yield from gqa_backend_candidates(model, phase, backend, index)
            else:
                yield backend

        choice = select_attention(spec, (format,), candidates)
        choices.append((spec, choice))
        phase_slots = []
        for phase, backends, offer in (("prefill", prefill, choice.prefill), ("decode", decode, choice.decode)):
            key = phase, id(offer.backend), offer.metadata_key
            if key not in slots:
                selected = offer.bind(spec)
                if selected in backends:
                    slots[key] = backends.index(selected)
                else:
                    slots[key] = len(backends)
                    backends.append(selected)
            phase_slots.append(slots[key])
        (mla_plans if isinstance(spec, MlaSpec) else plans).append((layer, state, spec, format, tuple(phase_slots)))
    model.attention_plans = tuple(plans)
    model.mla_attention_plans = tuple(mla_plans)
    model.attention_choices = tuple(choices)
    model.attention_resources = tuple(resources)
    planned = {id(layer) for layer, *_ in (*plans, *mla_plans)}
    for phase_index, backends in enumerate((prefill, decode)):
        used = {plan[-1][phase_index] for plan in (*plans, *mla_plans)}
        if any(id(layer) not in planned for layer, _ in resources):
            used.add(0)
        used.add(1)
        for index in range(len(backends)):
            if index not in used:
                backends[index] = None
    recurrent = []
    for state in manager.get_bounded_attention_states(model.att_state_role):
        if not isinstance(state, (GDNState, KDAState)):
            continue
        phase_slots = []
        for phase, backends, backend in (
            ("prefill", prefill, model.prefill_att_backend1),
            ("decode", decode, model.decode_att_backend1),
        ):
            if getattr(backend, "recurrent_family", None) != state.algorithm:
                raise ValueError(f"{state.algorithm} state requires a compatible {phase} implementation")
            selected = backend.for_recurrent_state(state.config, state.mtp_step)
            slot = len(backends) if recurrent else 1
            if slot == len(backends):
                backends.append(selected)
            else:
                backends[slot] = selected
            phase_slots.append(slot)
        recurrent.append((state, tuple(phase_slots)))
    model.recurrent_attention_plans = tuple(recurrent)
    model.attention_prefill_backends = tuple(prefill)
    model.attention_decode_backends = tuple(decode)


def bind_state_resources(manager, role, logical_layers):
    bindings = {layer: {} for layer in logical_layers}
    for state in (*manager.get_full_attention_states(role), *manager.get_bounded_attention_states(role)):
        kind = state.resource_kind
        groups = getattr(state.config, "logical_layer_groups", ())
        for layer, resources in bindings.items():
            if not state.contains_logical_layer(layer):
                continue
            if kind in resources:
                raise ValueError(f"ambiguous {kind} resource for layer {layer}")
            producer = next((group[0] for group in groups if layer in group), layer)
            resources[kind] = ResourceBinding(ResourceId(role, producer, kind), state, state.get_layer_view(layer))
    return bindings


def bind_model_attention(model):
    if not hasattr(model, "attention_plans"):
        prepare_model_attention(model)
    for layer, state in model.attention_resources:
        layer.full_att_state = state
        layer.full_att_view = state.get_layer_view(layer.layer_num_)
        layer.write_full_att_kv = bind_kv_writer(state, layer.layer_num_)
        if isinstance(state, (PackedMLAFullAttentionState, FP8E8M0PackedMLAState)):
            layer.attention = BoundSparse(
                ResourceId(model.att_state_role, layer.layer_num_),
                layer.full_att_view.kv,
                layer.softmax_scale,
                layer.kv_lora_rank,
                getattr(layer, "qk_rope_head_dim", 0),
            )
    for layer, state, spec, format, phase_slots in model.mla_attention_plans:
        layer.attention = BoundMla(spec.resource, layer.full_att_view, spec.softmax_scale, spec.rope_dim, *phase_slots)
    model.bound_states = bind_state_resources(
        model.att_state_manager, model.att_state_role, [layer.layer_num_ for layer in model.layers_infer]
    )
    for layer in model.layers_infer:
        states = model.bound_states[layer.layer_num_]
        layer.state_bindings = states
        bind = getattr(layer, "bind_attention_resources", None)
        if bind is not None:
            bind(states)
        bind = getattr(model.prefill_att_backend, "bind_layer_attention", None)
        if bind is not None:
            layer.attention = bind(layer, states, model.trans_layers_weight[model.layers_infer.index(layer)])
    bindings = {}
    for layer, state, spec, format, phase_slots in model.attention_plans:
        row = state.get_storage_layer_index(layer.layer_num_)
        control = AttControl(
            logical_layer_id=layer.layer_num_,
            use_alibi=spec.alibi,
            use_sliding_window=spec.window != (-1, -1),
            sliding_window=spec.window,
            use_att_sink=spec.attention_sink,
        )
        weight = (
            model.trans_layers_weight[model.layers_infer.index(layer)]
            if hasattr(model, "trans_layers_weight")
            else None
        )
        if spec.alibi:
            control.tp_alibi = weight.tp_alibi
        if spec.attention_sink:
            control.sink_weight = weight.attn_sinks.weight
        if isinstance(state, StaticFP8FullAttentionState):
            control.kv_scales = state.scales[row]
            control.cpu_kv_scales = state.cpu_scales[row]
            control.q_scale = state.get_query_scale(layer.layer_num_)
        writer_backend = model.attention_prefill_backends[phase_slots[0]]
        if hasattr(writer_backend, "bind_cache_writer"):
            layer.write_full_att_kv = writer_backend.bind_cache_writer(layer.full_att_view)
        binding = BoundAttention(
            spec,
            format,
            layer.full_att_view,
            _shared_write if spec.cache_writer == "shared" else layer.write_full_att_kv,
            control,
            *phase_slots,
        )
        layer.write_full_att_kv = binding.write_kv
        layer.attention = binding
        bindings[layer.layer_num_] = binding
    model.attention_bindings = bindings
    recurrent_plans = model.recurrent_attention_plans
    for state, slots in recurrent_plans:
        model.attention_prefill_backends[slots[0]].mtp_state_index = state.mtp_state_index
        model.attention_decode_backends[slots[1]].mtp_state_index = state.mtp_state_index
    for i, layer in enumerate(model.layers_infer):
        matches = [(state, slots) for state, slots in recurrent_plans if state.contains_logical_layer(layer.layer_num_)]
        if len(matches) > 1:
            raise ValueError(f"ambiguous recurrent state for layer {layer.layer_num_}")
        if matches:
            state, slots = matches[0]
            family = "gdn" if isinstance(state, GDNState) else "kda"
            layer.recurrent = BoundRecurrent(
                ResourceId(model.att_state_role, layer.layer_num_, family),
                state.get_layer_view(layer.layer_num_),
                model.trans_layers_weight[i],
                *slots,
            )
