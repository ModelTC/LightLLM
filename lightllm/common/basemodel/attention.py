from dataclasses import dataclass, replace

import torch

from lightllm.common.attention.api import (
    AttentionFamily,
    AttentionPhase,
    CacheLayout,
    LayerBinding,
    LayerSpec,
    PlanCapacity,
)
from lightllm.common.attention.api.execution import ExecutionRequirements
from lightllm.common.attention.api.inputs import DenseParams, GDNParams, KDAParams, MLAParams
from lightllm.common.attention.backends._workspace import AttentionWorkspace


@dataclass(frozen=True)
class AttentionGroup:
    prefill_backend: object
    decode_backend: object
    prefill_spec: LayerSpec
    decode_spec: LayerSpec

    def backend_for(self, phase):
        return self.prefill_backend if phase is AttentionPhase.PREFILL else self.decode_backend


def _alloc_tensor(shape, dtype, device="cuda"):
    from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

    return g_cache_manager.alloc_tensor(shape, dtype, device=device)


def describe_attention_layer(model, layer, family, phase):
    config = model.config
    linear = family in (AttentionFamily.GDN, AttentionFamily.KDA)
    if family is AttentionFamily.GDN:
        heads = config["linear_num_key_heads"] // model.tp_world_size_
        head_dim = config["linear_key_head_dim"]
        value_dim = config["linear_value_head_dim"]
        kv_heads = heads
    elif family is AttentionFamily.KDA:
        heads = config["linear_attn_config"]["num_heads"] // model.tp_world_size_
        head_dim = value_dim = config["linear_attn_config"]["head_dim"]
        kv_heads = heads
    else:
        heads = layer.tp_q_head_num_
        if family in (AttentionFamily.MLA, AttentionFamily.SPARSE_MLA):
            head_dim = getattr(layer, "qk_nope_head_dim", 0) + getattr(layer, "qk_rope_head_dim", 0)
            head_dim = head_dim or layer.head_dim_
            value_dim = getattr(layer, "kv_lora_rank", getattr(layer, "v_head_dim", head_dim))
            kv_heads = 1
        else:
            head_dim = layer.head_dim_
            value_dim = head_dim
            kv_heads = layer.tp_k_head_num_
    encoding = family.value if linear else model.args.llm_kv_type
    ssm_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(model.args.linear_att_ssm_data_type)
    states = tuple(
        state
        for state in (
            *model.att_state_manager.get_full_attention_states(model.att_state_role),
            *model.att_state_manager.get_bounded_attention_states(model.att_state_role),
        )
        if state.contains_logical_layer(layer.layer_num_)
    )
    if not states:
        raise ValueError(f"layer {layer.layer_num_} has no registered attention storage")
    schema = tuple(state.describe_layout() for state in states)
    cache = CacheLayout(
        encoding,
        model.args.page_size,
        getattr(states[0], "storage_dtype", getattr(states[0].config, "dtype", model.data_type)),
        kv_heads,
        head_dim,
        value_dim,
        ssm_dtype if linear else None,
        storage_schema=schema,
        quant_group_size=getattr(states[0].config, "group_size", 0),
        scale_mode=getattr(states[0].config, "scale_mode", "none"),
    )
    draft_step = model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model)
    return LayerSpec(
        family=family,
        phase=phase,
        dtype=model.data_type,
        cache=cache,
        num_q_heads=heads,
        head_size=head_dim,
        value_head_size=value_dim,
        sliding_window=layer.attention_window(phase),
        use_alibi=layer.attention_uses_alibi,
        use_sink=layer.attention_uses_sink,
        causal=not (model.is_mtp_draft_model and model.args.mtp_mode in ("dspark", "dflash", "dflash2")),
        use_mtp=draft_step > 0 and phase is AttentionPhase.DECODE,
        dynamic_verify=not model.is_mtp_draft_model and model.args.mtp_dynamic_verify,
        variant=layer.attention_variant,
    )


def describe_attention_layers(model, family, phase):
    return tuple(
        describe_attention_layer(model, layer, family, phase)
        for layer in model.layers_infer
        if layer.get_attention_family() is family
    )


def make_layer_params(model, layer, weight, family):
    manager, role, layer_id = model.att_state_manager, model.att_state_role, layer.layer_num_
    if family in (AttentionFamily.GDN, AttentionFamily.KDA):
        state = manager.get_state(role, family.value, layer_id)
        conv, ssm = state.get_layer_view(layer_id)
        if family is AttentionFamily.GDN:
            return GDNParams(
                conv,
                ssm,
                weight.linear_conv1d.mm_param.weight,
                weight.linear_conv1d.bias,
                weight.linear_A_log.weight,
                weight.linear_dt_bias.weight,
            )
        return KDAParams(
            conv, ssm, weight.get_merged_kda_conv_weight(), weight.linear_A_log.weight, weight.linear_dt_bias.weight
        )
    if family is AttentionFamily.DENSE:
        state = manager.get_state(role, "gqa", layer_id)
        key, value = layer.attention_kv_view(state)
        query_scale = key_scale = value_scale = None
        if isinstance(key, tuple):
            key, key_scale = key
            value, value_scale = value
        if model.args.llm_kv_type in ("fp8kv_sph", "fp8kv_spt"):
            index = state.get_storage_layer_index(layer_id)
            if model.args.llm_kv_type == "fp8kv_sph":
                heads = state.config.head_num
                key_scale, value_scale = state.scales[index, :heads], state.scales[index, heads:]
                query_scale = state.q_scales[index] if state.q_scales is not None else None
            else:
                key_scale, value_scale = map(float, state.cpu_scales[index])
        return DenseParams(
            key,
            value,
            getattr(weight, "tp_alibi", None),
            layer.attention_window(AttentionPhase.DECODE),
            weight.attn_sinks.weight if layer.attention_uses_sink else None,
            query_scale,
            key_scale,
            value_scale,
        )
    state = manager.get_state(role, "mla", layer_id)
    view = state.get_layer_view(layer_id)
    return MLAParams(
        view.kv if family is AttentionFamily.SPARSE_MLA else view,
        layer.softmax_scale,
        layer.kv_lora_rank,
        getattr(layer, "qk_rope_head_dim", 0),
    )


def initialize_attention_groups(model):
    if model.is_mtp_draft_model and model.args.mtp_mode in ("vanilla_no_att", "eagle_no_att"):
        model.attention_execution = ExecutionRequirements()
        model.attention_groups = ()
        model._attention_layer_groups = (None,) * len(model.layers_infer)
        model._attention_workspaces = {}
        return
    model.attention_execution = ExecutionRequirements.from_model(model)
    caps = model.device_runtime.caps
    for state in (
        *model.att_state_manager.get_full_attention_states(model.att_state_role),
        *model.att_state_manager.get_bounded_attention_states(model.att_state_role),
    ):
        state.check_execution(caps, model.attention_execution)
    from lightllm.common.attention.selection import select_attention_backend

    groups, keys, routes = [], {}, []
    for layer in model.layers_infer:
        family = layer.get_attention_family()
        specs = tuple(describe_attention_layer(model, layer, family, phase) for phase in AttentionPhase)
        if specs[0].cache != specs[1].cache:
            raise ValueError("prefill/decode physical cache layouts differ")
        if specs not in keys:
            backends = []
            for spec in specs:
                candidates, priority = model.attention_backend_candidates(family, spec.phase)
                linear = family in (AttentionFamily.GDN, AttentionFamily.KDA)
                option = f"llm_{spec.phase.value}_{'linear_' if linear else ''}att_backend"
                cls = select_attention_backend(
                    family=family,
                    phase=spec.phase,
                    model=model,
                    requested=getattr(model.args, option),
                    candidates=candidates,
                    priority=priority,
                    specs=(spec,),
                )
                backends.append(cls(model).for_spec(spec))
            keys[specs] = len(groups)
            groups.append(AttentionGroup(*backends, *specs))
        routes.append(keys[specs])
    model.attention_groups = tuple(groups)
    model._attention_layer_groups = tuple(routes)
    model._attention_workspaces = {}


def reserve_attention_resources(model):
    execution = model.attention_execution
    for group_id, group in enumerate(model.attention_groups):
        for phase in AttentionPhase:
            backend = group.backend_for(phase)
            capacity = PlanCapacity(phase, execution.max_batch_size, execution.max_batch_size, execution.max_seq_len, 1)
            resources = backend.resource_requirements(capacity, capture=phase in execution.graph_phases)
            for slot in range(execution.microbatch_count):
                get_attention_workspace(model, group_id, slot, model.device_runtime.device).reserve(resources)
            if resources.zero_initialize_cache:
                for layer, route in zip(model.layers_infer, model._attention_layer_groups):
                    if route != group_id:
                        continue
                    for state in model.att_state_manager.get_full_attention_states(model.att_state_role):
                        if state.contains_logical_layer(layer.layer_num_):
                            state.require_zero_initialization()


def bind_attention_layers(model):
    for layer, weight, group_id in zip(model.layers_infer, model.trans_layers_weight, model._attention_layer_groups):
        if group_id is None:
            continue
        group = model.attention_groups[group_id]
        params = layer.bind_attention_params(model, weight, group.prefill_spec.family)
        prefill_params = (
            replace(params, sliding_window=group.prefill_spec.sliding_window)
            if isinstance(params, DenseParams)
            else params
        )
        layer.prefill_attention = LayerBinding(
            group_id, layer.layer_num_, group.prefill_spec, prefill_params, group.prefill_backend.forward, _alloc_tensor
        )
        layer.decode_attention = LayerBinding(
            group_id, layer.layer_num_, group.decode_spec, params, group.decode_backend.forward, _alloc_tensor
        )
        if group.prefill_spec.family in (AttentionFamily.GDN, AttentionFamily.KDA):
            state = model.att_state_manager.get_state(
                model.att_state_role, group.prefill_spec.family.value, layer.layer_num_
            )
            group.prefill_backend.mtp_state_index = state.mtp_state_index
            group.decode_backend.mtp_state_index = state.mtp_state_index
    model._attention_storage_generation = model.att_state_manager.storage_generation


def get_attention_workspace(model, group_id, slot, device):
    # overlap 的 microbatch 可能同时执行，scratch buffer 不能只按 group 共享。
    key = group_id, slot
    if key not in model._attention_workspaces:
        model._attention_workspaces[key] = AttentionWorkspace(runtime=model.device_runtime)
    return model._attention_workspaces[key]
