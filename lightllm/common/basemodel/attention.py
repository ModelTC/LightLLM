from lightllm.common.attention.api import AttentionBatch, AttentionPhase, ContextCapacity, LayerAttention
from lightllm.common.attention.api.environment import BackendEnv
from lightllm.common.attention.api.execution import ServingConstraints
from lightllm.common.attention.api.workspace import AttentionWorkspace


def describe_serving_constraints(model):
    args = model.args
    phases = set()
    if not args.disable_cudagraph:
        phases.add(AttentionPhase.DECODE)
    if args.enable_prefill_cudagraph:
        phases.add(AttentionPhase.PREFILL)
    transfers = set()
    if args.enable_cpu_cache:
        transfers.add("cpu")
    if args.enable_dp_prompt_cache_fetch:
        transfers.add("dp")
    if args.run_mode in ("prefill", "decode"):
        transfers.add("pd")
    return ServingConstraints(
        graph_phases=frozenset(phases),
        microbatch_count=2 if args.enable_decode_microbatch_overlap or args.enable_prefill_microbatch_overlap else 1,
        max_batch_size=getattr(model, "max_req_num", 1),
        max_seq_len=max(getattr(model, "max_seq_length", 1), getattr(model, "graph_max_len_in_batch", 1)),
        expert_parallel=args.enable_ep_moe,
        transfer_paths=frozenset(transfers),
    )


def _alloc_tensor(shape, dtype, device="cuda"):
    from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

    return g_cache_manager.alloc_tensor(shape, dtype, device=device)


def backend_env(model):
    return BackendEnv(
        runtime=model.device_runtime,
        network_config=model.config,
        hold_req_id=model.req_manager.HOLD_REQUEST_ID,
        draft_step=model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model),
        max_seq_len=model.max_seq_length,
        tp_size=model.tp_world_size_,
        state_slots=model.args.mtp_step + 1,
        tp_rank=model.tp_rank_,
        layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
        alloc_tensor=_alloc_tensor,
        prefill_graph=model.args.enable_prefill_cudagraph,
        max_prefill_tokens=model.batch_max_tokens or 0,
        decode_only=model.args.run_mode == "decode",
    )


def build_attention_batch(state):
    return AttentionBatch(
        phase=AttentionPhase.PREFILL if state.is_prefill else AttentionPhase.DECODE,
        batch_size=state.batch_size,
        token_num=state.input_ids.shape[0],
        total_token_num=state.total_token_num,
        max_q_seq_len=state.max_q_seq_len,
        max_kv_seq_len=state.max_kv_seq_len,
        max_cache_len=state.max_cache_len or 0,
        b_req_idx=state.b_req_idx,
        b_seq_len=state.b_seq_len,
        b_q_seq_len=state.b_q_seq_len,
        b1_cu_q_seq_len=state.b1_cu_q_seq_len,
        b1_cu_kv_seq_len=state.b1_cu_kv_seq_len,
        req_to_token_indexs=state.req_manager.req_to_token_indexs,
        hold_req_id=state.req_manager.HOLD_REQUEST_ID,
        b_ready_cache_len=state.b_ready_cache_len,
        b_mtp_index=state.b_mtp_index,
        b_shared_seq_len=state.b_shared_seq_len,
        b_shared_radix_node_id=state.b_shared_radix_node_id,
        mem_index=state.mem_index,
        position_ids=state.position_ids,
        b_image_token_end=getattr(state, "b_image_token_end", None),
    )


class ModelAttention:
    def __init__(self, model):
        from lightllm.common.attention.selection import select_attention_backend

        self.groups = ()
        self.routes = ()
        self.adapters = ()
        self.workspaces = {}
        self.generation = None
        self.runtime = model.device_runtime
        self.constraints = describe_serving_constraints(model)
        if model.is_mtp_draft_model and model.args.mtp_mode in ("vanilla_no_att", "eagle_no_att"):
            self.routes = (None,) * len(model.layers_infer)
            return
        caps = self.runtime.caps
        for state in (
            *model.att_state_manager.get_full_attention_states(model.att_state_role),
            *model.att_state_manager.get_bounded_attention_states(model.att_state_role),
        ):
            state.check_execution(caps, self.constraints)
        groups, keys, routes, adapters = [], {}, [], []
        env = backend_env(model)
        for layer in model.layers_infer:
            specs = []
            adapter = layer.attention_adapter(layer.describe_attention(AttentionPhase.PREFILL).family)
            adapters.append(adapter)
            for phase in AttentionPhase:
                description = layer.describe_attention(phase)
                specs.append(adapter.describe(model, layer, phase, description))
            specs = tuple(specs)
            if specs[0].cache != specs[1].cache:
                raise ValueError("prefill/decode physical cache layouts differ")
            if specs not in keys:
                group = {}
                for spec in specs:
                    option = f"llm_{spec.phase.value}_{'linear_' if spec.family.is_recurrent else ''}att_backend"
                    cls = select_attention_backend(
                        caps=caps,
                        execution=self.constraints,
                        requested=getattr(model.args, option),
                        policy=model.attention_policy(spec.family, spec.phase),
                        specs=(spec,),
                    )
                    group[spec.phase] = cls(env, spec)
                keys[specs] = len(groups)
                groups.append(group)
            routes.append(keys[specs])
        self.groups = tuple(groups)
        self.routes = tuple(routes)
        self.adapters = tuple(adapters)

    def reserve(self, model):
        for group_id, group in enumerate(self.groups):
            for phase, backend in group.items():
                capacity = ContextCapacity(
                    phase=phase,
                    batch_size=self.constraints.max_batch_size,
                    token_num=self.constraints.max_batch_size,
                    max_kv_seq_len=self.constraints.max_seq_len,
                    max_q_seq_len=1,
                )
                budget = backend.resource_requirements(capacity, capture=phase in self.constraints.graph_phases)
                for slot in range(self.constraints.microbatch_count):
                    self.workspace(group_id, slot).reserve(budget)
                if backend.zero_initialize_cache:
                    for layer, route in zip(model.layers_infer, self.routes):
                        if route == group_id:
                            for state in model.att_state_manager.get_full_attention_states(model.att_state_role):
                                if state.contains_logical_layer(layer.layer_num_):
                                    state.require_zero_initialization()

    def bind(self, model):
        for layer, weight, group_id, adapter in zip(
            model.layers_infer, model.trans_layers_weight, self.routes, self.adapters
        ):
            if group_id is None:
                continue
            group = self.groups[group_id]
            resources = layer.bind_attention_resources(model, weight, adapter)
            for phase, backend in group.items():
                setattr(
                    layer,
                    f"{phase.value}_attention",
                    LayerAttention(
                        group_id=group_id,
                        logical_layer_id=layer.layer_num_,
                        resources=resources,
                        backend=backend,
                    ),
                )
        self.generation = model.att_state_manager.storage_generation

    def workspace(self, group_id, slot):
        key = group_id, slot
        if key not in self.workspaces:
            self.workspaces[key] = AttentionWorkspace(runtime=self.runtime)
        return self.workspaces[key]

    def session(self, state, *, max_kv_seq_len=None):
        if self.generation != state.att_state_manager.storage_generation:
            raise RuntimeError("attention storage changed; rebind layers and recapture CUDA graphs")
        batch = build_attention_batch(state)
        capacity = ContextCapacity(
            phase=batch.phase,
            batch_size=batch.batch_size,
            token_num=batch.token_num,
            max_kv_seq_len=batch.max_kv_seq_len if max_kv_seq_len is None else max_kv_seq_len,
            max_q_seq_len=batch.max_q_seq_len,
        )
        return state.attention.prepare(
            batch,
            self.groups,
            capacity,
            workspace=lambda group_id: self.workspace(group_id, state.microbatch_index),
            capture=state.is_cuda_graph_capture,
            replay=state.is_cuda_graph_replay,
        )
