from .base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from .create_utils import (
    get_prefill_att_backend_class,
    get_decode_att_backend_class,
    get_mla_prefill_att_backend_class,
    get_mla_decode_att_backend_class,
    get_nsa_prefill_att_backend_class,
    get_nsa_decode_att_backend_class,
    get_neo_prefill_att_backend_class,
)


from contextlib import ExitStack, contextmanager
from types import MappingProxyType

from lightllm.common.attention.api import AttentionBatch, AttentionPhase, ContextCapacity, LayerAttention
from lightllm.common.attention.api.environment import BackendEnv
from lightllm.common.attention.api.execution import ExecutionFailure, ServingConstraints
from lightllm.common.attention.api.workspace import AttentionWorkspace


def describe_serving_constraints(model):
    args = model.args
    if args.enable_prefill_cudagraph and args.enable_prefill_microbatch_overlap:
        raise ValueError("segmented prefill Graph does not support microbatch overlap")
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


class ModelBinding:
    def __init__(self, owners):
        from lightllm.common.attention.api.graph import capture_entries

        self.owners = MappingProxyType(owners)
        self.valid = True
        self.capture_entries = tuple(capture_entries("binding", "shared", self.owners, objects=True))

    def check(self):
        if not self.valid:
            raise RuntimeError("model binding changed; recapture Graph")


class ModelAttention:
    def __init__(self, model):
        from lightllm.common.attention.selection import select_attention_backend

        self.groups = ()
        self.routes = ()
        self.adapters = ()
        self.workspaces = {}
        self.replay_states = ()
        self.generation = None
        self.binding = None
        self.runtime = model.device_runtime
        self.execution = model.execution
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
        self.draft_step = env.draft_step
        self.dynamic_verify = model.args.mtp_dynamic_verify and not model.is_mtp_draft_model
        self.replay_states = tuple(
            state
            for state in model.att_state_manager.get_bounded_attention_states(model.att_state_role)
            if getattr(state, "state_mode", None) == "replay"
        )
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
        self.groups = tuple(MappingProxyType(group) for group in groups)
        self.routes = tuple(routes)
        self.adapters = tuple(adapters)

    def reserve(self, model):
        for group_id, group in enumerate(self.groups):
            for phase, backend in group.items():
                for capture in (False, True) if phase in self.constraints.graph_phases else (False,):
                    capacity = self.reservation_capacity(model, phase, capture=capture)
                    budget = backend.resource_requirements(capacity, capture=capture)
                    for slot in range(self.constraints.microbatch_count):
                        self.workspace(group_id, slot).reserve(budget)
                if backend.zero_initialize_cache:
                    for layer, route in zip(model.layers_infer, self.routes):
                        if route == group_id:
                            for state in model.att_state_manager.get_full_attention_states(model.att_state_role):
                                if state.contains_logical_layer(layer.layer_num_):
                                    state.require_zero_initialization()

    def reservation_capacity(self, model, phase, *, capture):
        if phase is AttentionPhase.PREFILL:
            tokens = max(1, model.batch_max_tokens or model.max_seq_length)
            if model.args.enable_tpsp_mix_mode:
                tokens = ((tokens + model.tp_world_size_ - 1) // model.tp_world_size_) * model.tp_world_size_
            requests = self.constraints.max_batch_size + 1
            query_length = tokens
            kv_length = self.constraints.max_seq_len
        else:
            width = model.mtp_manager.get_decode_tokens_per_request(model.is_mtp_draft_model)
            tokens = self.constraints.max_batch_size * width
            if capture:
                tokens = model.graph_max_batch_size
            if hasattr(model, "_align_decode_batch_size"):
                tokens = model._align_decode_batch_size(tokens)
            requests = tokens
            query_length = max(1, width)
            kv_length = model.graph_max_len_in_batch if capture else self.constraints.max_seq_len
        return ContextCapacity(
            phase=phase,
            batch_size=requests,
            token_num=tokens,
            max_kv_seq_len=kv_length,
            max_q_seq_len=query_length,
        )

    def capacity_memory_bytes(self, model, cache_tokens):
        return (
            sum(
                backend.capacity_memory_bytes(
                    self.reservation_capacity(model, phase, capture=False), cache_tokens=cache_tokens
                )
                for group in self.groups
                for phase, backend in group.items()
            )
            * self.constraints.microbatch_count
        )

    def unbind(self, model):
        if self.binding is not None:
            self.binding.valid = False
        self.binding = None
        self.generation = None
        for layer in model.layers_infer:
            for phase in AttentionPhase:
                name = f"{phase.value}_attention"
                if hasattr(layer, name):
                    delattr(layer, name)
        for workspace in self.workspaces.values():
            workspace.clear_objects()

    def bind(self, model):
        with self.execution.join():
            if self.binding is not None:
                self.binding.valid = False
            owners = {"weights": (model.pre_post_weight, *model.trans_layers_weight)}
            for layer, weight, group_id, adapter in zip(
                model.layers_infer, model.trans_layers_weight, self.routes, self.adapters
            ):
                if group_id is None:
                    continue
                group = self.groups[group_id]
                resources = layer.bind_attention_resources(model, weight, adapter)
                layer._legacy_attention_execution = False
                owners[f"layer.{layer.layer_num_}"] = resources
                for phase, backend in group.items():
                    setattr(
                        layer,
                        f"{phase.value}_attention",
                        LayerAttention(
                            group_id=group_id,
                            logical_layer_id=layer.layer_num_,
                            resources=resources,
                            backend=backend,
                            output_spec=adapter.output_spec,
                        ),
                    )
            self.binding = ModelBinding(owners)
            self.generation = model.att_state_manager.storage_generation

    def finish_decode(self, infer_state):
        if self.replay_states:
            from lightllm.common.attention.backends._shared.linear.layout import decode_request_layout

            reqs, cu = decode_request_layout(infer_state.attention.batch, self.draft_step, self.dynamic_verify)
            for state in self.replay_states:
                state.finish_decode(reqs, cu)

    def workspace(self, group_id, slot):
        key = group_id, slot
        if key not in self.workspaces:
            self.workspaces[key] = AttentionWorkspace(runtime=self.runtime, execution=self.execution)
        return self.workspaces[key]

    @contextmanager
    def session(self, state, *other_states, max_kv_seq_len=None):
        states = (state, *other_states)
        batches = []
        for incoming in states:
            incoming.att_state_manager.check_execution_health()
            if self.generation != incoming.att_state_manager.storage_generation:
                raise RuntimeError("attention storage changed; rebind layers and recapture CUDA graphs")
            batch = build_attention_batch(incoming)
            batch.validate()
            batches.append(batch)
        if len({incoming.microbatch_index for incoming in states}) != len(states):
            raise ValueError("attention microbatch lanes must be distinct")
        prepared = []
        try:
            with self.execution.join(retain=states), ExitStack() as scopes:
                for incoming, batch in zip(states, batches):
                    capacity = ContextCapacity(
                        phase=batch.phase,
                        batch_size=batch.batch_size,
                        token_num=batch.token_num,
                        max_kv_seq_len=batch.max_kv_seq_len if max_kv_seq_len is None else max_kv_seq_len,
                        max_q_seq_len=batch.max_q_seq_len,
                    )
                    scopes.enter_context(
                        incoming.attention.prepare(
                            batch,
                            self.groups,
                            capacity,
                            workspace=lambda group_id, lane=incoming.microbatch_index: self.workspace(group_id, lane),
                            capture=incoming.is_cuda_graph_capture,
                            replay=incoming.is_cuda_graph_replay,
                            retain=(incoming,),
                            binding=getattr(self, "binding", None),
                        )
                    )
                    prepared.append(incoming)
                yield state.attention
        except BaseException as error:
            if any(incoming.attention.state_may_be_modified for incoming in prepared):
                failure = getattr(error, "attention_failure", None)
                if failure is None:
                    failure = ExecutionFailure(
                        state.att_state_role, tuple(batch.b_req_idx for batch in batches), "model forward"
                    )
                state.att_state_manager.invalidate_execution(failure)
                error.attention_failure = failure
            raise

    @contextmanager
    def layer_pass(self, state, *other_states):
        states = (state, *other_states)
        with ExitStack() as passes:
            for incoming in states:
                passes.enter_context(incoming.attention.layer_pass())
            yield
            if not state.is_prefill:
                for incoming in states:
                    self.finish_decode(incoming)
