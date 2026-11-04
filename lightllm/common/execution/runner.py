"""Startup assembly and per-batch preparation for bound attention families."""

from .dense import DenseAttentionBackend, DenseBatchMetadata, DenseKVWriter, DenseStaticParams
from .group_quantized import GroupQuantizedAttentionBackend, GroupQuantizedKVWriter
from .static_fp8 import StaticFP8AttentionBackend, StaticFP8KVWriter
from .flashinfer import FlashInferAttentionBackend
from .planner import AttentionRequest, Planner
from .specs import ExecutionLimits, FormatSpec, GatedDeltaNetSpec, MLAAttentionSpec, SparseMLAAttentionSpec
from .state_store import DenseStateStore, ConvSSMStateStore, MLAStateStore, PackedMLAStateStore
from .gdn import GatedDeltaNetBackend, GDNStaticParams
from .mla import MLAAttentionBackend, MLAKVWriter, MLAStaticParams
from .flashinfer_mla import FlashInferMLABackend
from .sparse_mla import SparseMLAAttentionBackend, SparseMLAStaticParams, PackedMLAKVWriter
from .specs import KimiDeltaAttentionSpec, PooledNoPEAttentionSpec
from .kda import KimiDeltaAttentionBackend, KDAStaticParams
from .pooled_sparse import PooledNoPEAttentionBackend, PooledNoPEStaticParams, PooledNoPEKVWriter
from .state_store import IndexerWindowStateStore

RECURRENT_SPECS = (GatedDeltaNetSpec, KimiDeltaAttentionSpec)


def attention_batch(batch):
    return DenseBatchMetadata(
        batch.batch_size,
        batch.max_q_seq_len,
        batch.max_kv_seq_len,
        batch.microbatch_index,
        batch.req_manager.req_to_token_indexs,
        batch.b_req_idx,
        batch.b_seq_len,
        batch.b_q_start_loc,
        batch.b_ready_cache_len,
        batch.b1_cu_q_seq_len,
        batch.b1_cu_kv_seq_len,
        total_kv_tokens=getattr(batch, "total_token_num", None),
        total_q_tokens=batch.input_ids.numel() if getattr(batch, "input_ids", None) is not None else None,
        hold_request_id=getattr(batch.req_manager, "HOLD_REQUEST_ID", None),
        mtp_indices=getattr(batch, "b_mtp_index", None),
    )


class Runner:
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
    ):
        specs = adapter.attention_specs()
        self.specs = specs
        declarations = adapter.state_specs()
        primary = [state for state in declarations if state.retention != "window"]
        windows = [state for state in declarations if state.retention == "window"]
        states = {consumer: state for state in primary for consumer in state.consumers}
        tails = {consumer: state for state in windows for consumer in state.consumers}
        if len(states) != sum(len(state.consumers) for state in primary):
            raise ValueError("a layer must have exactly one primary attention resource")
        if len(tails) != sum(len(state.consumers) for state in windows):
            raise ValueError("a pooled layer must have one indexer tail")
        backends = tuple(DenseAttentionBackend(name, phase) for name in ("fa3", "triton") for phase in requested)
        backends += tuple(GroupQuantizedAttentionBackend(phase) for phase in requested)
        backends += tuple(StaticFP8AttentionBackend(phase) for phase in requested)
        backends += tuple(FlashInferAttentionBackend(phase) for phase in requested)
        non_verify_phases = tuple(phase for phase in requested if phase != "verify")
        backends += tuple(GatedDeltaNetBackend(name, phase) for name in ("flashqla", "triton") for phase in requested)
        backends += tuple(MLAAttentionBackend(name, phase) for name in ("fa3", "triton") for phase in requested)
        backends += tuple(FlashInferMLABackend(phase) for phase in requested)
        backends += tuple(
            SparseMLAAttentionBackend(name, phase)
            for name in ("flashmla_sparse", "triton")
            for phase in non_verify_phases
        )
        backends += tuple(KimiDeltaAttentionBackend(phase) for phase in requested)
        backends += tuple(
            PooledNoPEAttentionBackend(name, phase) for name in ("flashmla_sparse", "triton") for phase in requested
        )
        planner = Planner(backends, runtime.caps, limits)
        requests = []
        for layer in layers:
            i = layer.layer_num_
            recurrent = isinstance(specs[i], RECURRENT_SPECS)
            selection = recurrent_requested if recurrent else requested
            params = (
                static_params[i] if static_params is not None else DenseStaticParams(specs[i], runtime.allocate_tensor)
            )
            for phase, name in selection.items():
                requests.append(
                    AttentionRequest(
                        i,
                        phase,
                        specs[i],
                        states[i],
                        recurrent_format if recurrent else format,
                        params,
                        name,
                        (("indexer_tail", tails[i], window_format),) if i in tails else (),
                    )
                )
        self.prepared = planner.prepare(requests)
        # Allocated before token capacity is profiled, so actual scratch contributes
        # to the same physical memory budget as weights, state and transfer buffers.
        self.workspaces = self.prepared.reserve(runtime.allocate_workspace)
        store_class = {
            "token_latent": MLAStateStore,
            "token_packed_mla": PackedMLAStateStore,
            "token_pooled_nope": PackedMLAStateStore,
        }.get(format.layout, DenseStateStore)
        self.store = store_class([s for s in declarations if s.retention == "history"], format, manager)
        recurrent_states = [s for s in declarations if s.retention == "recurrent"]
        self.recurrent_store = (
            ConvSSMStateStore(
                recurrent_states,
                recurrent_format,
                manager,
                manager._request_capacity,
                runtime.device,
                kind="kda" if any(isinstance(s, KimiDeltaAttentionSpec) for s in specs.values()) else "gdn",
            )
            if recurrent_states
            else None
        )
        self.window_store = (
            IndexerWindowStateStore(windows, window_format, manager, manager._request_capacity, runtime.device)
            if windows
            else None
        )
        self.plan = None
        self.decode_phase = "verify" if "verify" in requested else "decode"

    def bind(self, layers):
        if self.plan is not None:
            raise RuntimeError("bound execution plans cannot be replaced while resources or Graphs may be in use")
        views = self.store.views()
        if self.recurrent_store is not None:
            views.update(self.recurrent_store.views())
        if self.window_store is not None:
            views.update(self.window_store.views())
        plan = self.prepared.bind(views, self.workspaces)
        writer_classes = {
            "plain": DenseKVWriter,
            "int4": GroupQuantizedKVWriter,
            "int8": GroupQuantizedKVWriter,
            "fp8_per_head": StaticFP8KVWriter,
            "fp8_per_tensor": StaticFP8KVWriter,
        }
        writers = {
            view.state.producer: (
                PooledNoPEKVWriter(view)
                if view.format.layout == "token_pooled_nope"
                else PackedMLAKVWriter(view)
                if view.format.layout == "token_packed_mla"
                else MLAKVWriter(view)
                if view.format.layout == "token_latent"
                else writer_classes[view.format.encoding](view)
            )
            for view in views.values()
            if view.state.retention == "history"
        }
        bindings = {(item.layer_id, item.phase): item.execution for item in plan.bindings}
        for layer in layers:
            if layer.layer_num_ not in writers and not isinstance(self.specs[layer.layer_num_], RECURRENT_SPECS):
                raise ValueError("this model recipe does not support a read-only shared KV consumer")
        for layer in layers:
            layer.bound_prefill_attention = bindings.get((layer.layer_num_, "prefill"))
            layer.bound_decode_attention = bindings.get((layer.layer_num_, self.decode_phase))
            if layer.layer_num_ in writers:
                layer.bound_kv_writer = writers[layer.layer_num_]
        self.plan = plan
        self.metadata_sources = {}
        for item in plan.bindings:
            self.metadata_sources.setdefault(item.execution.metadata_key, item.execution)
        return plan

    def prepare_batch(self, batch):
        phase = "prefill" if batch.is_prefill else self.decode_phase
        executions = [execution for execution in self.metadata_sources.values() if execution.phase == phase]
        if not executions:
            return {}
        inputs = attention_batch(batch)
        return {execution.metadata_key: execution.create_metadata(inputs) for execution in executions}


def prepare_runner(model):
    # Explicit opt-in keeps inherited recipes with different semantics on their
    # existing path until their adapters declare those differences.
    adapter = getattr(model, "model_adapter", None)
    if adapter is None:
        return None
    if model.args.llm_kv_type not in ("None", "int4kv", "int8kv", "fp8kv_sph", "fp8kv_spt", "fp8kv_dsa"):
        return None
    phases = adapter.attention_phases
    if len(set(phases)) != len(phases) or any(phase not in ("prefill", "decode") for phase in phases):
        raise ValueError("model attention phases must be unique prefill/decode stages")
    requested = {phase: getattr(model.args, f"llm_{phase}_att_backend")[0] for phase in phases}
    verify_tokens = model.mtp_manager.get_decode_tokens_per_request(model.is_mtp_draft_model)
    if verify_tokens > 1 and "decode" in requested:
        requested["verify"] = requested.pop("decode")
    if any(name not in ("auto", "triton", "fa3", "flashinfer", "flashmla_sparse") for name in requested.values()):
        return None
    specs = adapter.attention_specs()
    mla = any(isinstance(spec, MLAAttentionSpec) for spec in specs.values())
    sparse = any(isinstance(spec, SparseMLAAttentionSpec) for spec in specs.values())
    pooled = any(isinstance(spec, PooledNoPEAttentionSpec) for spec in specs.values())
    if sparse and model.args.llm_kv_type not in ("None", "fp8kv_dsa"):
        raise ValueError("sparse MLA requires plain or fp8kv_dsa storage")
    if not sparse and not pooled and model.args.llm_kv_type == "fp8kv_dsa":
        return None
    if pooled and model.args.llm_kv_type not in ("None", "fp8kv_dsa"):
        raise ValueError("pooled NoPE requires plain or group64/E8M0 packed KV")
    if (
        mla
        and not sparse
        and (
            model.args.llm_kv_type != "None"
            or any(not hasattr(weight, "cc_kv_b_proj_") for weight in model.trans_layers_weight)
        )
    ):
        return None
    from .device_runtime import CudaRuntime

    runtime = CudaRuntime()
    # Draft KV catch-up also consumes the target's expanded verification rows.
    capacity_tokens = max(verify_tokens, model.mtp_manager.get_decode_tokens_per_request(False))
    batch_capacity = max((model.max_req_num + 1) * capacity_tokens, model.graph_max_batch_size)
    if model.args.enable_tpsp_mix_mode:
        batch_capacity = (batch_capacity + model.tp_world_size_ - 1) // model.tp_world_size_ * model.tp_world_size_
    limits = ExecutionLimits(
        runtime.workspace_budget(model.mem_fraction),
        batch_capacity,
        model.get_req_manager_max_seq_len(),
        2 if model.args.enable_decode_microbatch_overlap or model.args.enable_prefill_microbatch_overlap else 1,
        max_verify_tokens=verify_tokens,
    )
    import torch

    encodings = {
        "None": ("plain", model.data_type, ("k", "v"), None),
        "int4kv": ("int4", torch.int8, ("k", "v", "k_scale", "v_scale"), model.data_type),
        "int8kv": ("int8", torch.int8, ("k", "v", "k_scale", "v_scale"), model.data_type),
        "fp8kv_sph": ("fp8_per_head", torch.float8_e4m3fn, ("k", "v", "kv_scale", "q_scale"), torch.float32),
        "fp8kv_spt": ("fp8_per_tensor", torch.float8_e4m3fn, ("k", "v", "kv_scale", "q_scale"), torch.float32),
        "fp8kv_dsa": ("fp8_group128", torch.uint8, ("latent", "indexer"), torch.float32),
    }
    encoding, dtype, fields, scale_dtype = encodings[model.args.llm_kv_type]
    calibration = None
    if encoding in ("fp8_per_head", "fp8_per_tensor"):
        from lightllm.common.attention_state.calibration import calibration_table_identity

        calibration = calibration_table_identity(
            model.att_state_manager.read_full_att_calibration(model.args), encoding.removeprefix("fp8_")
        )
    format = FormatSpec(
        encoding,
        dtype,
        "token_latent" if mla else "token_head_dim",
        ("latent",) if mla else fields,
        model.args.page_size,
        (model.tp_rank_, model.tp_world_size_),
        calibration=calibration,
        group_size=model.args.llm_kv_quant_group_size if encoding in ("int4", "int8") else None,
        scale_dtype=scale_dtype,
    )
    if sparse:
        format = FormatSpec(
            encoding,
            dtype,
            "token_packed_mla",
            ("latent", "indexer"),
            model.args.page_size,
            (model.tp_rank_, model.tp_world_size_),
            group_size=128 if encoding == "fp8_group128" else None,
            scale_dtype=scale_dtype,
            field_dtypes=(dtype, torch.uint8),
        )
    window_format = None
    if pooled:
        quantized = model.args.llm_kv_type == "fp8kv_dsa"
        dtype = torch.uint8 if quantized else model.data_type
        format = FormatSpec(
            "fp8_group64_e8m0" if quantized else "plain",
            dtype,
            "token_pooled_nope",
            ("latent", "indexer"),
            model.args.page_size,
            (model.tp_rank_, model.tp_world_size_),
            group_size=64 if quantized else None,
            scale_dtype=torch.uint8 if quantized else None,
            field_dtypes=(dtype, torch.uint8),
        )
        window_format = FormatSpec(
            "plain",
            model.data_type,
            "request_indexer_window",
            ("tail",),
            1,
            (model.tp_rank_, model.tp_world_size_),
            candidate_slots=capacity_tokens,
        )
    recurrent_format, recurrent_requested = None, None
    if any(isinstance(spec, RECURRENT_SPECS) for spec in specs.values()):
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        recurrent_format = FormatSpec(
            "conv_ssm",
            model.data_type,
            "request_conv_ssm",
            ("conv", "ssm", "accepted_index") if verify_tokens > 1 else ("conv", "ssm"),
            1,
            (model.tp_rank_, model.tp_world_size_),
            field_dtypes=(model.data_type, get_torch_dtype(model.args.linear_att_ssm_data_type))
            + ((torch.int32,) if verify_tokens > 1 else ()),
            candidate_slots=verify_tokens,
        )
        recurrent_requested = {}
        for phase in requested:
            names = getattr(model.args, f"llm_{'decode' if phase == 'verify' else phase}_att_backend")
            recurrent_requested[phase] = names[1] if len(names) > 1 else "auto"
    weights = {
        recipe.layer_num_: weight for recipe, weight in zip(model.layers_infer, model.trans_layers_weight, strict=True)
    }
    params = {
        i: GDNStaticParams.from_weights(spec, weights[i])
        if isinstance(spec, GatedDeltaNetSpec)
        else KDAStaticParams.from_weights(spec, weights[i])
        if isinstance(spec, KimiDeltaAttentionSpec)
        else PooledNoPEStaticParams(spec, runtime.allocate_tensor, weights[i].index_kpool_compress_ape.weight)
        if isinstance(spec, PooledNoPEAttentionSpec)
        else SparseMLAStaticParams(spec, runtime.allocate_tensor)
        if isinstance(spec, SparseMLAAttentionSpec)
        else MLAStaticParams(spec, runtime.allocate_tensor, weights[i].cc_kv_b_proj_.mm)
        if isinstance(spec, MLAAttentionSpec)
        else DenseStaticParams(
            spec, runtime.allocate_tensor, weights[i].attn_sinks.weight if spec.attention_sink else None
        )
        for i, spec in specs.items()
    }
    return Runner(
        adapter,
        model.layers_infer,
        model.att_state_manager,
        runtime,
        limits,
        format,
        requested,
        recurrent_format,
        recurrent_requested,
        params,
        window_format,
    )
