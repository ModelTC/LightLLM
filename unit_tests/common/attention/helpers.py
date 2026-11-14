from lightllm.common.attention.api.environment import BackendEnv
from lightllm.common.attention.api.spec import AttentionFeature, GDNConfig, KDAConfig, MLAConfig, SparseMLAConfig
from lightllm.common.attention.session import AttentionSession
import importlib
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import torch

from lightllm.common.attention.api import (
    AttentionBatch,
    AttentionFamily,
    AttentionPhase,
    CacheLayout,
    ContextCapacity,
    LayerAttention,
    LayerSpec,
)
from lightllm.common.attention.api.workspace import AttentionWorkspace
from lightllm.common.attention.api.backend import AttentionBackend
from lightllm.common.basemodel.attention import describe_serving_constraints
from lightllm.platform import get_device_runtime
from lightllm.server.core.objs.start_args_type import StartArgs


def make_batch(lengths=(5, 2), *, phase=AttentionPhase.DECODE, device="cpu", **kwargs):
    total = sum(lengths)
    maximum = max(lengths) if lengths else 0
    lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    count = lengths.numel()
    q_lengths = torch.ones_like(lengths)
    offsets = torch.arange(count + 1, dtype=torch.int32, device=device)
    values = dict(
        phase=phase,
        batch_size=count,
        token_num=count,
        total_token_num=total,
        max_q_seq_len=1,
        max_kv_seq_len=maximum,
        max_cache_len=0,
        b_req_idx=torch.arange(count, dtype=torch.int32, device=device),
        b_seq_len=lengths,
        b_q_seq_len=q_lengths,
        b1_cu_q_seq_len=offsets,
        b1_cu_kv_seq_len=torch.cat((offsets[:1], lengths.cumsum(0).int())),
        req_to_token_indexs=torch.zeros((max(count, 8), 256), dtype=torch.int32, device=device),
        hold_req_id=7,
        b_mtp_index=torch.zeros_like(lengths),
        b_ready_cache_len=torch.zeros_like(lengths),
    )
    values.update(kwargs)
    return AttentionBatch(**values)


def make_spec(family=AttentionFamily.DENSE, phase=AttentionPhase.DECODE, **kwargs):
    values = dict(
        family=family,
        phase=phase,
        dtype=torch.bfloat16,
        cache=CacheLayout(
            family.value if family in (AttentionFamily.GDN, AttentionFamily.KDA) else "None", 1, torch.bfloat16, 2, 8, 8
        ),
        num_q_heads=4,
        head_size=8,
        value_head_size=8,
    )
    values["config"] = {
        AttentionFamily.GDN: GDNConfig(2, 4, "silu"),
        AttentionFamily.KDA: KDAConfig(4, -5.0),
        AttentionFamily.MLA: MLAConfig(0, 8),
        AttentionFamily.SPARSE_MLA: SparseMLAConfig(2048, 4),
    }.get(family)
    features = set(kwargs.pop("features", ()))
    for key, feature in (
        ("use_mtp", "MTP"),
        ("dynamic_verify", "DYNAMIC_VERIFY"),
        ("use_alibi", "ALIBI"),
        ("use_sink", "SINK"),
    ):
        if kwargs.pop(key, False):
            features.add(AttentionFeature[feature])
    if not kwargs.pop("causal", True):
        features.add(AttentionFeature.NONCAUSAL)
    if kwargs.get("sliding_window") is not None:
        features.add(AttentionFeature.SLIDING_WINDOW)
        if AttentionFeature.MTP in features:
            features.add(AttentionFeature.MTP_WINDOW)
    values.update(kwargs, features=frozenset(features))
    return LayerSpec(**values)


def make_backend(cls=None, **kwargs):
    cls = cls or StubBackend
    backend = cls.__new__(cls)
    values = dict(
        spec=make_spec(cls.family, cls.phase),
        alloc_tensor=lambda shape, dtype, device="cuda": torch.empty(shape, dtype=dtype, device=device),
        page_size=1,
        causal=True,
        dynamic_verify=False,
        draft_step=0,
        state_slots=1,
        is_draft_model=False,
        hold_req_id=7,
    )
    values.update(kwargs)
    if "mtp_step" in kwargs:
        values["state_slots"] = kwargs["mtp_step"] + 1
    page_size = values.pop("kernel_page_size", values.pop("page_size", 1))
    spec = values["spec"]
    features = set(spec.features)
    if not values.pop("causal", True):
        features.add(AttentionFeature.NONCAUSAL)
    if values.pop("dynamic_verify", False):
        features.add(AttentionFeature.DYNAMIC_VERIFY)
    values["spec"] = replace(
        spec,
        cache=replace(
            spec.cache,
            page_size=page_size,
            quant_group_size=values.pop("quant_group_size", spec.cache.quant_group_size),
        ),
        features=frozenset(features),
    )
    if "softmax_scale" in values:
        values["spec"] = replace(values["spec"], softmax_scale=values.pop("softmax_scale"))
    vars(backend).update(values)
    return backend


def make_context(backend, batch=None, *, capture=False, max_length=None, prepare=False):
    batch = make_batch() if batch is None else batch
    capacity = ContextCapacity(
        phase=batch.phase,
        batch_size=batch.batch_size,
        token_num=batch.token_num,
        max_kv_seq_len=max_length or batch.max_kv_seq_len,
        max_q_seq_len=batch.max_q_seq_len,
    )
    backend = phase_backend(backend, batch.phase)
    context = AttentionSession._allocate(backend, capacity, capture=capture, workspace=AttentionWorkspace(batch.device))
    if prepare:
        plan(backend, context, batch)
    else:
        context.batch = batch
        context.current_max_kv_seq_len = batch.max_kv_seq_len
    return context


def make_layer_attention(backend, context, params, **spec_kwargs):
    if not hasattr(backend, "spec"):
        backend.spec = make_spec(backend.family, backend.phase)
    if spec_kwargs:
        backend.spec = replace(backend.spec, **spec_kwargs)
    backend.alloc_tensor = lambda shape, dtype, device="cpu": torch.empty(
        shape, dtype=dtype, device=context.batch.device
    )
    return LayerAttention(
        group_id=0,
        logical_layer_id=0,
        resources=params,
        backend=backend,
    )


def model_stub(**args):
    args.setdefault("disable_cudagraph", not torch.cuda.is_available())
    model = SimpleNamespace(
        args=StartArgs(**args),
        device_runtime=get_device_runtime("cuda" if torch.cuda.is_available() else "cpu"),
        att_state_role="main",
        att_state_manager=SimpleNamespace(
            get_full_attention_states=lambda *args: (), get_bounded_attention_states=lambda *args: ()
        ),
        config={},
        is_mtp_draft_model=False,
        tp_world_size_=1,
        tp_rank_=0,
        max_seq_length=256,
        batch_max_tokens=32,
        layers_infer=[],
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=7),
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0),
    )
    model.attention_requirements = describe_serving_constraints(model)
    return model


def batch_from_fixture(fixture, phase, token_count=None, device=None):
    indices = getattr(fixture, "b_req_idx", None)
    lengths = getattr(fixture, "b_seq_len", None)
    device = device or (
        indices.device if indices is not None else getattr(getattr(fixture, "mem_index", None), "device", "cpu")
    )
    count = indices.numel() if indices is not None else (token_count or getattr(fixture, "batch_size", None) or 1)
    if lengths is None:
        lengths = torch.ones(count, dtype=torch.int32, device=device)
    values = {}
    for name in AttentionBatch.__dataclass_fields__:
        value = getattr(fixture, name, None)
        if value is not None and name not in ("phase", "token_num", "batch_size"):
            values[name] = value
    manager = getattr(fixture, "req_manager", None)
    if manager is not None:
        for name in ("req_to_token_indexs",):
            value = getattr(manager, name, None)
            if value is not None:
                values[name] = value
        values["hold_req_id"] = getattr(manager, "HOLD_REQUEST_ID", 7)

    max_len = getattr(fixture, "max_kv_seq_len", None) or 1
    cache_key = (phase, count, str(device))
    cache = getattr(fixture, "_attention_test_batches", None)
    if cache is None:
        cache = fixture._attention_test_batches = {}
    if cache_key not in cache:
        cache[cache_key] = make_batch((1,) * count, phase=phase, device=device)
    defaults = cache[cache_key]
    values.update(
        phase=phase, batch_size=count, token_num=token_count or count, b_seq_len=lengths, max_kv_seq_len=max_len
    )
    return replace(defaults, **values)


def backend_context(cls, phase, settings=None, **values):
    phase = AttentionPhase(phase)
    if isinstance(settings, AttentionBackend):
        backend = settings
    else:
        config = dict(vars(settings)) if settings is not None else {}
        model = config.pop("model", None)
        if model is not None:
            manager = getattr(model, "mtp_manager", None)
            if manager is not None:
                config["draft_step"] = manager.get_decode_draft_step(getattr(model, "is_mtp_draft_model", False))
            req_manager = getattr(model, "req_manager", None)
            if req_manager is not None:
                config["hold_req_id"] = getattr(req_manager, "HOLD_REQUEST_ID", 7)
        backend = make_backend(cls, **config)
    defaults = dict(
        prefill_graph_enabled=False,
        decode_only=False,
        prefill_token_capacity=0,
        mtp_step=0,
        draft_step=0,
    )
    for name, value in defaults.items():
        if not hasattr(backend, name):
            setattr(backend, name, value)
    context = make_context(backend, make_batch((1,), phase=phase))
    if hasattr(backend, "_test_workspace"):
        context.workspace = backend._test_workspace
    for name, value in values.items():
        setattr(context, name, value)
    return context


def update_context(context, fixture):
    context.batch = batch_from_fixture(fixture, context.capacity.phase)
    context.capacity = ContextCapacity(
        phase=context.batch.phase,
        batch_size=context.batch.batch_size,
        token_num=context.batch.token_num,
        max_kv_seq_len=context.batch.max_kv_seq_len,
        max_q_seq_len=context.batch.max_q_seq_len,
    )
    plan(context.backend_identity, context, context.batch)
    build(context.backend_identity, context)


def run_sparse(context, fixture, query, kv, parameters, *, out=None, alloc_func=None, session=None):
    from lightllm.common.attention.api.inputs import MLAResources, SparseMLADecodeInputs, SparseMLAPrefillInputs

    q = query[0] if isinstance(query, tuple) else query
    context.capture = bool(getattr(fixture, "is_cuda_graph_capture", context.capture))
    if getattr(fixture, "prefill_cuda_graph_exe_list", None) is not None:
        context.capture = True
    context.batch = batch_from_fixture(fixture, context.capacity.phase, q.shape[0], q.device)
    backend = context.backend_identity
    if context.workspace.device != q.device:
        context.workspace = AttentionWorkspace(q.device)
        backend._test_workspace = context.workspace
    if backend.__class__.__module__.startswith("lightllm.models.deepseek_v4"):
        from lightllm.models.deepseek_v4.attention.sparse_mla import DeepseekV4Inputs, DeepseekV4Resources

        params = DeepseekV4Resources(
            kv[0],
            kv[0],
            kv[1],
            parameters["compress_ratio"],
            parameters["head_dim_v"],
            parameters["softmax_scale"],
            parameters["attn_sink"],
        )
        inputs = DeepseekV4Inputs(
            q,
            parameters["swa_indices"],
            parameters["swa_lengths"],
            parameters.get("extra_indices"),
            parameters.get("extra_lengths"),
        )
    else:
        params = MLAResources(kv, parameters["kv_lora_rank"], parameters.get("qk_rope_head_dim", 0))
        if context.capacity.phase is AttentionPhase.PREFILL:
            inputs = SparseMLAPrefillInputs(
                q,
                parameters.get("topk_mem_indices"),
                parameters.get("topk_indices"),
                parameters.get("prefill_cache_kv"),
            )
        else:
            inputs = SparseMLADecodeInputs(q, query[1], parameters["topk_mem_indices"])
    attention = make_layer_attention(backend, context, params, softmax_scale=parameters["softmax_scale"])
    if alloc_func is not None:
        backend.alloc_tensor = alloc_func

    if (
        hasattr(context, "capture_flashmla_sched_meta")
        and context.capture_flashmla_sched_meta is None
        and torch.cuda.is_available()
    ):
        import flash_mla

        context.capture_flashmla_sched_meta = flash_mla.get_mla_metadata()[0]
    if session is not None:
        return attention.forward(session, inputs, out=out)
    return backend.forward(attention, context, inputs, out=out)


def run_dense(context, fixture, query, key, value, *, out=None, alloc_func=None, params=None):
    from lightllm.common.attention.api.inputs import DenseInputs, DenseResources

    context.batch = batch_from_fixture(fixture, context.capacity.phase, query.shape[0], query.device)
    if params is None:
        ks = vs = None
        if isinstance(key, tuple):
            key, ks = key
            value, vs = value
        params = DenseResources(key, value, key_scale=ks, value_scale=vs)
    attention = make_layer_attention(context.backend_identity, context, params)
    if alloc_func is not None:
        context.backend_identity.alloc_tensor = alloc_func
    return context.backend_identity.forward(attention, context, DenseInputs(query), out=out)


class StubBackend(AttentionBackend):
    family = AttentionFamily.DENSE
    provider = "triton"
    phase = AttentionPhase.DECODE

    def allocate_context(self, capacity, *, capture, workspace):
        from lightllm.common.attention.api import AttentionContext

        return AttentionContext(capacity, capture, workspace)

    def forward(self, layer, context, inputs, *, out=None):
        return inputs


def make_env(model=None, **kwargs):
    model = model or model_stub()
    values = dict(
        runtime=model.device_runtime,
        hold_req_id=getattr(model.req_manager, "HOLD_REQUEST_ID", 7),
        draft_step=model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model),
        state_slots=model.args.mtp_step + 1,
        max_seq_len=getattr(model, "max_seq_length", 256),
        tp_size=getattr(model, "tp_world_size_", 1),
        alloc_tensor=lambda shape, dtype, device=model.device_runtime.device: torch.empty(
            shape, dtype=dtype, device=device
        ),
        layer_ids=tuple(layer.layer_num_ for layer in getattr(model, "layers_infer", ())),
    )
    values.update(kwargs)
    return BackendEnv(**values)


def phase_class(cls, phase):
    suffix = "PrefillBackend" if phase is AttentionPhase.PREFILL else "DecodeBackend"
    name = cls.__name__.replace("PrefillBackend", suffix).replace("DecodeBackend", suffix)
    return getattr(importlib.import_module(cls.__module__), name)


def phase_backend(backend, phase):
    if not hasattr(backend, "spec"):
        backend.spec = make_spec(backend.family, backend.phase)
    if backend.phase is not phase:
        cls = phase_class(type(backend), phase)
        backend.__class__ = cls
        backend.spec = replace(backend.spec, phase=phase)
    return backend


def plan(backend, context, batch):
    AttentionSession._check_context(backend, context)
    batch.validate()
    context.capacity.check(batch)
    context.current_max_kv_seq_len = batch.max_kv_seq_len
    if context.capture:
        context.batch = context.batch or replace(
            batch,
            max_kv_seq_len=context.capacity.max_kv_seq_len,
            total_token_num=context.capacity.batch_size * context.capacity.max_kv_seq_len,
        )
        context.batch.copy_inputs(batch)
    else:
        context.batch = batch
    backend.plan_host(context)


def build(backend, context):
    AttentionSession._check_context(backend, context)
    backend.build_device_metadata(context)


@contextmanager
def prepared_session(context, session=None):
    from unittest.mock import patch

    session = session or AttentionSession()
    backend = context.backend_identity

    def allocate(capacity, *, capture, workspace):
        context.workspace = workspace
        return context

    with patch.object(backend, "allocate_context", allocate), patch.object(backend, "plan_host", lambda ctx: None):
        with session.prepare(
            context.batch,
            ({context.capacity.phase: backend},),
            context.capacity,
            workspace=lambda _: context.workspace,
            capture=context.capture,
        ):
            yield session


def run_attention(layer, context, inputs=None, *, out=None):
    with prepared_session(context) as session:
        return layer.forward(session, inputs, out=out)


def dense_class(provider, phase):
    from lightllm.common.attention.registry import BACKENDS, load_backend_class

    return next(
        cls
        for reference in BACKENDS
        for cls in (load_backend_class(reference),)
        if cls.family is AttentionFamily.DENSE
        and cls.phase is phase
        and cls.provider == provider
        and cls.cache_encodings == {"plain"}
        and not cls.variant
    )
