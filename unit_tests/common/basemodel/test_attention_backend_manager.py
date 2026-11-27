"""Model initialization selects layer implementations and builds each group's attention state once per batch."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention import selector
from lightllm.common.attention.backend import AttentionBackend
from lightllm.common.attention.backends.gqa import GQAKVScales, GQALayerConfig
from lightllm.common.basemodel import attention_backend_manager
from lightllm.common.basemodel.attention_backend_manager import (
    AttentionBackendManager,
    LayerAttentionConfig,
)
from lightllm.common.attention.backends.gqa import gqa_kv_quant_fields
from lightllm.common.basemodel.infer_struct import InferStateInfo
from unit_tests.common.attention.backends.groups_runner import init_backend


def start_args(**overrides):
    args = dict(
        llm_prefill_att_backend=["triton"],
        llm_decode_att_backend=["triton"],
        mtp_step=0,
        enable_ep_moe=False,
        page_size=16,
        flashinfer_workspace_size=64,
        running_max_req_size=8,
        enable_tpsp_mix_mode=False,
        enable_decode_microbatch_overlap=False,
        diverse_mode=False,
        llm_kv_type="None",
        ssm_state_mode="native",
        linear_att_ssm_data_type="float32",
    )
    return SimpleNamespace(**{**args, **overrides})


def fake_model(**args):
    return SimpleNamespace(
        args=start_args(**args),
        max_seq_length=1024,
        graph_max_batch_size=4,
        graph_max_len_in_batch=1024,
        tp_world_size_=1,
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0),
        att_state_role="main",
        att_state_manager=SimpleNamespace(
            get_full_attention_states=lambda role: (object(),), get_bounded_attention_states=lambda role: ()
        ),
        layers_infer=[],
    )


class ConfigLayer:
    def __init__(self, layer_num, config):
        self.layer_num_, self.config = layer_num, config

    def attention_config(self, model):
        return self.config


def registered(model, manager=None):
    """Register the model's layers in a new manager, or in the given manager shared with other roles."""
    model.attention_backend_manager = manager or AttentionBackendManager()
    model.attention_backend_manager.register_backends(model)
    return model.attention_backend_manager


def register(model, *configs):
    """Register one layer per LayerAttentionConfig and return each layer's (prefill, decode) backends."""
    model.layers_infer = [ConfigLayer(i, config) for i, config in enumerate(configs)]
    return layer_backends(registered(model), model.att_state_role, range(len(configs)))


def layer_backends(manager, role, layer_ids):
    return [tuple(manager._backends.get((role, phase, i)) for phase in selector.PHASES) for i in layer_ids]


def backends(model, phase):
    groups = model.attention_backend_manager.state_groups[model.att_state_role][phase]
    return tuple(type(backend) for backend in groups.values())


def gqa(**fields) -> GQALayerConfig:
    return GQALayerConfig(**{"num_heads": 8, "num_kv_heads": 2, "head_dim": 64, "dtype": torch.bfloat16, **fields})


def test_triton_layers_share_one_group_across_layer_configs(monkeypatch):
    # Gemma4-like: layers differ in head_dim and window, but Triton's step builders ignore the layer config.
    groups = fake_model()
    sliding_config = gqa(sliding_window=(511, 0))
    full_config = LayerAttentionConfig(gqa(head_dim=128), gqa(head_dim=128))
    sliding, full, again = register(
        groups, LayerAttentionConfig(sliding_config, sliding_config), full_config, full_config
    )
    assert full == again
    prepared_by = []
    create_att_state = AttentionBackend.create_att_state

    def record_create_att_state(self, infer_state):
        prepared_by.append(self)
        return create_att_state(self, infer_state)

    monkeypatch.setattr(AttentionBackend, "create_att_state", record_create_att_state)
    for is_prefill, representative, other in zip((True, False), sliding, full):
        assert representative is not other
        state = step(is_prefill)
        state.att_states = groups.attention_backend_manager.create_att_state(state)
        assert list(state.att_states) == [representative.state_key]
        assert state.att_states[representative.state_key] is state.att_states[other.state_key]
    assert prepared_by == list(sliding)


@pytest.mark.parametrize("prefill", [True, False])
def test_layer_attention_state_belongs_to_its_microbatch(prefill):
    model = fake_model()
    (layer,) = register(model, LayerAttentionConfig(gqa(), gqa()))
    backend = layer[0 if prefill else 1]
    first = step(prefill)
    first.att_states = model.attention_backend_manager.create_att_state(first)
    second = step(prefill)
    second.microbatch_index = 1
    second.att_states = model.attention_backend_manager.create_att_state(second)
    state = first.att_states[backend.state_key]
    assert state is not second.att_states[backend.state_key]
    assert state.b_req_idx is first.b_req_idx
    assert second.att_states[backend.state_key].b_req_idx is second.b_req_idx


def test_flashinfer_layers_with_different_head_dims_form_separate_groups():
    groups = fake_model(llm_prefill_att_backend=["flashinfer"], llm_decode_att_backend=["flashinfer"])
    register(groups, LayerAttentionConfig(gqa(), gqa()), LayerAttentionConfig(gqa(head_dim=128), gqa(head_dim=128)))
    assert all(len(backends(groups, phase)) == 2 for phase in selector.PHASES)


@pytest.mark.parametrize("backend, groups", [("flashinfer", 2), ("fa3", 1), ("triton", 1)])
def test_mla_layers_group_by_what_their_step_builders_plan_with(backend, groups):
    from lightllm.common.attention.backends.mla import MLADecodeLayerConfig, MLAPrefillLayerConfig

    def mla(scale):
        common = dict(num_heads=16, qk_rope_head_dim=64, softmax_scale=scale, dtype=torch.bfloat16)
        return (
            MLAPrefillLayerConfig(qk_nope_head_dim=128, v_head_dim=128, **common),
            MLADecodeLayerConfig(kv_lora_rank=512, **common),
        )

    model = fake_model(llm_prefill_att_backend=[backend], llm_decode_att_backend=[backend])
    register(model, *(LayerAttentionConfig(*mla(scale)) for scale in (0.1, 0.1, 0.2)))
    assert all(len(backends(model, phase)) == groups for phase in selector.PHASES)


def test_sparse_mla_step_builders_follow_the_layer_selection():
    from lightllm.common.attention.backends.sparse_mla import SparseMLALayerConfig
    from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLADecodeBackend
    from lightllm.common.attention.backends.sparse_mla.flashmla import FlashMLASparseDecodeBackend

    def config(**fields):
        fields = {"num_heads": 64, "kv_lora_rank": 512, "qk_rope_head_dim": 0, "softmax_scale": 0.1, **fields}
        return SparseMLALayerConfig(dtype=torch.bfloat16, **fields)

    backend = FA3SparseMLADecodeBackend(config(topk=2048, kpool=4, causal=False))
    assert backend.state_key[1] == (2048, 4, False)
    with pytest.raises(NotImplementedError, match="topk"):
        FA3SparseMLADecodeBackend(config())
    # 2048 pooled entries plus at most three kpool remainder entries bound the direct decode window.
    fp8 = dict(kv_cache_format="fp8_group64_e8m0", topk=2048)
    assert FlashMLASparseDecodeBackend(config(kpool=4, **fp8)).direct_decode_bound == 2051
    assert FlashMLASparseDecodeBackend(config(kpool=1, **fp8)).direct_decode_bound is None


def test_layer_family_selects_its_command_line_argument():
    from lightllm.common.attention.backends.gdn import GDNLayerConfig

    groups = fake_model(llm_prefill_att_backend=["fa3", "flashqla"])
    assert selector.requested_backend(groups.args, "prefill", gqa()) == "fa3"
    linear = GDNLayerConfig(2, 4, 64, 64, 4, "silu", torch.bfloat16)
    assert selector.requested_backend(groups.args, "prefill", linear) == "flashqla"
    assert selector.requested_backend(groups.args, "decode", linear) == "auto"


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("prefill_kernel, flash_len", [("triton", None), ("flashkda", None), ("auto", 2048)])
def test_kda_prefill_step_builders_follow_the_layer_config(monkeypatch, replay, prefill_kernel, flash_len):
    from lightllm.common.attention.backends.kda import KDALayerConfig
    from lightllm.common.attention.backends.kda import triton as kda_triton

    requests = []
    monkeypatch.setattr(kda_triton, "import_flash_kda_fwd", lambda: "fwd")
    monkeypatch.setattr(
        kda_triton, "load_flash_kda_fwd", lambda requested, **kwargs: requests.append((requested, kwargs)) or "fwd"
    )
    config = KDALayerConfig(
        4,
        128,
        4,
        -5.0,
        torch.bfloat16,
        state_mtp_step=2,
        replay=replay,
        ssm_dtype=torch.bfloat16,
        prefill_kernel=prefill_kernel,
    )
    backend = kda_triton.ReplayKDAPrefillBackend if replay else kda_triton.TritonKDAPrefillBackend
    backend = init_backend(backend, config, step(prefill=True))
    assert requests == [(prefill_kernel, {"dtype": torch.bfloat16, "ssm_dtype": torch.bfloat16})]
    assert backend.flash_kda_fwd == "fwd" and backend.max_flash_len == flash_len
    req = torch.tensor([0, 1, 2])
    att_state = backend.build(
        SimpleNamespace(
            b_req_idx=req,
            b1_cu_q_seq_len=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            b_ready_cache_len=torch.zeros(3, dtype=torch.int32),
            max_q_seq_len=1,
        )
    )
    assert att_state.ssm_idx.tolist() == (req * (1 if replay else 3)).tolist()
    assert att_state.flash_kda_fwd == "fwd"


class Built(SimpleNamespace):
    pass


class FakeConfig(SimpleNamespace):
    def __hash__(self):
        return hash(tuple(sorted(vars(self).items())))


class FakeBuilder:
    def __init__(self, backend):
        self.backend, self.batches = backend, []

    def build(self, batch):
        self.batches.append(batch)
        return Built(backend=self.backend, batch=batch)


class FakeImpl(AttentionBackend):
    def forward(self, att_state, layer_inputs, *, out, alloc_func=torch.empty):
        return out


created = []


def fake_backend(backend_name, *, sync=False):
    class Backend(FakeImpl):
        name = backend_name
        phase = backend_name.rsplit("_", 1)[1]

        def init_runtime(self, infer_state):
            created.append((backend_name, self.state_key[1]))
            self.builder = FakeBuilder(backend_name)

        def build(self, infer_state):
            return self.builder.build(infer_state)

        @classmethod
        def group_key(cls, layer_config):
            return layer_config.key

        requires_cpu_cache_sync = sync

    return Backend


FakePrefill = fake_backend("fake_prefill")
FakeDecode = fake_backend("fake_decode", sync=True)


@pytest.fixture
def fake_family(monkeypatch):
    created.clear()
    backends = {"prefill": {"fake": (f"{__name__}:FakePrefill",)}, "decode": {"fake": (f"{__name__}:FakeDecode",)}}
    monkeypatch.setitem(selector._REGISTRY, FakeConfig, selector.Family("fake", backends))
    return fake_model(llm_prefill_att_backend=["fake"], llm_decode_att_backend=["fake"])


def step(prefill: bool, batch_size: int = 2) -> InferStateInfo:
    state = InferStateInfo()
    state.is_prefill = prefill
    state.batch_size, state.total_token_num, state.max_kv_seq_len = batch_size, 4 * batch_size, 4
    state.b_req_idx = torch.arange(batch_size, dtype=torch.int32)
    state.b_seq_len = torch.full((batch_size,), 4, dtype=torch.int32)
    state.req_manager = SimpleNamespace(req_to_token_indexs=torch.zeros(batch_size, 8, dtype=torch.int32))
    return state


def test_step_builders_are_created_per_group_and_phase_on_first_build(fake_family):
    groups = fake_family
    (first_prefill, first_decode), (second_prefill, second_decode) = register(
        groups,
        LayerAttentionConfig(FakeConfig(key=1), FakeConfig(key=1)),
        LayerAttentionConfig(FakeConfig(key=2), FakeConfig(key=1, other=2)),
    )
    assert created == []

    prefill = step(prefill=True)
    prefill.att_states = groups.attention_backend_manager.create_att_state(prefill)
    # A prefill-only node never creates decode builders.
    assert sorted(created) == [("fake_prefill", 1), ("fake_prefill", 2)]
    assert set(prefill.att_states) == {first_prefill.state_key, second_prefill.state_key}
    assert prefill.att_states[first_prefill.state_key] is not prefill.att_states[second_prefill.state_key]
    # Every group's builder saw the same batch, built once for the step.
    batches = {id(args.batch) for args in prefill.att_states.values()}
    assert len(batches) == 1

    decode = step(prefill=False)
    decode.att_states = groups.attention_backend_manager.create_att_state(decode)
    first_args = decode.att_states[first_decode.state_key]
    assert first_decode is not second_decode
    assert first_args is decode.att_states[second_decode.state_key]
    overlap = step(prefill=False, batch_size=3)
    overlap.microbatch_index = 1
    overlap.att_states = groups.attention_backend_manager.create_att_state(overlap)
    assert overlap.att_states[first_decode.state_key] is overlap.att_states[second_decode.state_key]
    assert overlap.att_states[first_decode.state_key].batch.microbatch_index == 1
    assert decode.att_states[first_decode.state_key] is first_args
    assert first_args.batch.batch_size == 2
    assert sorted(created) == [("fake_decode", 1), ("fake_prefill", 1), ("fake_prefill", 2)]
    assert groups.attention_backend_manager.requires_cpu_cache_sync


def test_decode_only_layers_build_no_prefill_att_state(fake_family):
    groups = fake_family
    ((prefill_backend, decode_backend),) = register(groups, LayerAttentionConfig(None, FakeConfig(key=1)))
    assert prefill_backend is None
    assert backends(groups, "prefill") == () and backends(groups, "decode") == (FakeDecode,)

    prefill = step(prefill=True)
    prefill.att_states = groups.attention_backend_manager.create_att_state(prefill)
    assert prefill.att_states == {} and created == []

    decode = step(prefill=False)
    decode.att_states = groups.attention_backend_manager.create_att_state(decode)
    assert set(decode.att_states) == {decode_backend.state_key} and created == [("fake_decode", 1)]


def test_layer_override_is_independent_for_identical_configs():
    groups = fake_model(llm_prefill_att_backend=["fa3"], llm_decode_att_backend=["fa3"])
    config = gqa()
    pinned, default, prefill_pinned, decode_pinned = register(
        groups,
        LayerAttentionConfig(config, config, prefill_backend_name="triton", decode_backend_name="triton"),
        LayerAttentionConfig(config, config),
        LayerAttentionConfig(config, config, prefill_backend_name="triton"),
        LayerAttentionConfig(config, config, decode_backend_name="triton"),
    )
    assert prefill_pinned == (pinned[0], default[1])
    assert decode_pinned == (default[0], pinned[1])
    for phase in selector.PHASES:
        assert [backend.name for backend in backends(groups, phase)] == [
            f"triton_gqa_{phase}",
            f"fa3_gqa_{phase}",
        ]


def test_models_without_layers_build_no_att_state(fake_family):
    state = step(prefill=True)
    state.att_states = registered(fake_family).create_att_state(state)
    assert state.att_states == {}


def test_roles_share_the_manager_but_not_backend_instances():
    config = LayerAttentionConfig(gqa(), gqa())
    main = fake_model()
    ((main_prefill, main_decode),) = register(main, config)
    draft = fake_model()
    draft.att_state_role, draft.is_mtp_draft_model = "draft0", True
    draft.layers_infer = [ConfigLayer(0, config)]
    manager = registered(draft, main.attention_backend_manager)
    ((draft_prefill, draft_decode),) = layer_backends(manager, "draft0", [0])
    # Each instance keeps its own role's cross-step and CUDA graph storage.
    assert draft_prefill is not main_prefill and draft_decode is not main_decode
    assert manager.get_backend("main", "decode", 0) is main_decode
    assert manager.get_backend("draft0", "decode", 0) is draft_decode
    with pytest.raises(RuntimeError, match="already registered"):
        manager.register_backends(draft)
    # Each step builds its own role's states.
    state = step(prefill=False)
    state.att_state_role = "draft0"
    state.att_states = manager.create_att_state(state)
    assert list(state.att_states) == [draft_decode.state_key]


@pytest.mark.parametrize(
    "kv_type, scales, expected",
    [
        ("None", None, dict(kv_quant=None, kv_quant_group_size=None, has_q_scale=False)),
        (
            "int8kv",
            GQAKVScales(torch.ones(4, 2, 8), torch.ones(4, 2, 8)),
            dict(kv_quant="int8_group", kv_quant_group_size=8, has_q_scale=False),
        ),
        (
            "int4kv",
            GQAKVScales(torch.ones(4, 2, 2), torch.ones(4, 2, 2)),
            dict(kv_quant="int4_group", kv_quant_group_size=32, has_q_scale=False),
        ),
        (
            "fp8kv_spt",
            GQAKVScales(0.5, 0.5),
            dict(kv_quant="fp8_per_tensor", kv_quant_group_size=None, has_q_scale=False),
        ),
        (
            "fp8kv_sph",
            GQAKVScales(torch.ones(2), torch.ones(2)),
            dict(kv_quant="fp8_per_head", kv_quant_group_size=None, has_q_scale=False),
        ),
        (
            "fp8kv_sph",
            GQAKVScales(torch.ones(2), torch.ones(2), torch.ones(2)),
            dict(kv_quant="fp8_per_head", kv_quant_group_size=None, has_q_scale=True),
        ),
    ],
)
def test_kv_quant_fields_follow_the_kv_type_and_the_cache_scales(kv_type, scales, expected):
    fields = gqa_kv_quant_fields(kv_type, scales, head_dim=64)
    assert fields == expected
    GQALayerConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, **fields)


def test_kv_quant_fields_reject_mismatched_caches():
    with pytest.raises(ValueError, match="does not match"):
        gqa_kv_quant_fields("int8kv", None, head_dim=64)
    with pytest.raises(ValueError, match="no GQA KV cache layout"):
        gqa_kv_quant_fields("fp8kv_dsa", None, head_dim=64)


def test_fa3_decode_page_tables_allocate_lazily_per_microbatch_and_check_capacity(monkeypatch):
    from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeBackend
    from lightllm.common.basemodel.triton_kernel import fa3_utils

    requests = []

    def get_workspace(name, numel, dtype):
        requests.append((name, numel, dtype))
        return torch.zeros(numel, dtype=dtype)

    monkeypatch.setattr(fa3_utils, "page_table_copy", lambda **kwargs: None)
    ctx = step(prefill=False)
    ctx.page_size, ctx.model_max_seq_len, ctx.max_decode_rows, ctx.max_cuda_graph_batch_size = 16, 1024, 8, 4
    manager = AttentionBackendManager()
    monkeypatch.setattr(manager, "get_workspace", get_workspace)
    backend = init_backend(FA3GQADecodeBackend, gqa(), ctx)
    assert requests == []

    def decode_step(rows, max_kv_len, microbatch_index):
        state = step(prefill=False, batch_size=rows)
        state.attention_backend_manager = manager
        state.model_max_seq_len = ctx.model_max_seq_len
        state.max_decode_rows = ctx.max_decode_rows
        state.microbatch_index = microbatch_index
        state.max_kv_seq_len = max_kv_len
        state.b_seq_len.fill_(max_kv_len)
        state.b1_cu_q_seq_len = torch.arange(rows + 1, dtype=torch.int32)
        state.b1_cu_kv_seq_len = torch.arange(rows + 1, dtype=torch.int32) * max_kv_len
        return state

    view = backend.build(decode_step(rows=3, max_kv_len=100, microbatch_index=1)).page_table
    assert view.shape == (3, 7) and view.is_contiguous()
    rows = max(ctx.max_decode_rows, ctx.max_cuda_graph_batch_size)
    # Only the step's microbatch buffer is requested; the workspace shares it by name.
    assert requests == [("fa3_page_table_1", rows * 64, torch.int32)]
    with pytest.raises(RuntimeError, match="batch size"):
        backend.build(decode_step(rows=rows + 1, max_kv_len=16, microbatch_index=0))
    with pytest.raises(RuntimeError, match="max KV sequence length"):
        backend.build(decode_step(rows=1, max_kv_len=ctx.model_max_seq_len + 1, microbatch_index=0))


class FakeState:
    def __init__(self, kv_heads=2, scales=None):
        self.k = torch.empty(16, kv_heads, 64)
        self.scales = scales

    def get_layer_view(self, logical_layer_id):
        return self.k, self.k

    def get_layer_kv_scales(self, logical_layer_id):
        return self.scales


def llama_layer():
    from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer

    return LlamaTransformerLayerInfer(
        3, {"rms_norm_eps": 1e-6, "num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512}
    )


def model_with_state(state, **args):
    model = fake_model(**args)
    model.data_type = torch.bfloat16
    model.att_state_role = "draft0"
    model.att_state_manager = SimpleNamespace(
        get_state=lambda **query: state,
        get_full_attention_states=lambda role: (state,),
        get_bounded_attention_states=lambda role: (),
    )
    return model


def test_llama_layer_creates_its_attention_from_its_cache(runtime_start_args):
    scales = GQAKVScales(torch.ones(16, 2, 8), torch.ones(16, 2, 8))
    model = model_with_state(FakeState(scales=scales), llm_kv_type="int8kv")
    model.layers_infer = [llama_layer()]
    ((prefill, decode),) = layer_backends(registered(model), "draft0", [3])
    expected = gqa(kv_quant="int8_group", kv_quant_group_size=8)
    assert prefill.layer_config == decode.layer_config == expected
    assert backends(model, "decode")[0].name == "triton_quant_gqa_decode"


def test_only_prefill_attends_inside_image_spans(runtime_start_args):
    from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import NeoChatTransformerLayerInfer

    layer = NeoChatTransformerLayerInfer(
        3,
        {"rms_norm_eps": 1e-6, "num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512, "head_dim": 64},
    )
    model = model_with_state(FakeState())
    assert layer.attention_config(model) == LayerAttentionConfig(gqa(bidirectional_image_spans=True), gqa())


def test_block_draft_layers_are_not_causal(monkeypatch):
    monkeypatch.setattr(attention_backend_manager, "get_env_start_args", lambda: SimpleNamespace(mtp_mode="dflash"))
    model = model_with_state(FakeState(), llm_prefill_att_backend=["fa3"], llm_decode_att_backend=["fa3"])
    model.is_mtp_draft_model = True
    layer = llama_layer()
    assert layer.attention_config(model).decode == gqa(causal=False)


def test_unsupported_layer_configs_fail_when_the_model_loads(runtime_start_args):
    # FlashInfer cannot run the static FP8 per-head cache that only FA3 supports.
    scales = GQAKVScales(torch.ones(2), torch.ones(2))
    model = model_with_state(
        FakeState(scales=scales),
        llm_kv_type="fp8kv_sph",
        llm_prefill_att_backend=["flashinfer"],
        llm_decode_att_backend=["flashinfer"],
    )
    model.layers_infer = [llama_layer()]
    with pytest.raises(ValueError, match="does not support"):
        registered(model)


def test_models_with_attention_states_create_their_layers_attention():
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

    class Layer:
        def __init__(self):
            self.models = []

        def attention_config(self, model):
            self.models.append(model)
            return LayerAttentionConfig(None, gqa())

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.__dict__.update(vars(fake_model()))
    model.is_mtp_draft_model = False
    model.layers_infer = [Layer(), Layer()]
    for i, layer in enumerate(model.layers_infer):
        layer.layer_num_ = i
    states = [object()]
    model.att_state_manager = SimpleNamespace(
        get_full_attention_states=lambda role: tuple(states), get_bounded_attention_states=lambda role: ()
    )
    manager = model.attention_backend_manager = AttentionBackendManager()

    model._init_attention()
    assert manager.has_attention("main")
    assert all(layer.models == [model] for layer in model.layers_infer)
    (first_prefill, first_decode), (second_prefill, second_decode) = layer_backends(manager, "main", [0, 1])
    assert first_prefill is second_prefill is None
    assert first_decode is second_decode is not None

    # A draft without attention states (FFN-only) registers no backends and builds no attention state.
    states.clear()
    model.is_mtp_draft_model, model.mtp_previous_draft_models = True, []
    model._init_attention()
    assert manager.has_attention("main") and not manager.has_attention("draft0")


@pytest.mark.parametrize("ssm_state_mode, decode", [("native", "triton_gdn_decode"), ("replay", "replay_gdn_decode")])
def test_qwen3next_layers_split_into_full_and_linear_attention_groups(runtime_start_args, ssm_state_mode, decode):
    from lightllm.common.attention.backends.gdn import GDNLayerConfig
    from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerInfer

    net = {
        "rms_norm_eps": 1e-6,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 512,
        "full_attention_interval": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "hidden_act": "silu",
    }
    # Linear attention follows the second value of the backend arguments.
    model = model_with_state(
        FakeState(), llm_prefill_att_backend=["triton", "triton"], mtp_step=1, ssm_state_mode=ssm_state_mode
    )
    model.layers_infer = [Qwen3NextTransformerLayerInfer(i, net) for i in range(2)]
    ((linear_prefill, linear_decode),) = layer_backends(registered(model), "draft0", [0])
    expected = GDNLayerConfig(
        2, 4, 64, 64, 4, "silu", torch.bfloat16, state_mtp_step=1, replay=ssm_state_mode == "replay"
    )
    assert linear_prefill.layer_config == linear_decode.layer_config == expected
    assert {backend.name for backend in backends(model, "decode")} == {
        "triton_gqa_decode",
        decode,
    }
