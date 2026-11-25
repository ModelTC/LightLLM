"""Model initialization selects layer implementations and builds each group's attention state once per batch."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention import selector
from lightllm.common.attention.backend import AttentionBackend
from lightllm.common.attention.backends.gqa import GQAKVScales, GQALayerConfig
from lightllm.common.attention.backends.gqa.fa3 import FA3PageTables
from lightllm.common.basemodel import attention_runtime
from lightllm.common.basemodel.attention_runtime import (
    AttentionRuntime,
    AttentionSpec,
    AttentionStepConfig,
    attention_build_resources,
    configure_attention_step,
)
from lightllm.common.attention.backends.gqa import gqa_kv_quant_fields
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from unit_tests.common.attention.backends.groups_runner import build_context


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


class FakeModel(SimpleNamespace):
    requires_cpu_cache_sync = TpPartBaseModel.requires_cpu_cache_sync


def fake_model(**args):
    return FakeModel(
        args=start_args(**args),
        max_seq_length=1024,
        graph_max_batch_size=4,
        graph_max_len_in_batch=1024,
        tp_world_size_=1,
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0),
        att_state_role="main",
        att_state_manager=SimpleNamespace(get_bounded_attention_states=lambda role: ()),
    )


def setup_model(model):
    model.attention_runtime = AttentionRuntime(model)
    return model


def backends(model, phase):
    return tuple(model.attention_runtime.backends(phase))


def gqa(**fields) -> GQALayerConfig:
    return GQALayerConfig(**{"num_heads": 8, "num_kv_heads": 2, "head_dim": 64, "dtype": torch.bfloat16, **fields})


def test_triton_layers_share_one_group_across_layer_configs(monkeypatch):
    # Gemma4-like: layers differ in head_dim and window, but Triton's step builders ignore the layer config.
    groups = setup_model(fake_model())
    sliding = groups.attention_runtime.bind(AttentionSpec(gqa(sliding_window=(511, 0)), gqa(sliding_window=(511, 0))))
    full_spec = AttentionSpec(gqa(head_dim=128), gqa(head_dim=128))
    full = groups.attention_runtime.bind(full_spec)
    again = groups.attention_runtime.bind(full_spec)
    assert full == again
    prepared_by = []
    prepare = AttentionBackend.prepare

    def record_prepare(self, infer_state, resources):
        prepared_by.append(self)
        return prepare(self, infer_state, resources)

    monkeypatch.setattr(AttentionBackend, "prepare", record_prepare)
    for is_prefill, representative, other in zip((True, False), sliding, full):
        assert representative is not other
        state = step(is_prefill)
        state.att_states = groups.attention_runtime.prepare(state)
        assert list(state.att_states) == [representative.state_key]
        assert state.att_states[representative.state_key] is state.att_states[other.state_key]
    assert prepared_by == list(sliding)


@pytest.mark.parametrize("prefill", [True, False])
def test_layer_attention_state_belongs_to_its_microbatch(prefill):
    model = setup_model(fake_model())
    impls = model.attention_runtime.bind(AttentionSpec(gqa(), gqa()))
    impl = impls[0 if prefill else 1]
    first = step(prefill)
    first.att_states = model.attention_runtime.prepare(first)
    second = step(prefill)
    second.microbatch_index = 1
    second.att_states = model.attention_runtime.prepare(second)
    state = first.att_states[impl.state_key]
    assert state is not second.att_states[impl.state_key]
    assert state.b_req_idx is first.b_req_idx
    assert second.att_states[impl.state_key].b_req_idx is second.b_req_idx


def test_flashinfer_layers_with_different_head_dims_form_separate_groups():
    groups = setup_model(fake_model(llm_prefill_att_backend=["flashinfer"], llm_decode_att_backend=["flashinfer"]))
    groups.attention_runtime.bind(AttentionSpec(gqa(), gqa()))
    groups.attention_runtime.bind(AttentionSpec(gqa(head_dim=128), gqa(head_dim=128)))
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

    model = setup_model(fake_model(llm_prefill_att_backend=[backend], llm_decode_att_backend=[backend]))
    for scale in (0.1, 0.1, 0.2):
        model.attention_runtime.bind(AttentionSpec(*mla(scale)))
    assert all(len(backends(model, phase)) == groups for phase in selector.PHASES)


def test_sparse_mla_step_builders_follow_the_layer_selection():
    from lightllm.common.attention.backends.sparse_mla import SparseMLALayerConfig
    from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLADecodeBackend
    from lightllm.common.attention.backends.sparse_mla.flashmla import FlashMLASparseDecodeBackend

    ctx = attention_build_resources(fake_model())

    def config(**fields):
        fields = {"num_heads": 64, "kv_lora_rank": 512, "qk_rope_head_dim": 0, "softmax_scale": 0.1, **fields}
        return SparseMLALayerConfig(dtype=torch.bfloat16, **fields)

    builder = build_context(FA3SparseMLADecodeBackend, config(topk=2048, kpool=4, causal=False), ctx)
    assert (builder.topk, builder.kpool, builder.causal) == (2048, 4, False)
    with pytest.raises(ValueError, match="topk"):
        build_context(FA3SparseMLADecodeBackend, config(), ctx)
    # 2048 pooled entries plus at most three kpool remainder entries bound the direct decode window.
    fp8 = dict(kv_cache_format="fp8_group64_e8m0", topk=2048)
    assert build_context(FlashMLASparseDecodeBackend, config(kpool=4, **fp8), ctx).direct_decode_bound == 2051
    assert build_context(FlashMLASparseDecodeBackend, config(kpool=1, **fp8), ctx).direct_decode_bound is None


def test_layer_family_selects_its_command_line_argument():
    groups = setup_model(fake_model(llm_prefill_att_backend=["fa3", "flashqla"]))
    assert attention_runtime.requested_backend(groups.args, "prefill", gqa()) == "fa3"
    linear = SimpleNamespace()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(selector, "arg_index", lambda layer_config: 1 if layer_config is linear else 0)
        assert attention_runtime.requested_backend(groups.args, "prefill", linear) == "flashqla"
        assert attention_runtime.requested_backend(groups.args, "decode", linear) == "auto"


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("prefill_kernel, flash_len", [("triton", None), ("flashkda", None), ("auto", 2048)])
def test_kda_prefill_step_builders_follow_the_layer_config(monkeypatch, replay, prefill_kernel, flash_len):
    from lightllm.common.attention.backends.kda import KDALayerConfig
    from lightllm.common.attention.backends.kda import triton as kda_triton

    requests = []
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
    builder = build_context(backend, config, attention_build_resources(fake_model()))
    assert builder.ssm_slots_per_request == (1 if replay else 3)
    assert requests == [(prefill_kernel, {"dtype": torch.bfloat16, "ssm_dtype": torch.bfloat16})]
    assert builder.flash_kda_fwd == "fwd" and builder.max_flash_len == flash_len


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


def fake_backend(name, *, sync=False):
    class Backend(FakeImpl):
        @staticmethod
        def get_name():
            return name

        build_context_type = FakeBuilder

        @classmethod
        def build(cls, metadata, context):
            return context.build(metadata)

        @classmethod
        def create_build_context(cls, build_key, ctx):
            created.append((name, build_key))
            return FakeBuilder(name)

        @classmethod
        def build_context_key(cls, layer_config):
            return layer_config.key

        @classmethod
        def requires_cpu_cache_sync(cls):
            return sync

    return Backend


FakePrefill = fake_backend("fake_prefill")
FakeDecode = fake_backend("fake_decode", sync=True)


@pytest.fixture
def fake_family(monkeypatch):
    created.clear()
    backends = {"prefill": {"fake": (f"{__name__}:FakePrefill",)}, "decode": {"fake": (f"{__name__}:FakeDecode",)}}
    monkeypatch.setitem(selector._REGISTRY, FakeConfig, selector.Family(backends))
    return setup_model(fake_model(llm_prefill_att_backend=["fake"], llm_decode_att_backend=["fake"]))


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
    first_prefill, first_decode = groups.attention_runtime.bind(AttentionSpec(FakeConfig(key=1), FakeConfig(key=1)))
    second_prefill, second_decode = groups.attention_runtime.bind(
        AttentionSpec(FakeConfig(key=2), FakeConfig(key=1, other=2))
    )
    assert created == []

    prefill = step(prefill=True)
    prefill.att_states = groups.attention_runtime.prepare(prefill)
    # A prefill-only node never creates decode builders.
    assert sorted(created) == [("fake_prefill", 1), ("fake_prefill", 2)]
    assert set(prefill.att_states) == {first_prefill.state_key, second_prefill.state_key}
    assert prefill.att_states[first_prefill.state_key] is not prefill.att_states[second_prefill.state_key]
    # Every group's builder saw the same batch, built once for the step.
    batches = {id(args.batch) for args in prefill.att_states.values()}
    assert len(batches) == 1

    decode = step(prefill=False)
    decode.att_states = groups.attention_runtime.prepare(decode)
    first_args = decode.att_states[first_decode.state_key]
    assert first_decode is not second_decode
    assert first_args is decode.att_states[second_decode.state_key]
    overlap = step(prefill=False, batch_size=3)
    overlap.microbatch_index = 1
    overlap.att_states = groups.attention_runtime.prepare(overlap)
    assert overlap.att_states[first_decode.state_key] is overlap.att_states[second_decode.state_key]
    assert overlap.att_states[first_decode.state_key].batch.microbatch_index == 1
    assert decode.att_states[first_decode.state_key] is first_args
    assert first_args.batch.batch_size == 2
    assert sorted(created) == [("fake_decode", 1), ("fake_prefill", 1), ("fake_prefill", 2)]
    assert groups.requires_cpu_cache_sync


def test_decode_only_layers_build_no_prefill_att_state(fake_family):
    groups = fake_family
    prefill_impl, decode_impl = groups.attention_runtime.bind(AttentionSpec(None, FakeConfig(key=1)))
    assert prefill_impl is None
    assert backends(groups, "prefill") == () and backends(groups, "decode") == (FakeDecode,)

    prefill = step(prefill=True)
    prefill.att_states = groups.attention_runtime.prepare(prefill)
    assert prefill.att_states == {} and created == []

    decode = step(prefill=False)
    decode.att_states = groups.attention_runtime.prepare(decode)
    assert set(decode.att_states) == {decode_impl.state_key} and created == [("fake_decode", 1)]


def test_layer_override_is_independent_for_identical_configs():
    model = fake_model(llm_prefill_att_backend=["fa3"], llm_decode_att_backend=["fa3"])
    groups = setup_model(model)
    config = gqa()
    pinned = groups.attention_runtime.bind(
        AttentionSpec(config, config, prefill_implementation="triton", decode_implementation="triton")
    )
    default = groups.attention_runtime.bind(AttentionSpec(config, config))
    assert groups.attention_runtime.bind(AttentionSpec(config, config, prefill_implementation="triton")) == (
        pinned[0],
        default[1],
    )
    assert groups.attention_runtime.bind(AttentionSpec(config, config, decode_implementation="triton")) == (
        default[0],
        pinned[1],
    )
    for phase in selector.PHASES:
        assert [backend.get_name() for backend in backends(groups, phase)] == [
            f"triton_gqa_{phase}",
            f"fa3_gqa_{phase}",
        ]


def test_models_without_layers_build_no_att_state(fake_family):
    state = step(prefill=True)
    state.att_states = fake_family.attention_runtime.prepare(state)
    assert state.att_states == {}


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


def test_step_builder_context_covers_speculative_and_padded_decode_rows():
    model = fake_model(mtp_step=2, enable_tpsp_mix_mode=True, enable_decode_microbatch_overlap=True, diverse_mode=True)
    model.tp_world_size_ = 4
    ctx = attention_build_resources(model)
    # 8 running requests x 3 rows, padded to a multiple of the TP size.
    assert ctx.max_decode_rows == 24
    assert ctx.max_cuda_graph_batch_size == 4
    assert ctx.microbatch_count == 2
    assert ctx.workspace_bytes == 64 * 1024 * 1024
    assert ctx.diverse_decode
    model.graph_max_batch_size = 64
    assert attention_build_resources(model).max_decode_rows == 64


def test_fa3_page_tables_allocate_lazily_per_microbatch_and_check_capacity(monkeypatch):
    requests = []

    def get_workspace(name, numel, dtype):
        requests.append((name, numel, dtype))
        return torch.zeros(numel, dtype=dtype)

    model = fake_model(enable_decode_microbatch_overlap=True)
    ctx = attention_build_resources(model)
    ctx = type(ctx)(**{**vars(ctx), "get_workspace": get_workspace})
    tables = FA3PageTables(ctx, page_size=16)
    assert requests == []
    view = tables.view(rows=3, max_kv_len=100, microbatch_index=1)
    assert view.shape == (3, 7) and view.is_contiguous()
    rows = max(ctx.max_decode_rows, ctx.max_cuda_graph_batch_size)
    assert requests == [(f"fa3_page_table_{i}", rows * 64, torch.int32) for i in range(2)]
    with pytest.raises(RuntimeError, match="batch size"):
        tables.view(rows=rows + 1, max_kv_len=16, microbatch_index=0)
    with pytest.raises(RuntimeError, match="max KV sequence length"):
        tables.view(rows=1, max_kv_len=ctx.max_seq_len + 1, microbatch_index=0)


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
        get_state=lambda **query: state, get_bounded_attention_states=lambda role: ()
    )
    return setup_model(model)


def test_llama_layer_creates_its_attention_from_its_cache(runtime_start_args):
    scales = GQAKVScales(torch.ones(16, 2, 8), torch.ones(16, 2, 8))
    model = model_with_state(FakeState(scales=scales), llm_kv_type="int8kv")
    layer = llama_layer()
    layer.prefill_impl, layer.decode_impl = model.attention_runtime.bind(layer.attention_spec(model))
    expected = gqa(kv_quant="int8_group", kv_quant_group_size=8)
    assert layer.prefill_impl.layer_config == layer.decode_impl.layer_config == expected
    assert backends(model, "decode")[0].get_name() == "triton_quant_gqa_decode"


def test_only_prefill_attends_inside_image_spans(runtime_start_args):
    from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import NeoChatTransformerLayerInfer

    layer = NeoChatTransformerLayerInfer(
        3,
        {"rms_norm_eps": 1e-6, "num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512, "head_dim": 64},
    )
    model = model_with_state(FakeState())
    assert layer.attention_spec(model) == AttentionSpec(gqa(bidirectional_image_spans=True), gqa())
    assert layer.prefill_impl is layer.decode_impl is None
    assert not model.attention_runtime.has_attention


def test_block_draft_layers_are_not_causal(monkeypatch):
    monkeypatch.setattr(attention_runtime, "get_env_start_args", lambda: SimpleNamespace(mtp_mode="dflash"))
    model = model_with_state(FakeState(), llm_prefill_att_backend=["fa3"], llm_decode_att_backend=["fa3"])
    model.is_mtp_draft_model = True
    layer = llama_layer()
    assert layer.attention_spec(model).decode == gqa(causal=False)


def test_unsupported_layer_configs_fail_when_the_model_loads(runtime_start_args):
    # FlashInfer cannot run the static FP8 per-head cache that only FA3 supports.
    scales = GQAKVScales(torch.ones(2), torch.ones(2))
    model = model_with_state(
        FakeState(scales=scales),
        llm_kv_type="fp8kv_sph",
        llm_prefill_att_backend=["flashinfer"],
        llm_decode_att_backend=["flashinfer"],
    )
    with pytest.raises(ValueError, match="does not support"):
        model.attention_runtime.bind(llama_layer().attention_spec(model))


def test_models_with_attention_states_create_their_layers_attention():
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

    class Layer:
        attention_phases = ("decode",)

        def __init__(self):
            self.models = []

        def attention_spec(self, model):
            self.models.append(model)
            return AttentionSpec(None, gqa())

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.__dict__.update(vars(fake_model()))
    model.is_mtp_draft_model = False
    model.layers_infer = [Layer(), Layer()]
    states = [object()]
    model.att_state_manager = SimpleNamespace(
        get_full_attention_states=lambda role: tuple(states), get_bounded_attention_states=lambda role: ()
    )

    model._init_attention()
    assert model.attention_runtime.has_attention
    assert all(layer.models == [model] for layer in model.layers_infer)
    first, second = model.layers_infer
    assert first.prefill_impl is second.prefill_impl is None
    assert first.decode_impl is second.decode_impl is not None

    # A draft without attention states (FFN-only) binds no layers and builds no attention state.
    states.clear()
    model._init_attention()
    assert not model.attention_runtime.has_attention


@pytest.mark.parametrize(
    "required,spec,error",
    [
        (("prefill", "decode"), AttentionSpec(None, gqa()), "missing prefill"),
        (("prefill", "decode"), AttentionSpec(gqa(), None), "missing decode"),
        (("decode",), AttentionSpec(gqa(), gqa()), "unexpected prefill"),
    ],
)
def test_model_rejects_phase_config_that_disagrees_with_layer_execution(required, spec, error):
    model = model_with_state(FakeState())
    model.att_state_manager.get_full_attention_states = lambda role: (object(),)
    model.layers_infer = [SimpleNamespace(layer_num_=7, attention_phases=required, attention_spec=lambda model: spec)]
    with pytest.raises(ValueError, match=f"draft0 layer 7: {error} attention configuration"):
        TpPartBaseModel._init_attention(model)
    assert not model.attention_runtime.has_attention


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
    linear, full = (Qwen3NextTransformerLayerInfer(i, net) for i in range(2))
    linear.prefill_impl, linear.decode_impl = model.attention_runtime.bind(linear.attention_spec(model))
    full.prefill_impl, full.decode_impl = model.attention_runtime.bind(full.attention_spec(model))
    expected = GDNLayerConfig(
        2, 4, 64, 64, 4, "silu", torch.bfloat16, state_mtp_step=1, replay=ssm_state_mode == "replay"
    )
    assert linear.prefill_impl.layer_config == linear.decode_impl.layer_config == expected
    assert {backend.get_name() for backend in backends(model, "decode")} == {
        "triton_gqa_decode",
        decode,
    }


def test_speculative_decode_batches_carry_the_recurrent_state_index(monkeypatch):
    monkeypatch.setattr(attention_runtime, "get_env_start_args", lambda: SimpleNamespace(mtp_dynamic_verify=False))
    index = torch.arange(8, dtype=torch.int32)
    recurrent = SimpleNamespace(mtp_state_index=index)
    model = fake_model(mtp_step=1)
    model.att_state_role = "main"
    model.att_state_manager = SimpleNamespace(get_bounded_attention_states=lambda role: (object(), recurrent))
    assert attention_runtime.recurrent_mtp_state_index(model) is index
    for rows, expected in ((1, None), (2, index)):
        model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda _, rows=rows: rows - 1)
        config = attention_runtime.attention_step_config(model)
        assert config.decode_rows_per_request == rows and config.req_to_mtp_state_index is index
        assert configure_attention_step(step(prefill=False), config).req_to_mtp_state_index is expected
        # Prefill rows are never speculative.
        assert configure_attention_step(step(prefill=True), config).req_to_mtp_state_index is None
    model.att_state_manager = SimpleNamespace(get_bounded_attention_states=lambda role: (recurrent, recurrent))
    with pytest.raises(ValueError, match="at most one recurrent state"):
        attention_runtime.recurrent_mtp_state_index(model)


@pytest.mark.parametrize("dynamic", [False, True])
def test_step_config_is_read_from_the_model_once(monkeypatch, dynamic):
    monkeypatch.setattr(attention_runtime, "get_env_start_args", lambda: SimpleNamespace(mtp_dynamic_verify=dynamic))
    model = fake_model(mtp_step=2)
    model.att_state_role = "main"
    model.att_state_manager = SimpleNamespace(get_bounded_attention_states=lambda role: ())
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda is_draft: 2)
    runtime = AttentionRuntime(model)
    assert runtime.step_config == AttentionStepConfig(
        decode_rows_per_request=3, dynamic_decode_rows=dynamic, cuda_graph_max_kv_seq_len=1024
    )
    # Steps read nothing more from the model.
    model.mtp_manager = model.graph_max_len_in_batch = None
    metadata = configure_attention_step(step(prefill=False), runtime.step_config)
    assert (metadata.decode_rows_per_request, metadata.dynamic_decode_rows) == (3, dynamic)


@pytest.mark.parametrize("graph_mode", list(GraphMode))
def test_only_cuda_graph_steps_carry_the_graph_kv_capacity(graph_mode):
    state = step(prefill=False)
    state.graph_mode = graph_mode
    metadata = configure_attention_step(state, AttentionStepConfig(cuda_graph_max_kv_seq_len=512))
    assert metadata is state
    assert metadata.cuda_graph_max_kv_seq_len == (None if graph_mode is GraphMode.EAGER else 512)
    assert metadata.uses_cuda_graph == (graph_mode is not GraphMode.EAGER)
    if graph_mode is not GraphMode.EAGER:
        with pytest.raises(ValueError, match="kv length capacity"):
            configure_attention_step(state, AttentionStepConfig())
