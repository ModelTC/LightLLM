"""AttentionGroups chooses impls per layer config, groups layers by step builder and builds step args once per group."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention import selector
from lightllm.common.attention.backend import AttentionBackend, AttentionImpl, AttentionStepBuilder
from lightllm.common.attention.backends.gqa import GQAKVScales, GQALayerConfig
from lightllm.common.attention.backends.gqa.fa3 import FA3PageTables
from lightllm.common.basemodel.attention_groups import AttentionGroups, gqa_kv_quant_fields, step_builder_context
from lightllm.common.basemodel.infer_struct import InferStateInfo


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
    )


def gqa(**fields) -> GQALayerConfig:
    return GQALayerConfig(**{"num_heads": 8, "num_kv_heads": 2, "head_dim": 64, "dtype": torch.bfloat16, **fields})


def test_triton_layers_share_one_group_across_layer_configs():
    # Gemma4-like: layers differ in head_dim and window, but Triton's step builders ignore the layer config.
    groups = AttentionGroups(fake_model())
    sliding = groups.create_attention(gqa(sliding_window=(511, 0)), gqa(sliding_window=(511, 0)))
    full = groups.create_attention(gqa(head_dim=128), gqa(head_dim=128))
    again = groups.create_attention(gqa(head_dim=128), gqa(head_dim=128))
    assert sliding.group == full.group == again.group == "triton_gqa_prefill+triton_gqa_decode"
    assert full.prefill_impl is again.prefill_impl and full.decode_impl is again.decode_impl
    assert sliding.prefill_impl is not full.prefill_impl
    assert list(groups.groups) == [full.group]


def test_flashinfer_layers_with_different_head_dims_form_separate_groups():
    groups = AttentionGroups(fake_model(llm_prefill_att_backend=["flashinfer"], llm_decode_att_backend=["flashinfer"]))
    small = groups.create_attention(gqa(), gqa())
    large = groups.create_attention(gqa(head_dim=128), gqa(head_dim=128))
    assert small.group != large.group
    assert set(groups.groups) == {small.group, large.group}
    assert large.group.endswith("#1")


class Built(SimpleNamespace):
    pass


class FakeConfig(SimpleNamespace):
    def __hash__(self):
        return hash(tuple(sorted(vars(self).items())))


class FakeBuilder(AttentionStepBuilder):
    def __init__(self, backend, layer_config):
        self.backend, self.layer_config, self.batches = backend, layer_config, []

    def build(self, batch):
        self.batches.append(batch)
        return Built(backend=self.backend, batch=batch)


class FakeImpl(AttentionImpl):
    def forward(self, step_args, layer_inputs, *, out, alloc_func=torch.empty):
        return out


created = []


def fake_backend(name, *, sync=False):
    class Backend(AttentionBackend):
        @staticmethod
        def get_name():
            return name

        @staticmethod
        def get_step_builder_cls():
            return FakeBuilder

        @staticmethod
        def get_impl_cls():
            return FakeImpl

        @classmethod
        def create_step_builder(cls, layer_config, ctx):
            created.append((name, layer_config.key))
            return FakeBuilder(name, layer_config)

        @classmethod
        def step_builder_key(cls, layer_config):
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
    family = {"prefill": {"fake": (f"{__name__}:FakePrefill",)}, "decode": {"fake": (f"{__name__}:FakeDecode",)}}
    monkeypatch.setitem(selector._REGISTRY, FakeConfig, family)
    return AttentionGroups(fake_model(llm_prefill_att_backend=["fake"], llm_decode_att_backend=["fake"]))


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
    a = groups.create_attention(FakeConfig(key=1), FakeConfig(key=1))
    b = groups.create_attention(FakeConfig(key=1, other=2), FakeConfig(key=1, other=2))
    c = groups.create_attention(FakeConfig(key=2), FakeConfig(key=2))
    assert a.group == b.group != c.group
    assert created == []

    prefill = step(prefill=True)
    prefill.att_groups = groups
    prefill.init_att_state()
    # A prefill-only node never creates decode builders.
    assert sorted(created) == [("fake_prefill", 1), ("fake_prefill", 2)]
    assert set(prefill.att_step_args) == {a.group, c.group}
    # Every group's builder saw the same batch, built once for the step.
    batches = {id(args.batch) for args in prefill.att_step_args.values()}
    assert len(batches) == 1

    decode = step(prefill=False)
    decode.att_groups = groups
    decode.init_att_state()
    decode.init_att_state()
    assert sorted(created) == [("fake_decode", 1), ("fake_decode", 2), ("fake_prefill", 1), ("fake_prefill", 2)]
    assert {args.backend for args in decode.att_step_args.values()} == {"fake_decode"}
    assert groups.requires_cpu_cache_sync


def test_models_without_layers_build_no_step_args(fake_family):
    state = step(prefill=True)
    state.att_groups = fake_family
    state.init_att_state()
    assert state.att_step_args == {}


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
    ctx = step_builder_context(model)
    # 8 running requests x 3 rows, padded to a multiple of the TP size.
    assert ctx.max_decode_rows == 24
    assert ctx.max_cuda_graph_batch_size == 4
    assert ctx.microbatch_count == 2
    assert ctx.workspace_bytes == 64 * 1024 * 1024
    assert ctx.diverse_decode
    model.graph_max_batch_size = 64
    assert step_builder_context(model).max_decode_rows == 64


def test_fa3_page_tables_allocate_lazily_per_microbatch_and_check_capacity(monkeypatch):
    requests = []

    def get_workspace(name, numel, dtype):
        requests.append((name, numel, dtype))
        return torch.zeros(numel, dtype=dtype)

    model = fake_model(enable_decode_microbatch_overlap=True)
    ctx = step_builder_context(model)
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


def test_models_on_attention_groups_skip_the_full_attention_adapters():
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

    class Layer:
        def __init__(self):
            self.models = []

        def init_attention(self, model):
            self.models.append(model)

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.__dict__.update(vars(fake_model()))
    model.is_mtp_draft_model = False
    model.layers_infer = [Layer(), Layer()]
    states = [object()]
    model.att_state_manager = SimpleNamespace(get_full_attention_states=lambda role: tuple(states))

    model._init_attention()
    assert model.att_groups is None and model.layers_infer[0].models == []

    model.uses_attention_groups = True
    model._init_attention()
    model._init_att_backend()
    assert isinstance(model.att_groups, AttentionGroups)
    assert all(layer.models == [model] for layer in model.layers_infer)
    assert model.prefill_att_backend is None and model.decode_att_backend is None

    # A draft without full-attention states (e.g. linear-only) creates no groups.
    states.clear()
    model._init_attention()
    assert model.att_groups is None
