"""Model initialization selects layer implementations and builds each group's attention state once per batch."""
from typing import Any
from unit_tests.common.attention.backends.step_fixture import create_att_state

from functools import partial
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.selection import PHASES
from lightllm.common.attention.base_backend import AttRuntimeConfig, PrefillAttBackend, DecodeAttBackend
from lightllm.common.basemodel.layer_infer.transformer_layer_infer import TransformerLayerInfer
from lightllm.common.attention.backends.gqa import GQAKVScales, GQAAttConfig
from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeBackend
from lightllm.common.attention.backends.gqa.flashinfer import FlashInferGQADecodeBackend, FlashInferGQAPrefillBackend
from lightllm.common.attention.backends.gqa.triton import TritonGQADecodeBackend
from lightllm.common.attention.backends.gqa.triton_quant import TritonQuantGQADecodeBackend
from lightllm.common.attention.attention_backend_manager import AttentionBackendManager
from lightllm.common.attention.backends.gqa import get_gqa_kv_quant
from lightllm.common.attention_state import AttentionStateManager
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo
from lightllm.common.basemodel.basemodel import TpPartBaseModel


class FakePrefill(PrefillAttBackend):
    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        self.prepared_microbatches: list[int] = []

    @staticmethod
    def get_name() -> str:
        return "fake_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> SimpleNamespace:
        self.prepared_microbatches.append(infer_state.microbatch_index)
        return SimpleNamespace(b_req_idx=infer_state.b_req_idx)

    def prefill(
        self,
        att_state: object,
        layer_inputs: object,
        *,
        out: torch.Tensor,
        alloc_func: Callable = torch.empty,
    ) -> torch.Tensor:
        return out


class FakeDecode(DecodeAttBackend):
    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        self.prepared_microbatches: list[int] = []

    @staticmethod
    def get_name() -> str:
        return "fake_decode"

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return True

    def create_att_state(self, infer_state: InferStateInfo) -> SimpleNamespace:
        self.prepared_microbatches.append(infer_state.microbatch_index)
        return SimpleNamespace(b_req_idx=infer_state.b_req_idx)

    def decode(
        self,
        att_state: object,
        layer_inputs: object,
        *,
        out: torch.Tensor,
        alloc_func: Callable = torch.empty,
    ) -> torch.Tensor:
        return out


def start_args(**overrides):
    args = dict(
        att_backend={"gqa.prefill": "triton", "gqa.decode": "triton", **overrides.pop("att_backend", {})},
        mtp_step=0,
        mtp_mode=None,
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
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda is_draft_model: 0),
        att_state_role="main",
        att_state_manager=SimpleNamespace(get_bounded_attention_states=lambda role: ()),
    )


def setup_model(model):
    model.attention_backend_manager = AttentionBackendManager()
    return model


def backends(model, phase):
    return model.attention_backend_manager.get_backends(role=model.att_state_role, phase=phase)


def layer_backend(model: Any, layer: Any, phase: Any):
    return model.attention_backend_manager.get_backend(
        role=model.att_state_role, phase=phase, logical_layer_id=layer.layer_num_
    )


def gqa(**fields) -> GQAAttConfig:
    return GQAAttConfig(
        num_heads=fields.get("num_heads", 8),
        num_kv_heads=fields.get("num_kv_heads", 2),
        head_dim=fields.get("head_dim", 64),
        dtype=fields.get("dtype", torch.bfloat16),
        sliding_window=fields.get("sliding_window", (-1, -1)),
        softmax_scale=fields.get("softmax_scale", None),
        causal=fields.get("causal", True),
        has_sinks=fields.get("has_sinks", False),
        kv_quant=fields.get("kv_quant", None),
        kv_quant_group_size=fields.get("kv_quant_group_size", None),
        has_q_scale=fields.get("has_q_scale", False),
        has_alibi=fields.get("has_alibi", False),
        bidirectional_image_spans=fields.get("bidirectional_image_spans", False),
    )


def test_registered_instances_define_layer_mapping_and_prepare_once() -> None:
    manager = AttentionBackendManager()
    first = FakePrefill(att_config=gqa(), runtime_config=AttRuntimeConfig(logical_layer_ids=(3, 12)))
    other = FakePrefill(att_config=gqa(head_dim=128), runtime_config=AttRuntimeConfig(logical_layer_ids=(7,)))
    decode = FakeDecode(att_config=gqa(), runtime_config=AttRuntimeConfig(logical_layer_ids=(3, 7, 12)))
    manager.register_backends(role="main", backends=(first, other, decode))
    assert manager.get_backend(role="main", phase="prefill", logical_layer_id=3) is first
    assert manager.get_backend(role="main", phase="prefill", logical_layer_id=12) is first
    assert manager.get_backend(role="main", phase="prefill", logical_layer_id=7) is other
    assert manager.get_backends(role="main", phase="decode") == (decode,)
    assert manager.main_fake_prefill_3_backend is first
    assert manager.main_fake_prefill_7_backend is other
    assert first.prepared_microbatches == other.prepared_microbatches == decode.prepared_microbatches == []

    states = manager.create_att_state(infer_state=step(prefill=True))
    assert set(states) == {first, other}
    assert states[first] is not states[other]
    assert first.prepared_microbatches == other.prepared_microbatches == [0]
    assert decode.prepared_microbatches == []
    assert manager.requires_cpu_cache_sync(role="main")

    states = manager.create_att_state(infer_state=step(prefill=False))
    assert set(states) == {decode}
    assert decode.prepared_microbatches == [0]


@pytest.mark.parametrize("prefill", [True, False])
def test_layer_attention_state_belongs_to_its_microbatch(prefill: bool) -> None:
    manager = AttentionBackendManager()
    runtime = AttRuntimeConfig(logical_layer_ids=(3,))
    backend = (
        FakePrefill(att_config=gqa(), runtime_config=runtime)
        if prefill
        else FakeDecode(att_config=gqa(), runtime_config=runtime)
    )
    manager.register_backends(role="main", backends=(backend,))
    first = step(prefill=prefill)
    second = step(prefill=prefill)
    second.microbatch_index = 1
    first.att_states = manager.create_att_state(infer_state=first)
    second.att_states = manager.create_att_state(infer_state=second)
    assert first.att_states[backend] is not second.att_states[backend]
    assert first.att_states[backend].b_req_idx is first.b_req_idx
    assert second.att_states[backend].b_req_idx is second.b_req_idx
    assert backend.prepared_microbatches == [0, 1]


def test_registration_preserves_distinct_instances_with_identical_configs() -> None:
    manager = AttentionBackendManager()
    first = FakeDecode(att_config=gqa(), runtime_config=AttRuntimeConfig(logical_layer_ids=(2,)))
    second = FakeDecode(att_config=gqa(), runtime_config=AttRuntimeConfig(logical_layer_ids=(7,)))
    manager.register_backends(role="main", backends=(first, second))
    assert manager.get_backends(role="main", phase="decode") == (first, second)
    assert first.logical_layer_ids == (2,)
    assert second.logical_layer_ids == (7,)
    states = manager.create_att_state(infer_state=step(prefill=False))
    assert set(states) == {first, second}


def test_decode_only_registration_prepares_no_prefill_state() -> None:
    manager = AttentionBackendManager()
    backend = FakeDecode(att_config=gqa(), runtime_config=AttRuntimeConfig(logical_layer_ids=(9,)))
    manager.register_backends(role="main", backends=(backend,))
    assert manager.get_backend(role="main", phase="prefill", logical_layer_id=9) is None
    assert manager.create_att_state(infer_state=step(prefill=True)) == {}
    assert backend.prepared_microbatches == []
    assert set(manager.create_att_state(infer_state=step(prefill=False))) == {backend}
    assert backend.prepared_microbatches == [0]


def test_registration_finishes_before_inference_and_cannot_repeat() -> None:
    manager = AttentionBackendManager()
    with pytest.raises(expected_exception=RuntimeError, match="registration must finish"):
        manager.create_att_state(infer_state=step(prefill=False))
    manager.register_backends(role="main", backends=())
    assert manager.create_att_state(infer_state=step(prefill=False)) == {}
    assert not manager.has_attention(role="main")
    assert not manager.requires_cpu_cache_sync(role="main")
    with pytest.raises(expected_exception=RuntimeError, match="registration is already finished"):
        manager.register_backends(role="main", backends=())


def test_real_model_factory_shares_backends_and_registers_ffn_only_draft() -> None:
    model = LlamaTpPartModel.__new__(LlamaTpPartModel)
    model.__dict__.update(vars(model_with_state(state=FakeState())))
    model.is_mtp_draft_model = False
    model.layers_infer = [SimpleNamespace(layer_num_=3), SimpleNamespace(layer_num_=9)]
    model.att_state_manager.get_full_attention_states = lambda role: (object(),) if role == "main" else ()
    model._init_attention()
    manager = model.attention_backend_manager
    for phase in ("prefill", "decode"):
        backend = manager.get_backend(role="main", phase=phase, logical_layer_id=3)
        assert backend is manager.get_backend(role="main", phase=phase, logical_layer_id=9)
        assert backend.logical_layer_ids == (3, 9)

    draft = TpPartBaseModel.__new__(TpPartBaseModel)
    draft.is_mtp_draft_model = True
    draft.main_model = model
    draft.mtp_previous_draft_models = []
    draft.att_state_manager = model.att_state_manager
    draft._init_attention()
    assert draft.attention_backend_manager is manager
    assert manager.has_attention(role="main")
    assert not manager.has_attention(role="draft0")
    state = step(prefill=False)
    state.att_state_role = "draft0"
    assert manager.create_att_state(infer_state=state) == {}


def test_model_roles_register_independent_instances_and_prepare_only_their_role() -> None:
    class Model(TpPartBaseModel):
        @classmethod
        def create_att_backends(cls, model: TpPartBaseModel) -> tuple[PrefillAttBackend, DecodeAttBackend]:
            runtime = AttRuntimeConfig(logical_layer_ids=(3,))
            return (
                FakePrefill(att_config=gqa(), role=model.att_state_role, runtime_config=runtime),
                FakeDecode(att_config=gqa(), role=model.att_state_role, runtime_config=runtime),
            )

    main = Model.__new__(Model)
    main.att_state_manager = SimpleNamespace(get_full_attention_states=lambda role: (object(),))
    main._init_attention()
    manager = main.attention_backend_manager
    drafts = []
    for _ in range(2):
        draft = Model.__new__(Model)
        draft.is_mtp_draft_model = True
        draft.main_model = main
        draft.mtp_previous_draft_models = tuple(drafts)
        draft.att_state_manager = main.att_state_manager
        state = step(prefill=False)
        state.att_state_role = draft.att_state_role
        with pytest.raises(expected_exception=RuntimeError, match=f"registration must finish.*{draft.att_state_role}"):
            manager.create_att_state(infer_state=state)
        draft._init_attention()
        assert draft.attention_backend_manager is manager
        backend = manager.get_backend(role=draft.att_state_role, phase="decode", logical_layer_id=3)
        assert set(manager.create_att_state(infer_state=state)) == {backend}
        drafts.append(draft)

    for phase in ("prefill", "decode"):
        instances = []
        for model in (main, drafts[0], drafts[1]):
            backend = manager.get_backend(role=model.att_state_role, phase=phase, logical_layer_id=3)
            name = backend.get_registration_name()
            assert name == f"{model.att_state_role}_fake_{phase}_3_backend"
            assert getattr(manager, name) is backend
            assert not hasattr(backend, "backend_name")
            instances.append(backend)
        assert len(set(instances)) == 3


def test_sparse_mla_step_builders_follow_the_layer_selection():
    from lightllm.common.attention.backends.sparse_mla import SparseMLAAttConfig
    from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLADecodeBackend
    from lightllm.common.attention.backends.sparse_mla.flashmla import FlashMLASparseDecodeBackend

    def config(**fields):
        fields = {"num_heads": 64, "kv_lora_rank": 512, "qk_rope_head_dim": 0, "softmax_scale": 0.1, **fields}
        return SparseMLAAttConfig(
            num_heads=fields["num_heads"],
            kv_lora_rank=fields["kv_lora_rank"],
            qk_rope_head_dim=fields["qk_rope_head_dim"],
            softmax_scale=fields["softmax_scale"],
            dtype=fields.get("dtype", torch.bfloat16),
            kv_cache_format=fields.get("kv_cache_format", "bf16"),
            topk=fields.get("topk", None),
            kpool=fields.get("kpool", 1),
            causal=fields.get("causal", True),
        )

    batch = step(prefill=False)
    backend = FA3SparseMLADecodeBackend(att_config=config(topk=2048, kpool=4, causal=False))
    att_state = create_att_state(backend=backend, infer_state=batch)
    assert backend.att_config.causal is False
    torch.testing.assert_close(att_state.cache_seqlens, batch.b_seq_len)
    with pytest.raises(ValueError, match="topk"):
        create_att_state(backend=FA3SparseMLADecodeBackend(att_config=config()), infer_state=batch)
    # 2048 个池化条目加上最多三个 kpool 余数条目限制了直接解码窗口。
    fp8 = dict(kv_cache_format="fp8_group64_e8m0", topk=2048)
    assert FlashMLASparseDecodeBackend(att_config=config(kpool=4, **fp8)).direct_decode_bound == 2051
    assert FlashMLASparseDecodeBackend(att_config=config(kpool=1, **fp8)).direct_decode_bound is None


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("prefill_kernel, flash_len", [("triton", None), ("flashkda", None), ("auto", 2048)])
def test_kda_prefill_step_builders_follow_the_layer_config(monkeypatch, replay, prefill_kernel, flash_len):
    from lightllm.common.attention.backends.kda import KDAAttConfig
    from lightllm.common.attention.backends.kda import triton as kda_triton

    requests = []
    monkeypatch.setattr(kda_triton.TritonKDAPrefillBackend, "_check_prefill_kernel", lambda att_config: None)
    monkeypatch.setattr(
        kda_triton.TritonKDAPrefillBackend,
        "load_flash_kda_fwd",
        lambda requested, **kwargs: requests.append((requested, kwargs)) or "fwd",
    )
    config = KDAAttConfig(
        num_heads=4,
        head_dim=128,
        conv_kernel_size=4,
        lower_bound=-5.0,
        dtype=torch.bfloat16,
        state_mtp_step=2,
        replay=replay,
        ssm_dtype=torch.bfloat16,
        prefill_kernel=prefill_kernel,
    )
    backend = kda_triton.ReplayKDAPrefillBackend if replay else kda_triton.TritonKDAPrefillBackend
    batch = step(prefill=True)
    batch.b1_cu_q_seq_len = torch.arange(3, dtype=torch.int32)
    batch.max_q_seq_len = 1
    impl = backend(config)
    att_state = create_att_state(backend=impl, infer_state=batch)
    torch.testing.assert_close(att_state.ssm_idx, batch.b_req_idx * (1 if replay else 3))
    assert requests == [(prefill_kernel, {"dtype": torch.bfloat16, "ssm_dtype": torch.bfloat16})]
    assert att_state.flash_kda_fwd == "fwd"
    batch.max_q_seq_len = 2049
    assert create_att_state(backend=impl, infer_state=batch).flash_kda_fwd == (None if flash_len else "fwd")
    assert len(requests) == 1


def step(prefill: bool, batch_size: int = 2) -> InferStateInfo:
    state = InferStateInfo()
    state.is_prefill = prefill
    state.batch_size, state.total_token_num, state.max_kv_seq_len = batch_size, 4 * batch_size, 4
    state.b_req_idx = torch.arange(batch_size, dtype=torch.int32)
    state.b_seq_len = torch.full((batch_size,), 4, dtype=torch.int32)
    state.req_manager = SimpleNamespace(req_to_token_indexs=torch.zeros(batch_size, 8, dtype=torch.int32))
    return state


@pytest.mark.parametrize(
    "kv_type, scales, expected",
    [
        ("None", None, (None, None, False)),
        (
            "int8kv",
            GQAKVScales(torch.ones(4, 2, 8), torch.ones(4, 2, 8)),
            ("int8_group", 8, False),
        ),
        (
            "int4kv",
            GQAKVScales(torch.ones(4, 2, 2), torch.ones(4, 2, 2)),
            ("int4_group", 32, False),
        ),
        (
            "fp8kv_spt",
            GQAKVScales(0.5, 0.5),
            ("fp8_per_tensor", None, False),
        ),
        (
            "fp8kv_sph",
            GQAKVScales(torch.ones(2), torch.ones(2)),
            ("fp8_per_head", None, False),
        ),
        (
            "fp8kv_sph",
            GQAKVScales(torch.ones(2), torch.ones(2), torch.ones(2)),
            ("fp8_per_head", None, True),
        ),
    ],
)
def test_kv_quant_fields_follow_the_kv_type_and_the_cache_scales(kv_type, scales, expected):
    kv_quant, group_size, has_q_scale = get_gqa_kv_quant(llm_kv_type=kv_type, kv_scales=scales, head_dim=64)
    assert (kv_quant, group_size, has_q_scale) == expected
    gqa(kv_quant=kv_quant, kv_quant_group_size=group_size, has_q_scale=has_q_scale)


def test_kv_quant_fields_reject_mismatched_caches():
    with pytest.raises(ValueError, match="does not match"):
        get_gqa_kv_quant(llm_kv_type="int8kv", kv_scales=None, head_dim=64)
    with pytest.raises(ValueError, match="no GQA KV cache layout"):
        get_gqa_kv_quant(llm_kv_type="fp8kv_dsa", kv_scales=None, head_dim=64)


def test_constructor_parameters_cover_speculative_and_padded_decode_rows():
    model = fake_model(mtp_step=2, enable_tpsp_mix_mode=True, enable_decode_microbatch_overlap=True, diverse_mode=True)
    model.tp_world_size_ = 4
    config = FA3GQADecodeBackend.create_runtime_config(model=model)
    assert config.max_decode_rows == 24
    assert config.microbatch_count == 2
    assert FlashInferGQADecodeBackend.create_runtime_config(model=model).max_batch_size == 4
    assert FlashInferGQAPrefillBackend.create_runtime_config(model=model).workspace_bytes == 64 * 1024 * 1024
    assert TritonQuantGQADecodeBackend.create_runtime_config(model=model).diverse_decode
    assert TritonGQADecodeBackend.create_runtime_config(model=model).decode_rows_per_request == 1
    model.graph_max_batch_size = 64
    model.args.att_backend["gqa.decode"] = "fa3"
    from lightllm.common.attention.backends.gqa import selector as gqa_selector

    decode = gqa_selector.select_gqa_decode_backend(att_config=gqa(), model=model, logical_layer_ids=(0,))
    assert isinstance(decode, FA3GQADecodeBackend)
    assert decode.runtime_config.max_decode_rows == 64


def test_fa3_page_tables_are_private_lazy_and_check_capacity(monkeypatch: pytest.MonkeyPatch):
    requests = []

    def empty(size: tuple[int, ...], *, dtype: torch.dtype, device: str | int):
        requests.append((size[0], dtype, device))
        return torch.zeros(size=size, dtype=dtype)

    monkeypatch.setattr(torch, "empty", empty)
    model = fake_model(enable_decode_microbatch_overlap=True)
    config = FA3GQADecodeBackend.create_runtime_config(model=model)
    backend = FA3GQADecodeBackend(att_config=gqa(), runtime_config=config)
    tables = partial(FA3GQADecodeBackend.fa3_page_table_view, config, config.page_size, backend._page_tables)
    assert requests == []
    view = tables(rows=3, max_kv_len=100, microbatch_index=1)
    assert view.shape == (3, 7) and view.is_contiguous()
    rows = config.max_decode_rows
    assert len(requests) == 2 and all(item[0] == rows * 64 for item in requests)
    second_backend = FA3GQADecodeBackend(att_config=gqa(), runtime_config=config)
    other = partial(FA3GQADecodeBackend.fa3_page_table_view, config, config.page_size, second_backend._page_tables)
    assert other(rows=3, max_kv_len=100, microbatch_index=1).data_ptr() != view.data_ptr()
    with pytest.raises(RuntimeError, match="batch size"):
        tables(rows=rows + 1, max_kv_len=16, microbatch_index=0)
    with pytest.raises(RuntimeError, match="max KV sequence length"):
        tables(rows=1, max_kv_len=config.max_seq_len + 1, microbatch_index=0)


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
    model.create_att_backends = LlamaTpPartModel.create_att_backends
    model.data_type = torch.bfloat16
    model.config = dict(num_attention_heads=8, num_key_value_heads=2, hidden_size=512)
    model.att_state_role = "draft0"
    model.att_state_manager = SimpleNamespace(
        get_state=lambda **query: state, get_bounded_attention_states=lambda role: ()
    )
    return setup_model(model)


def test_llama_layer_creates_its_attention_from_its_cache(runtime_start_args):
    scales = GQAKVScales(torch.ones(16, 2, 8), torch.ones(16, 2, 8))
    model = model_with_state(FakeState(scales=scales), llm_kv_type="int8kv")
    layer = llama_layer()
    model.layers_infer = [layer]
    model.attention_backend_manager.register_backends(
        role=model.att_state_role, backends=model.create_att_backends(model=model)
    )
    expected = gqa(kv_quant="int8_group", kv_quant_group_size=8)
    assert (
        layer_backend(model=model, layer=layer, phase="prefill").att_config
        == layer_backend(model=model, layer=layer, phase="decode").att_config
        == expected
    )
    assert backends(model, "decode")[0].get_name() == "triton_quant_gqa_decode"


def test_only_prefill_attends_inside_image_spans(runtime_start_args):
    from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import NeoChatTransformerLayerInfer

    layer = NeoChatTransformerLayerInfer(
        3,
        {"rms_norm_eps": 1e-6, "num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512, "head_dim": 64},
    )
    model = model_with_state(FakeState())
    from lightllm.models.neo_chat.model import NeoTpPartModel

    model.layers_infer = [layer]
    prefill, decode = NeoTpPartModel.create_att_backends(model=model)
    assert prefill.att_config == gqa(bidirectional_image_spans=True)
    assert decode.att_config == gqa()
    assert not model.attention_backend_manager.has_attention(role=model.att_state_role)


def test_block_draft_layers_are_not_causal():
    model = model_with_state(
        state=FakeState(), att_backend={"gqa.prefill": "fa3", "gqa.decode": "fa3"}, mtp_mode="dflash"
    )
    model.is_mtp_draft_model = True
    layer = llama_layer()
    model.layers_infer = [layer]
    prefill, decode = LlamaTpPartModel.create_att_backends(model=model)
    assert prefill.att_config == decode.att_config == gqa(causal=False)


def test_unsupported_layer_configs_fail_when_the_model_loads(runtime_start_args):
    # FlashInfer cannot run the static FP8 per-head cache that only FA3 supports.
    scales = GQAKVScales(torch.ones(2), torch.ones(2))
    model = model_with_state(
        state=FakeState(scales=scales),
        llm_kv_type="fp8kv_sph",
        att_backend={"gqa.prefill": "flashinfer", "gqa.decode": "flashinfer"},
    )
    with pytest.raises(ValueError, match="does not support"):
        model.layers_infer = [llama_layer()]
        model.attention_backend_manager.register_backends(
            role=model.att_state_role, backends=model.create_att_backends(model=model)
        )


def test_model_factory_can_supply_a_prebuilt_backend():
    backend = FakeDecode(att_config=gqa(), runtime_config=AttRuntimeConfig(logical_layer_ids=(11,)))

    class Model(TpPartBaseModel):
        @classmethod
        def create_att_backends(cls, model: TpPartBaseModel) -> tuple[DecodeAttBackend, ...]:
            return (backend,)

    model = Model.__new__(Model)
    layer = TransformerLayerInfer(layer_num=11, network_config={})
    model.layers_infer = [layer]
    model.att_state_manager = SimpleNamespace(get_full_attention_states=lambda role: (object(),))
    model._init_attention()

    assert model.attention_backend_manager.get_backend(role="main", phase="decode", logical_layer_id=11) is backend


def test_registration_rejects_overlapping_layers_and_wrong_roles():
    config = gqa()
    runtime = AttRuntimeConfig(logical_layer_ids=(7,))
    first = FakeDecode(att_config=config, runtime_config=runtime)
    second = FakeDecode(att_config=config, runtime_config=runtime)
    draft = FakeDecode(att_config=config, role="draft0", runtime_config=runtime)
    manager = AttentionBackendManager()
    with pytest.raises(ValueError, match="layer 7 already has a decode backend"):
        manager.register_backends(role="main", backends=[first, second])
    with pytest.raises(ValueError, match="cannot register draft0:fake_decode for main"):
        manager.register_backends(role="main", backends=[draft])
    manager.register_backends(role="main", backends=[first])
    assert manager.get_backend(role="main", phase="decode", logical_layer_id=7) is first


@pytest.mark.parametrize("ssm_state_mode, decode", [("native", "triton_gdn_decode"), ("replay", "replay_gdn_decode")])
def test_qwen3next_layers_split_into_full_and_linear_attention_groups(runtime_start_args, ssm_state_mode, decode):
    from lightllm.common.attention.backends.gdn import GDNAttConfig
    from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerInfer
    from lightllm.models.qwen3next.model import Qwen3NextTpPartModel

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
        state=FakeState(), att_backend={"gdn.prefill": "triton"}, mtp_step=1, ssm_state_mode=ssm_state_mode
    )
    model.config = net
    model.create_att_backends = Qwen3NextTpPartModel.create_att_backends
    linear, full = (Qwen3NextTransformerLayerInfer(i, net) for i in range(2))
    model.layers_infer = [linear, full]
    model.attention_backend_manager.register_backends(
        role=model.att_state_role, backends=model.create_att_backends(model=model)
    )
    expected = GDNAttConfig(
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=64,
        head_v_dim=64,
        conv_kernel_dim=4,
        activation="silu",
        dtype=torch.bfloat16,
        state_mtp_step=1,
        replay=ssm_state_mode == "replay",
    )
    assert (
        layer_backend(model=model, layer=linear, phase="prefill").att_config
        == layer_backend(model=model, layer=linear, phase="decode").att_config
        == expected
    )
    assert {backend.get_name() for backend in backends(model, "decode")} == {
        "triton_gqa_decode",
        decode,
    }


def test_speculative_decode_state_reads_its_own_layer_index(monkeypatch: pytest.MonkeyPatch):
    from lightllm.common.attention.backends.gdn.triton import TritonGDNDecodeBackend
    from lightllm.common.attention.backends.gdn import GDNAttConfig

    index = torch.arange(8, dtype=torch.int32)
    recurrent = SimpleNamespace(mtp_state_index=index)
    queries = []
    resources = AttentionStateManager(request_capacity=8)

    def get_state(model_role: Any, state_type_filter: Any, logical_layer_id: Any):
        queries.append((model_role, state_type_filter, logical_layer_id))
        return recurrent

    monkeypatch.setattr(resources, "get_state", get_state)
    runtime = AttRuntimeConfig(logical_layer_ids=(9,), decode_rows_per_request=2)
    backend = TritonGDNDecodeBackend(
        att_config=GDNAttConfig(
            num_k_heads=2,
            num_v_heads=4,
            head_k_dim=64,
            head_v_dim=64,
            conv_kernel_dim=4,
            activation="silu",
            dtype=torch.bfloat16,
        ),
        role="draft1",
        runtime_config=runtime,
    )
    state = step(prefill=False, batch_size=4)
    state.att_state_manager = resources
    state.b_req_idx = torch.tensor(data=[2, 2, 3, 3], dtype=torch.int32)
    att_state = create_att_state(backend=backend, infer_state=state)
    assert queries == [("draft1", "gdn", 9)]
    torch.testing.assert_close(att_state.num_accepted_tokens, index[torch.tensor(data=[2, 3])] + 1)
    assert not hasattr(state, "req_to_mtp_state_index")


@pytest.mark.parametrize("graph_mode", list(GraphMode))
def test_backend_owns_cuda_graph_kv_capacity(graph_mode: Any):
    state = step(prefill=False)
    state.graph_mode = graph_mode
    runtime = AttRuntimeConfig(cuda_graph_max_kv_seq_len=512)
    backend = TritonGQADecodeBackend(att_config=gqa(), runtime_config=runtime)
    att_state = create_att_state(backend=backend, infer_state=state)
    assert att_state.kv_seq_len_bound == (state.max_kv_seq_len if graph_mode is GraphMode.EAGER else 512)
    assert not hasattr(state, "cuda_graph_max_kv_seq_len")
    if graph_mode is not GraphMode.EAGER:
        backend.runtime_config = AttRuntimeConfig()
        with pytest.raises(ValueError, match="kv length capacity"):
            create_att_state(backend=backend, infer_state=state)
