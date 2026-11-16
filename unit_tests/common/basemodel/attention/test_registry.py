import sys
from types import ModuleType, SimpleNamespace

import pytest

import lightllm.common.basemodel.attention.registry as registry
from lightllm.common.basemodel.attention.fa3 import neo
from lightllm.common.basemodel.attention.fa3.fp import Fa3AttBackend
from lightllm.common.basemodel.attention.fa3.fp8 import Fp8Fa3AttBackend
from lightllm.common.basemodel.attention.fa3.mla import MlaFa3AttBackend
from lightllm.common.basemodel.attention.fa3.neo import NeoFa3AttBackend
from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferAttBackend
from lightllm.common.basemodel.attention.flashinfer.fp8 import Fp8FlashInferAttBackend
from lightllm.common.basemodel.attention.flashinfer.mla import MlaFlashInferAttBackend
from lightllm.common.basemodel.attention.linear.flashqla import FlashQlaLinearAttBackend
from lightllm.common.basemodel.attention.linear.replay_gdn import ReplayGDNFlashQlaAttBackend, ReplayGDNTritonAttBackend
from lightllm.common.basemodel.attention.linear.triton import TritonLinearAttBackend
from lightllm.common.basemodel.attention.nsa.dsv4_fp8_flashmla_sparse import DeepseekV4FlashMlaFp8SparseAttBackend
from lightllm.common.basemodel.attention.nsa.flashmla_sparse import NsaFlashMlaSparseAttBackend
from lightllm.common.basemodel.attention.nsa.fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend
from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend
from lightllm.common.basemodel.attention.triton.int4kv import Int4kvTritonAttBackend
from lightllm.common.basemodel.attention.triton.int8kv import Int8kvTritonAttBackend
from lightllm.common.basemodel.attention.triton.mla import MlaTritonAttBackend
from lightllm.common.basemodel.attention.triton.neo import NeoTritonAttBackend
from lightllm.common.basemodel.attention.registry import (
    ANY_KV_TYPE,
    DECODE,
    PREFILL,
    register_att_backend,
    resolve_att_backend_class,
)


@pytest.fixture
def selection(monkeypatch):
    """Control the start args and the validation result seen by the resolver."""
    state = SimpleNamespace(
        args=SimpleNamespace(
            llm_kv_type="None",
            llm_prefill_att_backend=["auto"],
            llm_decode_att_backend=["auto"],
            mtp_step=0,
            enable_ep_moe=False,
        ),
        valid=None,  # None: every backend validates. Otherwise the set of names that validate.
        validate_calls=[],
    )

    def fake_validate(name):
        state.validate_calls.append(name)
        return state.valid is None or name in state.valid

    monkeypatch.setattr(registry, "get_env_start_args", lambda: state.args)
    monkeypatch.setattr(registry, "validate", fake_validate)
    return state


@pytest.fixture
def empty_registry(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})


@pytest.mark.parametrize(
    "family, phase, kv_type, name, expected",
    [
        ("dense", PREFILL, "None", "triton", TritonAttBackend),
        ("dense", PREFILL, "None", "fa3", Fa3AttBackend),
        ("dense", DECODE, "None", "flashinfer", FlashInferAttBackend),
        ("dense", DECODE, "int4kv", "triton", Int4kvTritonAttBackend),
        ("dense", PREFILL, "int8kv", "triton", Int8kvTritonAttBackend),
        ("dense", DECODE, "fp8kv_sph", "fa3", Fp8Fa3AttBackend),
        ("dense", PREFILL, "fp8kv_spt", "flashinfer", Fp8FlashInferAttBackend),
        ("mla", PREFILL, "None", "triton", MlaTritonAttBackend),
        ("mla", PREFILL, "None", "fa3", MlaFa3AttBackend),
        ("mla", DECODE, "None", "flashinfer", MlaFlashInferAttBackend),
        ("nsa", PREFILL, "None", "flashmla_sparse", NsaFlashMlaSparseAttBackend),
        ("nsa", DECODE, "fp8kv_dsa", "flashmla_sparse", NsaFlashMlaFp8SparseAttBackend),
        ("neo", PREFILL, "None", "triton", NeoTritonAttBackend),
        ("dsv4_nsa", PREFILL, "fp8kv_dsa", "flashmla_sparse", DeepseekV4FlashMlaFp8SparseAttBackend),
        ("dsv4_nsa", DECODE, "fp8kv_dsa", "flashmla_sparse", DeepseekV4FlashMlaFp8SparseAttBackend),
    ],
)
def test_explicit_backend_is_returned_without_validation(selection, family, phase, kv_type, name, expected):
    selection.args.llm_kv_type = kv_type
    selection.args.llm_prefill_att_backend = [name]
    selection.args.llm_decode_att_backend = [name]

    assert resolve_att_backend_class(family, phase) is expected
    assert selection.validate_calls == []


@pytest.mark.parametrize(
    "family, phase, mtp_step, expected, expected_first_validation",
    [
        ("dense", PREFILL, 0, Fa3AttBackend, "fa3"),
        ("dense", DECODE, 0, FlashInferAttBackend, "flashinfer"),
        # With MTP, decode prefers FA3.
        ("dense", DECODE, 1, Fa3AttBackend, "fa3"),
        ("dense", PREFILL, 1, Fa3AttBackend, "fa3"),
        ("mla", PREFILL, 0, MlaFa3AttBackend, "fa3"),
        ("mla", DECODE, 0, MlaFlashInferAttBackend, "flashinfer"),
        ("mla", DECODE, 2, MlaFa3AttBackend, "fa3"),
        ("nsa", DECODE, 0, NsaFlashMlaSparseAttBackend, "flashmla_sparse"),
    ],
)
def test_auto_selection_follows_family_priority(
    selection, family, phase, mtp_step, expected, expected_first_validation
):
    selection.args.mtp_step = mtp_step

    assert resolve_att_backend_class(family, phase) is expected
    assert selection.validate_calls == [expected_first_validation]


def test_auto_selection_tries_next_candidate_when_validation_fails(selection):
    selection.valid = {"triton"}

    assert resolve_att_backend_class("dense", PREFILL) is TritonAttBackend
    assert selection.validate_calls == ["fa3", "flashinfer", "triton"]


def test_auto_selection_skips_candidates_missing_for_kv_type(selection):
    selection.args.llm_kv_type = "fp8kv_spt"

    assert resolve_att_backend_class("dense", PREFILL) is Fp8FlashInferAttBackend
    assert selection.validate_calls == ["flashinfer"]


def test_auto_selection_excludes_flashinfer_with_ep_moe(selection):
    selection.args.enable_ep_moe = True

    assert resolve_att_backend_class("dense", DECODE) is Fa3AttBackend
    assert "flashinfer" not in selection.validate_calls


def test_explicit_flashinfer_is_kept_with_ep_moe(selection):
    selection.args.enable_ep_moe = True
    selection.args.llm_decode_att_backend = ["flashinfer"]

    assert resolve_att_backend_class("dense", DECODE) is FlashInferAttBackend


def test_auto_selection_falls_back_to_triton_when_nothing_validates(selection):
    selection.valid = set()

    assert resolve_att_backend_class("dense", DECODE) is TritonAttBackend


def test_auto_selection_fails_clearly_without_a_triton_fallback(selection):
    selection.valid = set()

    with pytest.raises(RuntimeError, match=r"No usable nsa prefill attention backend.*\['flashmla_sparse'\]"):
        resolve_att_backend_class("nsa", PREFILL)


def test_explicit_backend_unsupported_for_kv_type_lists_valid_choices(selection):
    selection.args.llm_kv_type = "int8kv"
    selection.args.llm_decode_att_backend = ["fa3"]

    with pytest.raises(ValueError, match=r"--llm_decode_att_backend value 'fa3'.*int8kv.*\['auto', 'triton'\]"):
        resolve_att_backend_class("dense", DECODE)


def test_kv_type_without_any_backend_lists_supported_kv_types(selection):
    selection.args.llm_kv_type = "fp8kv_dsa"

    with pytest.raises(ValueError, match=r"No mla prefill attention backend supports --llm_kv_type fp8kv_dsa"):
        resolve_att_backend_class("mla", PREFILL)


def test_allowed_rejects_an_explicit_backend_the_model_cannot_run(selection):
    selection.args.llm_decode_att_backend = ["triton"]

    with pytest.raises(ValueError, match=r"'triton'.*this model only supports \['fa3'\]"):
        resolve_att_backend_class("dense", DECODE, allowed=("fa3",))


def test_allowed_restricts_auto_selection_and_its_fallback(selection):
    assert resolve_att_backend_class("dense", DECODE, allowed=("fa3",)) is Fa3AttBackend
    assert selection.validate_calls == ["fa3"]

    selection.valid = set()
    with pytest.raises(RuntimeError, match=r"this model only supports \['fa3'\]"):
        resolve_att_backend_class("dense", DECODE, allowed=("fa3",))


def test_neo_fa3_is_skipped_by_auto_selection_when_extension_is_missing(monkeypatch, selection):
    monkeypatch.setattr(neo, "HAS_FLASH_ATTN_INTERFACE", False)

    assert resolve_att_backend_class("neo", PREFILL) is NeoTritonAttBackend
    assert selection.validate_calls == ["triton"]


def test_neo_fa3_is_auto_selected_when_extension_is_present(monkeypatch, selection):
    monkeypatch.setattr(neo, "HAS_FLASH_ATTN_INTERFACE", True)

    assert resolve_att_backend_class("neo", PREFILL) is NeoFa3AttBackend


def test_explicit_neo_fa3_without_extension_is_rejected(monkeypatch, selection):
    monkeypatch.setattr(neo, "HAS_FLASH_ATTN_INTERFACE", False)
    selection.args.llm_prefill_att_backend = ["fa3"]

    with pytest.raises(ValueError, match="dependency is missing(.|\n)*--llm_prefill_att_backend triton"):
        resolve_att_backend_class("neo", PREFILL)


def test_missing_linear_slot_arg_uses_auto_selection(monkeypatch, selection):
    flashqla = ModuleType("flash_qla")
    flashqla.chunk_gated_delta_rule = lambda **kwargs: ("flashqla", kwargs)
    monkeypatch.setitem(sys.modules, "flash_qla", flashqla)
    selection.args.llm_prefill_att_backend = ["fa3"]
    selection.args.llm_decode_att_backend = ["flashinfer"]

    backend_class = resolve_att_backend_class("gdn", PREFILL, slot_index=1)

    assert backend_class is FlashQlaLinearAttBackend
    backend = object.__new__(backend_class)
    assert backend.get_prefill_kernel()(q="q")[0] == "flashqla"
    assert selection.validate_calls == ["flashqla"]
    # Linear decode only has the Triton kernel, which is trusted without validation.
    assert resolve_att_backend_class("gdn", DECODE, slot_index=1) is TritonLinearAttBackend
    assert selection.validate_calls == ["flashqla"]


def test_explicit_linear_slot_arg_skips_auto_selection(selection):
    selection.args.llm_prefill_att_backend = ["fa3", "triton"]
    selection.args.llm_decode_att_backend = ["flashinfer", "triton"]

    assert resolve_att_backend_class("gdn", PREFILL, slot_index=1) is TritonLinearAttBackend
    assert resolve_att_backend_class("gdn", DECODE, slot_index=1) is TritonLinearAttBackend
    assert selection.validate_calls == []


def test_linear_prefill_falls_back_to_triton_when_flashqla_validation_fails(selection):
    selection.valid = set()

    assert resolve_att_backend_class("gdn", PREFILL, slot_index=1) is TritonLinearAttBackend
    assert selection.validate_calls == ["flashqla"]


def test_linear_backends_ignore_kv_type(selection):
    selection.args.llm_kv_type = "fp8kv_sph"
    selection.args.llm_prefill_att_backend = ["fa3", "flashqla"]

    assert resolve_att_backend_class("gdn", PREFILL, slot_index=1) is FlashQlaLinearAttBackend


def test_linear_decode_rejects_prefill_only_backend(selection):
    selection.args.llm_decode_att_backend = ["flashinfer", "flashqla"]

    with pytest.raises(ValueError, match=r"'flashqla' at position 1.*\['auto', 'triton'\]"):
        resolve_att_backend_class("gdn", DECODE, slot_index=1)


@pytest.mark.parametrize(
    "phase, name, expected",
    [(PREFILL, "flashqla", ReplayGDNFlashQlaAttBackend), (DECODE, "triton", ReplayGDNTritonAttBackend)],
)
def test_replay_gdn_family_selects_replay_backends(selection, phase, name, expected):
    selection.args.llm_prefill_att_backend = ["fa3", name]
    selection.args.llm_decode_att_backend = ["fa3", name]

    assert resolve_att_backend_class("gdn_replay", phase, slot_index=1) is expected


def test_dsv4_family_only_supports_fp8kv_dsa(selection):
    with pytest.raises(ValueError, match=r"No dsv4_nsa decode attention backend supports --llm_kv_type None"):
        resolve_att_backend_class("dsv4_nsa", DECODE)


def test_unlisted_family_uses_registration_order(selection, empty_registry):
    class FirstBackend:
        pass

    class SecondBackend:
        pass

    register_att_backend("custom", "first", FirstBackend, kv_types=(ANY_KV_TYPE,))
    register_att_backend("custom", "second", SecondBackend, kv_types=(ANY_KV_TYPE,))
    selection.valid = {"second"}

    assert resolve_att_backend_class("custom", PREFILL) is SecondBackend
    assert selection.validate_calls == ["first", "second"]


def test_register_rejects_a_different_backend_under_the_same_key(empty_registry):
    register_att_backend("custom", "name", "nsa.first:FirstBackend")

    with pytest.raises(ValueError, match="already registered as nsa.first:FirstBackend"):
        register_att_backend("custom", "name", "nsa.second:SecondBackend")


def test_register_accepts_the_same_backend_again(empty_registry):
    register_att_backend("custom", "name", "nsa.first:FirstBackend")
    register_att_backend("custom", "name", "nsa.first:FirstBackend")

    assert registry._REGISTRY[("custom", PREFILL)]["None"]["name"].backend == "nsa.first:FirstBackend"


def test_register_rejects_unknown_phase():
    with pytest.raises(ValueError, match="unknown attention phase"):
        register_att_backend("custom", "name", "nsa.first:FirstBackend", phases=("train",))
