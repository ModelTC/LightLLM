from typing import Any
from unit_tests.common.attention.backends.step_fixture import make_backend_model

"""The selector chooses an implementation per layer config and phase, skipping implementations that reject it."""

import dataclasses
from types import SimpleNamespace
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch

from lightllm.common.attention.backends import selection
import lightllm.utils.backend_validator as backend_validator
from lightllm.common.attention.backends.gdn.selector import (
    select_gdn_prefill_backend,
    select_gdn_decode_backend,
    select_gdn_replay_decode_backend,
)
from lightllm.common.attention.backends.gqa.selector import (
    select_gqa_prefill_backend,
    select_gqa_decode_backend,
)
from lightllm.common.attention.backends.kda.selector import (
    select_kda_prefill_backend,
    select_kda_decode_backend,
    select_kda_replay_prefill_backend,
    select_kda_replay_decode_backend,
)
from lightllm.common.attention.backends.mla.selector import (
    select_mla_prefill_backend,
    select_mla_decode_backend,
)
from lightllm.common.attention.backends.sparse_mla.selector import (
    select_sparse_mla_prefill_backend,
    select_sparse_mla_decode_backend,
)
from lightllm.server.attention_backend_options import BACKEND_OVERRIDE_CHOICES
from lightllm.common.attention.backends.gqa import GQAAttConfig
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs.start_args_type import StartArgs

PHASES = ("prefill", "decode")

GQA_SELECTORS = {"prefill": select_gqa_prefill_backend, "decode": select_gqa_decode_backend}
SPARSE_MLA_SELECTORS = {"prefill": select_sparse_mla_prefill_backend, "decode": select_sparse_mla_decode_backend}


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


def recorder(monkeypatch: pytest.MonkeyPatch, passing: tuple[str, ...] = ("fa3", "flashinfer", "triton")):
    validated = []

    def validate(backend_name):
        validated.append(backend_name)
        return backend_name in passing

    monkeypatch.setattr(backend_validator, "validate", validate)
    return validated, validate


def test_backend_overrides_cover_the_registered_families():
    overrides = {key: choices[1] for key, choices in BACKEND_OVERRIDE_CHOICES.items()}
    assert overrides.keys() == {
        f"{family}.{phase}"
        for family in ("gqa", "mla", "sparse_mla", "swa_sparse_mla", "gdn", "kda")
        for phase in PHASES
    }
    parser = make_argument_parser()
    assert parser.parse_args([]).att_backend == StartArgs().att_backend == {}
    parsed = parser.parse_args(["--att_backend", *(f"{key}={name}" for key, name in overrides.items())])
    assert StartArgs(att_backend=parsed.att_backend).att_backend == overrides


@pytest.mark.parametrize(
    "values, error",
    [
        (["gqa.prefill"], "expected FAMILY.PHASE=BACKEND"),
        (["gqa.prefill="], "expected FAMILY.PHASE=BACKEND"),
        (["gqa.foo=fa3"], "unknown attention backend key"),
        (["gqa.prefill=flashkda"], "invalid attention backend"),
        (["gqa.prefill=fa3", "gqa.prefill=triton"], "duplicate attention backend key"),
        (["gqa.prefill=fa3", "--att_backend", "gqa.prefill=triton"], "duplicate attention backend key"),
    ],
)
def test_invalid_backend_overrides_are_rejected_by_cli(values: Any, error: Any, capsys: Any):
    with pytest.raises(SystemExit):
        make_argument_parser().parse_args(["--att_backend", *values])
    assert error in capsys.readouterr().err


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("name", ["triton", "fa3", "flashinfer"])
def test_named_implementation_is_used_without_validation(phase: Any, name: Any, monkeypatch: pytest.MonkeyPatch):
    validated, validate = recorder(monkeypatch=monkeypatch, passing=())
    selection = GQA_SELECTORS[phase](
        att_config=gqa(), requested=name, model=make_backend_model(), logical_layer_ids=(0,)
    )
    assert selection.get_name() == f"{name}_gqa_{phase}"
    assert selection.att_config == gqa()
    assert validated == []


def test_named_implementation_that_rejects_the_config_raises():
    with pytest.raises(ValueError, match="flashinfer.*does not support"):
        select_gqa_prefill_backend(
            att_config=gqa(sliding_window=(15, 0)),
            requested="flashinfer",
            model=make_backend_model(),
            logical_layer_ids=(0,),
        )


def test_unknown_implementation_name_raises():
    with pytest.raises(ValueError, match="not available"):
        select_gqa_decode_backend(
            att_config=gqa(), requested="flashmla_sparse", model=make_backend_model(), logical_layer_ids=(0,)
        )


def test_one_name_covers_the_quantized_triton_backend():
    config = gqa(kv_quant="int8_group", kv_quant_group_size=8)
    assert (
        select_gqa_decode_backend(
            att_config=config, requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "triton_quant_gqa_decode"
    )


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_triton_rejects_unsupported_head_dim_during_selection(phase: Any, monkeypatch: pytest.MonkeyPatch):
    config = gqa(head_dim=96)
    with pytest.raises(ValueError, match="head_dim=96"):
        GQA_SELECTORS[phase](att_config=config, requested="triton", model=make_backend_model(), logical_layer_ids=(0,))
    validated, validate = recorder(monkeypatch=monkeypatch)
    selected = GQA_SELECTORS[phase](
        att_config=config, requested="auto", model=make_backend_model(), logical_layer_ids=(0,)
    )
    expected = "fa3" if phase == "prefill" else "flashinfer"
    assert selected.get_name() == f"{expected}_gqa_{phase}"
    assert validated == [expected]


def test_triton_512_head_dim_depends_on_decode_kernel():
    mha = gqa(head_dim=512, num_kv_heads=8)
    assert (
        select_gqa_prefill_backend(
            att_config=mha, requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "triton_gqa_prefill"
    )
    with pytest.raises(ValueError, match="MHA decode does not support head_dim=512"):
        select_gqa_decode_backend(
            att_config=mha, requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
        )
    assert (
        select_gqa_decode_backend(
            att_config=gqa(head_dim=512), requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "triton_gqa_decode"
    )


@pytest.mark.parametrize("kind", ["int4_group", "int8_group"])
def test_quantized_triton_checks_the_head_dims_of_each_kernel(kind):
    small = gqa(head_dim=32, kv_quant=kind, kv_quant_group_size=8)
    assert (
        select_gqa_decode_backend(
            att_config=small, requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "triton_quant_gqa_decode"
    )
    with pytest.raises(ValueError, match="head dimensions"):
        select_gqa_prefill_backend(
            att_config=small, requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
        )
    for phase in ("prefill", "decode"):
        with pytest.raises(ValueError, match="head dimensions"):
            GQA_SELECTORS[phase](
                att_config=gqa(head_dim=96, kv_quant=kind, kv_quant_group_size=8),
                requested="triton",
                model=make_backend_model(),
                logical_layer_ids=(0,),
            )


def test_auto_follows_priority_and_validates_only_implementations_that_accept_the_config(
    monkeypatch: pytest.MonkeyPatch,
):
    validated, validate = recorder(monkeypatch=monkeypatch)
    assert (
        select_gqa_prefill_backend(
            att_config=gqa(), requested="auto", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "fa3_gqa_prefill"
    )
    assert validated == ["fa3"]

    # FA3 and FlashInfer reject an int8 cache, so neither is validated.
    validated, validate = recorder(monkeypatch=monkeypatch)
    config = gqa(kv_quant="int8_group", kv_quant_group_size=8)
    assert (
        select_gqa_decode_backend(
            att_config=config, requested="auto", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "triton_quant_gqa_decode"
    )
    assert validated == ["triton"]


def test_auto_skips_unsupported_configs_and_failed_validation(monkeypatch: pytest.MonkeyPatch):
    # FlashInfer has no sliding window, so decode moves on to FA3.
    validated, validate = recorder(monkeypatch=monkeypatch)
    selection = select_gqa_decode_backend(
        att_config=gqa(sliding_window=(15, 0)), requested="auto", model=make_backend_model(), logical_layer_ids=(0,)
    )
    assert selection.get_name() == "fa3_gqa_decode"
    assert validated == ["fa3"]

    validated, validate = recorder(monkeypatch=monkeypatch, passing=("triton",))
    assert (
        select_gqa_decode_backend(
            att_config=gqa(), requested="auto", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "triton_gqa_decode"
    )
    assert validated == ["flashinfer", "fa3", "triton"]


def test_ep_mode_skips_flashinfer(monkeypatch: pytest.MonkeyPatch):
    _, validate = recorder(monkeypatch=monkeypatch)
    model = make_backend_model()
    model.args.enable_ep_moe = True
    selected = select_gqa_decode_backend(att_config=gqa(), model=model, logical_layer_ids=(0,))
    assert selected.get_name() == "fa3_gqa_decode"


def test_auto_falls_back_to_triton_without_validation(monkeypatch: pytest.MonkeyPatch):
    validated, validate = recorder(monkeypatch=monkeypatch, passing=())
    assert (
        select_gqa_prefill_backend(
            att_config=gqa(), requested="auto", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "triton_gqa_prefill"
    )
    assert validated == ["fa3", "flashinfer", "triton"]

    # Only FA3 runs an FP8 per-head cache; without it nothing is left.
    with pytest.raises(ValueError, match="no prefill attention implementation supports"):
        select_gqa_prefill_backend(
            att_config=gqa(kv_quant="fp8_per_head"),
            requested="auto",
            model=make_backend_model(),
            logical_layer_ids=(0,),
        )


@pytest.mark.parametrize("mtp_step, expected", [(0, "flashinfer"), (2, "fa3")])
def test_decode_priority_puts_fa3_first_with_mtp(monkeypatch: pytest.MonkeyPatch, mtp_step: Any, expected: Any):
    from lightllm.common.attention.backends.mla import MLADecodeAttConfig

    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    model = make_backend_model()
    model.args.mtp_step = mtp_step
    mla = MLADecodeAttConfig(
        num_heads=16, qk_rope_head_dim=64, kv_lora_rank=512, softmax_scale=0.1, dtype=torch.bfloat16
    )
    for select, config in ((select_gqa_decode_backend, gqa()), (select_mla_decode_backend, mla)):
        validated, validate = recorder(monkeypatch=monkeypatch)
        select(att_config=config, model=model, logical_layer_ids=(0,))
        assert validated == [expected]


def test_auto_reports_all_rejected_backends():
    with pytest.raises(ValueError) as error:
        select_gqa_prefill_backend(
            att_config=gqa(dtype=torch.float16, kv_quant="fp8_per_head"),
            requested="auto",
            model=make_backend_model(),
            logical_layer_ids=(0,),
        )
    for reason in ("needs a BF16 model", "flashinfer_gqa_prefill", "triton_gqa_prefill"):
        assert reason in str(error.value)


def _select_with_missing_flashqla(rank, store):
    from lightllm.common.attention.backends.gdn import GDNAttConfig
    from lightllm.common.attention.backends.gdn.flashqla import FlashQLAGDNPrefillBackend

    torch.distributed.init_process_group(
        "gloo", init_method=store, rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:

        def load_kernel():
            if rank == 1:
                raise ImportError("missing FlashQLA")
            return object()

        def validate(backend_name):
            torch.distributed.barrier()
            return True

        config = GDNAttConfig(
            num_k_heads=2,
            num_v_heads=4,
            head_k_dim=64,
            head_v_dim=64,
            conv_kernel_dim=4,
            activation="silu",
            dtype=torch.bfloat16,
        )
        with patch.object(FlashQLAGDNPrefillBackend, "chunk_gated_delta_rule", staticmethod(load_kernel)):
            with patch.object(backend_validator, "validate", validate):
                choice = select_gdn_prefill_backend(
                    att_config=config, requested="auto", model=make_backend_model(), logical_layer_ids=(0,)
                )
        assert choice.get_name() == "triton_gdn_prefill"
    finally:
        torch.distributed.destroy_process_group()


def test_auto_falls_back_on_every_rank_if_one_cannot_load_flashqla(tmp_path):
    torch.multiprocessing.spawn(_select_with_missing_flashqla, args=((tmp_path / "store").as_uri(),), nprocs=2)


def test_unregistered_layer_config_type_is_rejected():
    with pytest.raises(NotImplementedError, match="is not a gqa attention config"):
        select_gqa_prefill_backend(
            att_config=object(), requested="auto", model=make_backend_model(), logical_layer_ids=(0,)
        )


def test_mla_phases_have_their_own_layer_config_types():
    from lightllm.common.attention.backends.mla import MLADecodeAttConfig, MLAPrefillAttConfig

    common = dict(num_heads=16, qk_rope_head_dim=64, softmax_scale=0.1, dtype=torch.bfloat16)
    prefill = MLAPrefillAttConfig(
        num_heads=common["num_heads"],
        qk_nope_head_dim=common.get("qk_nope_head_dim", 128),
        qk_rope_head_dim=common["qk_rope_head_dim"],
        v_head_dim=common.get("v_head_dim", 128),
        softmax_scale=common["softmax_scale"],
        dtype=common["dtype"],
        causal=common.get("causal", True),
    )
    decode = MLADecodeAttConfig(
        num_heads=common["num_heads"],
        kv_lora_rank=common.get("kv_lora_rank", 512),
        qk_rope_head_dim=common["qk_rope_head_dim"],
        softmax_scale=common["softmax_scale"],
        dtype=common["dtype"],
        causal=common.get("causal", True),
    )
    assert (
        select_mla_prefill_backend(
            att_config=prefill, requested="flashinfer", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "flashinfer_mla_prefill"
    )
    assert (
        select_mla_decode_backend(
            att_config=decode, requested="fa3", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "fa3_mla_decode"
    )
    triton = select_mla_decode_backend(
        att_config=decode, requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
    )
    assert triton.get_name() == "triton_mla_decode"
    assert triton.att_config is decode
    with pytest.raises(ValueError, match="no decode attention implementations"):
        select_mla_decode_backend(
            att_config=prefill, requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
        )


@pytest.mark.parametrize(
    "kv_cache_format, rope, expected",
    [
        ("bf16", 64, ("flashmla_sparse_mla_prefill", "fa3_sparse_mla_decode")),
        ("fp8_ds_mla", 64, ("flashmla_sparse_mla_prefill", "flashmla_sparse_mla_decode")),
        ("bf16", 0, ("flashmla_sparse_mla_prefill", "fa3_sparse_mla_decode")),
        ("fp8_group64_e8m0", 0, ("fa3_sparse_mla_prefill", "flashmla_sparse_mla_decode")),
    ],
)
def test_sparse_mla_runs_flashmla_where_it_supports_the_cache_and_fa3_elsewhere(kv_cache_format, rope, expected):
    from lightllm.common.attention.backends.sparse_mla import SparseMLAAttConfig

    config = SparseMLAAttConfig(
        num_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=rope,
        softmax_scale=0.1,
        dtype=torch.bfloat16,
        kv_cache_format=kv_cache_format,
        topk=2048,
    )
    names = tuple(
        SPARSE_MLA_SELECTORS[phase](
            att_config=config, requested="flashmla_sparse", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        for phase in PHASES
    )
    assert names == expected
    with pytest.raises(ValueError, match="not available"):
        select_sparse_mla_decode_backend(
            att_config=config, requested="fa3", model=make_backend_model(), logical_layer_ids=(0,)
        )


@pytest.mark.parametrize("replay, decode", [(False, "triton_gdn_decode"), (True, "replay_gdn_decode")])
def test_linear_attention_takes_triton_without_validation_and_follows_the_state_kind(
    replay: Any, decode: Any, monkeypatch: pytest.MonkeyPatch
):
    from lightllm.common.attention.backends.gdn import GDNAttConfig

    config = GDNAttConfig(
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=64,
        head_v_dim=64,
        conv_kernel_dim=4,
        activation="silu",
        dtype=torch.bfloat16,
        replay=replay,
    )
    validated, validate = recorder(monkeypatch=monkeypatch, passing=())
    # FlashQLA 在这里未通过验证；Triton 则直接采用而未进行验证。
    selection = select_gdn_prefill_backend(
        att_config=config, requested="auto", model=make_backend_model(), logical_layer_ids=(0,)
    )
    assert selection.get_name() == "triton_gdn_prefill"
    assert validated == (["flashqla"] if _imports("flash_qla") else [])
    select = select_gdn_replay_decode_backend if replay else select_gdn_decode_backend
    selection = select(att_config=config, requested="auto", model=make_backend_model(), logical_layer_ids=(0,))
    assert selection.get_name() == decode
    # 每个解码实现运行于一种状态类型上。
    with pytest.raises(NotImplementedError, match="replay"):
        type(selection)(dataclasses.replace(config, replay=not replay))


def _imports(module):
    try:
        __import__(module)
    except Exception:
        return False
    return True


@pytest.mark.parametrize("name", ["triton", "flashkda"])
@pytest.mark.parametrize("replay", [False, True])
def test_flashkda_names_the_kda_backends_of_the_state_kind(monkeypatch, name, replay):
    from lightllm.common.attention.backends.kda import KDAAttConfig
    from lightllm.common.attention.backends.kda import triton as kda_triton

    monkeypatch.setattr(kda_triton.TritonKDAPrefillBackend, "_check_prefill_kernel", lambda att_config: None)
    config = KDAAttConfig(
        num_heads=4,
        head_dim=128,
        conv_kernel_size=4,
        lower_bound=-5.0,
        dtype=torch.bfloat16,
        replay=replay,
        prefill_kernel=name,
    )
    expected = "replay_kda_prefill" if replay else "triton_kda_prefill"
    select_prefill = select_kda_replay_prefill_backend if replay else select_kda_prefill_backend
    select_decode = select_kda_replay_decode_backend if replay else select_kda_decode_backend
    backend = select_prefill(att_config=config, model=make_backend_model(), logical_layer_ids=(0,))
    assert backend.get_name() == expected
    assert backend.att_config is config
    with pytest.raises(ValueError, match="not available"):
        select_decode(att_config=config, requested="flashkda", model=make_backend_model(), logical_layer_ids=(0,))


@pytest.mark.parametrize("requested", ["auto", "triton"])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("family, phase", [("gdn", "decode"), ("kda", "prefill"), ("kda", "decode")])
def test_selector_loads_only_backends_for_the_requested_state_mode(
    monkeypatch: pytest.MonkeyPatch, family: Any, phase: Any, replay: Any, requested: Any
):
    from lightllm.common.attention.backends.gdn import GDNAttConfig
    from lightllm.common.attention.backends.kda import KDAAttConfig

    if family == "gdn":
        config = GDNAttConfig(
            num_k_heads=2,
            num_v_heads=4,
            head_k_dim=64,
            head_v_dim=64,
            conv_kernel_dim=4,
            activation="silu",
            dtype=torch.bfloat16,
            replay=replay,
        )
        native, replay_select = select_gdn_decode_backend, select_gdn_replay_decode_backend
    else:
        config = KDAAttConfig(
            num_heads=4,
            head_dim=128,
            conv_kernel_size=4,
            lower_bound=-5.0,
            dtype=torch.bfloat16,
            replay=replay,
            prefill_kernel=requested,
        )
        if phase == "prefill":
            native, replay_select = select_kda_prefill_backend, select_kda_replay_prefill_backend
        else:
            native, replay_select = select_kda_decode_backend, select_kda_replay_decode_backend
    load = selection._load
    loaded = []

    def record_load(path: Any):
        loaded.append(path)
        return load(path)

    monkeypatch.setattr(selection, "_load", record_load)
    select = replay_select if replay else native
    incompatible = native if replay else replay_select
    with pytest.raises(ValueError, match="requires att_config.replay="):
        if family == "kda" and phase == "prefill":
            incompatible(att_config=config, model=make_backend_model(), logical_layer_ids=(0,))
        else:
            incompatible(att_config=config, requested=requested, model=make_backend_model(), logical_layer_ids=(0,))
    assert loaded == []
    if family == "kda" and phase == "prefill":
        backend = select(att_config=config, model=make_backend_model(), logical_layer_ids=(0,))
    else:
        backend = select(att_config=config, requested=requested, model=make_backend_model(), logical_layer_ids=(0,))
    mode = "Replay" if replay else "Triton"
    assert len(loaded) == 1
    assert loaded[0].endswith(f":{mode}{family.upper()}{phase.capitalize()}Backend")
    assert backend.att_config.replay == replay


def test_importing_gqa_selector_loads_only_gqa_selection_policy():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = """
import sys
import lightllm.common.attention.backends.gqa.selector
allowed = {"gqa", "gqa.selector", "selection"}
prefix = "lightllm.common.attention.backends."
impls = [name for name in sys.modules if name.startswith(prefix) and name[len(prefix):] not in allowed]
assert not impls, impls
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_named_implementation_loads_only_its_backends(monkeypatch):
    loaded = []
    real_load = selection._load
    monkeypatch.setattr(selection, "_load", lambda path: loaded.append(path) or real_load(path))
    select_gqa_decode_backend(att_config=gqa(), requested="fa3", model=make_backend_model(), logical_layer_ids=(0,))
    assert loaded == ["lightllm.common.attention.backends.gqa.fa3:FA3GQADecodeBackend"]
