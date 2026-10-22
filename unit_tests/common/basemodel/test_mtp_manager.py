from types import SimpleNamespace

import pytest

from lightllm.common.mtp_config import FinalHiddenPolicy
import lightllm.common.basemodel.hidden_collector as hidden_collector_module
import lightllm.common.basemodel.mtp_manager as mtp_manager_module
from lightllm.common.basemodel.hidden_collector import (
    FinalHiddenCollector,
    LayerHiddenCollector,
    MtpHeadOutputCollector,
    NoopHiddenCollector,
)
from lightllm.common.basemodel.mtp_manager import MtpManager


@pytest.fixture(autouse=True)
def _reset_mtp_manager():
    MtpManager._instance = None
    yield
    MtpManager._instance = None


def _decode_tokens_per_request(monkeypatch, spec_mode, *, is_draft_model, mtp_step=7):
    args = SimpleNamespace(
        mtp_mode=spec_mode,
        mtp_step=mtp_step,
        mtp_dynamic_verify=False,
    )
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)
    return MtpManager.get_instance().get_decode_tokens_per_request(is_draft_model)


@pytest.mark.parametrize(
    "spec_mode,is_draft_model,expected",
    [
        (None, False, 1),
        ("eagle3", False, 8),
        ("eagle3", True, 1),
        ("vanilla_with_att", True, 1),
        ("vanilla_no_att", True, 1),
        ("eagle_with_att", True, 1),
        ("eagle_no_att", True, 1),
        ("dspark", True, 7),
        ("dflash", True, 7),
        ("dflash2", True, 8),
    ],
)
def test_decode_tokens_per_request(monkeypatch, spec_mode, is_draft_model, expected):
    assert _decode_tokens_per_request(monkeypatch, spec_mode, is_draft_model=is_draft_model) == expected


@pytest.mark.parametrize(
    "spec_mode,dynamic_verify,is_draft_model,expected",
    [
        ("vanilla_with_att", False, False, 8),
        ("vanilla_with_att", True, False, 1),
        ("vanilla_with_att", False, True, 1),
        ("vanilla_with_att", True, True, 1),
        ("dspark", True, True, 7),
        ("dflash", True, True, 7),
    ],
)
def test_decode_batch_alignment(monkeypatch, spec_mode, dynamic_verify, is_draft_model, expected):
    args = SimpleNamespace(
        mtp_mode=spec_mode,
        mtp_step=7,
        mtp_dynamic_verify=dynamic_verify,
    )
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)

    assert MtpManager.get_instance().get_decode_batch_alignment(is_draft_model) == expected


@pytest.mark.parametrize(
    "spec_mode,is_draft_model,expected",
    [
        (None, False, 0),
        ("eagle3", False, 7),
        ("eagle3", True, 0),
        ("vanilla_with_att", True, 0),
        ("dspark", True, 6),
        ("dflash", True, 6),
        ("dflash2", True, 7),
    ],
)
def test_decode_draft_step(monkeypatch, spec_mode, is_draft_model, expected):
    args = SimpleNamespace(mtp_mode=spec_mode, mtp_step=7, mtp_dynamic_verify=False)
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)

    assert MtpManager.get_instance().get_decode_draft_step(is_draft_model) == expected


def test_get_instance_returns_singleton(monkeypatch):
    args = SimpleNamespace(mtp_mode="eagle3", mtp_step=7, mtp_dynamic_verify=False)
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)

    assert MtpManager.get_instance() is MtpManager.get_instance()


@pytest.mark.parametrize(
    "spec_mode,is_draft_model,expected_type",
    [
        (None, False, NoopHiddenCollector),
        ("vanilla_with_att", False, FinalHiddenCollector),
        ("eagle3", False, LayerHiddenCollector),
        ("dspark", False, LayerHiddenCollector),
        ("eagle3", True, FinalHiddenCollector),
        ("dspark", True, MtpHeadOutputCollector),
        ("dflash2", True, MtpHeadOutputCollector),
    ],
)
def test_create_hidden_collector_selects_implementation(monkeypatch, spec_mode, is_draft_model, expected_type):
    args = SimpleNamespace(
        mtp_mode=spec_mode,
        mtp_step=7,
        mtp_dynamic_verify=False,
        mtp_draft_model_dir=["/models/draft"],
    )
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(hidden_collector_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(
        hidden_collector_module.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"model_type": "qwen3", "target_layer_ids": [0]}, {}),
    )
    model = SimpleNamespace(is_mtp_draft_model=is_draft_model, layers_num=2)

    prototype = MtpManager.get_instance().create_hidden_collector(model=model)
    collector = prototype.new_instance()

    assert isinstance(prototype, expected_type)
    assert isinstance(collector, expected_type)
    assert collector is not prototype


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_draft", [False, True])
@pytest.mark.parametrize(
    "model_types,modes,main_policy,draft_policy",
    [
        (("deepseek_v3",), ("vanilla_with_att", "eagle_with_att"), "POST_NORM", "POST_NORM"),
        (("glm4_moe_lite",), ("vanilla_with_att", "eagle_with_att"), "POST_NORM", "PRE_NORM"),
        (("glm5_next", "glm5_next_text"), ("vanilla_with_att", "eagle_with_att"), "POST_NORM", "POST_NORM"),
        (("qwen3_5", "qwen3_5_text"), ("vanilla_with_att", "eagle_with_att"), "POST_NORM", "POST_NORM"),
        (("qwen3_5_moe", "qwen3_5_moe_text"), ("vanilla_with_att", "eagle_with_att"), "POST_NORM", "POST_NORM"),
        (("mistral", "qwen3_moe"), ("vanilla_no_att", "eagle_no_att"), "PRE_NORM", "PRE_NORM"),
        (("qwen3",), ("eagle3",), "NONE", "PRE_NORM"),
        (("qwen3",), ("dflash2",), "NONE", "NONE"),
        (("qwen3", "qwen3_5", "qwen3_5_text"), ("dflash", "dspark"), "NONE", "NONE"),
    ],
)
def test_pair_hidden_output(tmp_path, model_types, modes, main_policy, draft_policy, enabled, is_draft):
    import json

    for model_type in model_types:
        directory = tmp_path / model_type
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
        for mode in modes:
            manager = MtpManager.__new__(MtpManager)
            manager.args = SimpleNamespace(mtp_mode=mode if enabled else None, mtp_draft_model_dir=[str(directory)])
            model = SimpleNamespace(is_mtp_draft_model=is_draft, weight_dir_=str(directory))
            expected = (
                FinalHiddenPolicy[draft_policy if is_draft else main_policy] if enabled else FinalHiddenPolicy.NONE
            )
            assert manager.get_final_hidden_policy(model) is expected


def test_same_main_uses_selected_pair_and_draft_uses_own_checkpoint(tmp_path):
    import json

    directories = []
    for model_type in ("deepseek_v3", "glm4_moe_lite"):
        directory = tmp_path / model_type
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
        directories.append(str(directory))
    manager = MtpManager.__new__(MtpManager)
    manager.args = SimpleNamespace(mtp_mode="vanilla_with_att", mtp_draft_model_dir=directories)
    main = SimpleNamespace(is_mtp_draft_model=False)
    assert manager.get_final_hidden_policy(main) is FinalHiddenPolicy.POST_NORM
    assert (
        manager.get_final_hidden_policy(SimpleNamespace(is_mtp_draft_model=True, weight_dir_=directories[0]))
        is FinalHiddenPolicy.POST_NORM
    )
    assert (
        manager.get_final_hidden_policy(SimpleNamespace(is_mtp_draft_model=True, weight_dir_=directories[1]))
        is FinalHiddenPolicy.PRE_NORM
    )

    # A different adapter can request different output from the same main object.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    other = MtpManager.__new__(MtpManager)
    other.args = SimpleNamespace(mtp_mode="eagle3", mtp_draft_model_dir=[str(tmp_path)])
    assert other.get_final_hidden_policy(main) is FinalHiddenPolicy.NONE


def test_pair_config_is_cached_and_non_mtp_does_not_load(monkeypatch):
    calls = []

    def load(directory):
        calls.append(directory)
        return {"model_type": "glm4_moe_lite"}, {}

    monkeypatch.setattr(mtp_manager_module.PretrainedConfig, "get_config_dict", load)
    manager = MtpManager.__new__(MtpManager)
    manager.args = SimpleNamespace(mtp_mode=None)
    assert manager.get_final_hidden_policy(SimpleNamespace()) is FinalHiddenPolicy.NONE
    assert calls == []
    manager.args = SimpleNamespace(mtp_mode="vanilla_with_att", mtp_draft_model_dir=["draft"])
    for _ in range(2):
        assert manager.get_final_hidden_policy(SimpleNamespace(is_mtp_draft_model=False)) is FinalHiddenPolicy.POST_NORM
        assert (
            manager.get_final_hidden_policy(SimpleNamespace(is_mtp_draft_model=True, weight_dir_="draft"))
            is FinalHiddenPolicy.PRE_NORM
        )
    assert calls == ["draft"]


@pytest.mark.parametrize("mode", ["eagle3", "dflash", "dspark"])
@pytest.mark.parametrize(
    "path",
    ["deepseek2.model:Deepseek2TpPartModel", "glm5_next.model:Glm5NextTpPartModel", "qwen3_5.model:Qwen3_5TpPartModel"],
)
def test_intermediate_hidden_output(monkeypatch, mode, path):
    from importlib import import_module

    args = SimpleNamespace(mtp_mode=mode, mtp_draft_model_dir=["/models/draft"])
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(hidden_collector_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(
        hidden_collector_module.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"model_type": "qwen3", "target_layer_ids": [0]}, {}),
    )
    module, name = path.split(":")
    cls = getattr(import_module("lightllm.models." + module), name)
    model = cls.__new__(cls)
    model.layers_num = 2
    collector = MtpManager.get_instance().create_hidden_collector(model).new_instance()
    assert isinstance(collector, LayerHiddenCollector)
    assert MtpManager.get_instance().get_final_hidden_policy(model) is FinalHiddenPolicy.NONE
