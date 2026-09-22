from types import SimpleNamespace

import pytest

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
        lambda _: ({"target_layer_ids": [0]}, {}),
    )
    model = SimpleNamespace(is_mtp_draft_model=is_draft_model, layers_num=2)

    prototype = MtpManager.get_instance().create_hidden_collector(model=model)
    collector = prototype.new_instance()

    assert isinstance(prototype, expected_type)
    assert isinstance(collector, expected_type)
    assert collector is not prototype
