from types import SimpleNamespace

import pytest

import lightllm.common.basemodel.mtp_manager as mtp_manager_module
from lightllm.common.basemodel.mtp_manager import MtpManager


@pytest.fixture(autouse=True)
def _reset_mtp_manager():
    MtpManager._instance = None
    yield
    MtpManager._instance = None


def _decode_batch_multiplier(monkeypatch, spec_mode, *, is_draft_model, mtp_step=7):
    args = SimpleNamespace(
        mtp_mode=spec_mode,
        mtp_step=mtp_step,
        mtp_dynamic_verify=False,
    )
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)
    return MtpManager.get_instance().get_decode_batch_multiplier(is_draft_model)


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
def test_decode_batch_multiplier(monkeypatch, spec_mode, is_draft_model, expected):
    assert _decode_batch_multiplier(monkeypatch, spec_mode, is_draft_model=is_draft_model) == expected


@pytest.mark.parametrize(
    "dynamic_verify,is_draft_model,expected",
    [
        (False, False, 8),
        (True, False, 1),
        (False, True, 1),
        (True, True, 1),
    ],
)
def test_decode_cuda_graph_grow_step_size(monkeypatch, dynamic_verify, is_draft_model, expected):
    args = SimpleNamespace(
        mtp_mode="vanilla_with_att",
        mtp_step=7,
        mtp_dynamic_verify=dynamic_verify,
    )
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)

    assert MtpManager.get_instance().get_decode_cuda_graph_grow_step_size(is_draft_model) == expected


def test_get_instance_returns_singleton(monkeypatch):
    args = SimpleNamespace(mtp_mode="eagle3", mtp_step=7, mtp_dynamic_verify=False)
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)

    assert MtpManager.get_instance() is MtpManager.get_instance()
