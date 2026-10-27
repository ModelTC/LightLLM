import pytest

from lightllm.utils.pd_timeout_utils import (
    get_pd_handshake_timeout_seconds,
    get_pd_kv_transfer_timeout_seconds,
)


@pytest.mark.parametrize("resource", [0, 20, 60, 200, 600])
def test_outer_handshake_covers_both_node_resource_waits(resource):
    master = get_pd_handshake_timeout_seconds(resource)
    prefill = get_pd_handshake_timeout_seconds(resource, prefill=True)
    assert master >= 2 * resource + 15
    assert prefill > master
    assert master >= 180


def test_200s_policy_is_not_cut_off_by_old_180s_handshake():
    assert get_pd_handshake_timeout_seconds(200) == 415
    assert get_pd_handshake_timeout_seconds(200, prefill=True) == 430


def test_transfer_timeout_is_configurable_without_changing_admission(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_KV_TRANSFER_TIMEOUT_SECONDS", "600")
    assert get_pd_kv_transfer_timeout_seconds() == 600
    assert get_pd_handshake_timeout_seconds(200) == 415


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_transfer_timeout_rejected(monkeypatch, value):
    monkeypatch.setenv("LIGHTLLM_PD_KV_TRANSFER_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        get_pd_kv_transfer_timeout_seconds()
