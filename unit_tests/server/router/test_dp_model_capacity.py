from types import SimpleNamespace

import pytest

from lightllm.server.router.manager import resolve_model_max_req_num


def test_model_request_capacity_defaults_to_global_running_limit():
    args = SimpleNamespace(
        running_max_req_size=256,
        per_dp_max_req_size=None,
        dp=8,
        nnodes=1,
    )

    assert resolve_model_max_req_num(args) == 256


def test_model_request_capacity_accepts_balanced_per_dp_limit():
    args = SimpleNamespace(
        running_max_req_size=256,
        per_dp_max_req_size=40,
        dp=8,
        nnodes=1,
    )

    assert resolve_model_max_req_num(args) == 40


def test_model_request_capacity_rejects_less_than_balanced_share():
    args = SimpleNamespace(
        running_max_req_size=256,
        per_dp_max_req_size=31,
        dp=8,
        nnodes=1,
    )

    with pytest.raises(ValueError, match="minimum 32"):
        resolve_model_max_req_num(args)
