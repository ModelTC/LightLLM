import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from lightllm.utils.envs_utils import get_local_dp_max_req_size


def test_model_kvargs_uses_local_dp_request_capacity():
    source = Path("lightllm/server/router/manager.py").read_text()
    module = ast.parse(source)

    for node in ast.walk(module):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "max_req_num":
                assert ast.unparse(value) == "self.local_dp_max_req_size"
                return

    raise AssertionError("max_req_num kvarg was not found")


@pytest.mark.parametrize(
    ("running_max_req_size", "dp", "nnodes", "enable_dp_prompt_cache_fetch", "expected"),
    [
        (256, 4, 1, False, 64),
        (257, 4, 1, False, 65),
        (256, 4, 2, False, 128),
        (256, 1, 1, False, 256),
        (256, 4, 1, True, 256),
    ],
)
def test_local_dp_max_req_size(
    running_max_req_size,
    dp,
    nnodes,
    enable_dp_prompt_cache_fetch,
    expected,
):
    args = SimpleNamespace(
        running_max_req_size=running_max_req_size,
        dp=dp,
        nnodes=nnodes,
        enable_dp_prompt_cache_fetch=enable_dp_prompt_cache_fetch,
    )

    assert get_local_dp_max_req_size(args) == expected


def test_request_queue_uses_local_dp_request_capacity(monkeypatch):
    from lightllm.server.router.req_queue import base_queue

    manager_module = ModuleType("lightllm.server.router.manager")
    manager_module.RouterManager = object
    monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
    monkeypatch.setattr(base_queue, "get_fixed_kv_len", lambda: 0)
    args = SimpleNamespace(
        max_total_token_num=1000,
        batch_max_tokens=4096,
        running_max_req_size=256,
        dp=4,
        nnodes=1,
        enable_dp_prompt_cache_fetch=False,
        router_token_ratio=0.0,
    )

    queue = base_queue.BaseQueue(args, SimpleNamespace(), dp_index=0, dp_size_in_node=4)

    assert queue.running_max_req_size == 64
