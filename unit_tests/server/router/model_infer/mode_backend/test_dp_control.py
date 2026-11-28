from types import SimpleNamespace

import torch

from lightllm.server.router.model_infer.mode_backend import base_backend as base_backend_module
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.dp_backend.control_state import DPControlState, RunWay


def test_dp_req_presence_uses_one_cpu_collective(monkeypatch):
    control_group = object()
    backend = ModeBackend.__new__(ModeBackend)
    backend.dp_control_tensor = torch.zeros(2, dtype=torch.int32, device="cpu")
    calls = []

    def all_reduce(tensor, op, group, async_op):
        calls.append((tensor.tolist(), tensor.device.type, op, group, async_op))
        tensor.fill_(1)

    monkeypatch.setattr(base_backend_module.dist_group_manager, "dp_control_group", control_group)
    monkeypatch.setattr(base_backend_module, "all_reduce", all_reduce)

    has_prefill, has_decode = backend._dp_all_reduce_req_presence(prefill_reqs=[object()], decode_reqs=[])

    assert (has_prefill, has_decode) == (True, True)
    assert calls == [
        ([1, 0], "cpu", torch.distributed.ReduceOp.MAX, control_group, False),
    ]


def test_new_request_readiness_uses_cpu_control_group(monkeypatch):
    control_group = object()
    backend = ModeBackend.__new__(ModeBackend)
    backend.is_master_in_node = True
    backend.shm_reqs_io_buffer = SimpleNamespace(is_ready=lambda: True)
    backend.node_broadcast_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
    backend.node_gloo_group = control_group
    backend.node_world_size = 8
    backend.args = SimpleNamespace(node_rank=0)
    backend.is_pd_mode = False
    calls = []

    def broadcast(tensor, src, group, async_op):
        calls.append((tensor.tolist(), tensor.device.type, src, group, async_op))

    backend._read_reqs_buffer_and_init_reqs = lambda: calls.append("read")
    monkeypatch.setattr(base_backend_module, "broadcast", broadcast)

    backend._try_read_new_reqs_normal()

    assert calls == [
        ([1], "cpu", 0, control_group, False),
        "read",
    ]


def test_aggressive_dp_control_prefers_prefill():
    state = DPControlState.__new__(DPControlState)
    state.is_aggressive_schedule = True
    state.step_count = 0

    assert state.select_run_way(has_prefill=True, has_decode=True) is RunWay.PREFILL
    assert state.select_run_way(has_prefill=False, has_decode=True) is RunWay.DECODE
    assert state.select_run_way(has_prefill=False, has_decode=False) is RunWay.PASS
    assert state.step_count == 3


def test_normal_dp_control_preserves_decode_budget():
    state = DPControlState.__new__(DPControlState)
    state.is_aggressive_schedule = False
    state.decode_max_step = 1
    state.left_decode_num = 1
    state.step_count = 0

    assert state.select_run_way(has_prefill=True, has_decode=True) is RunWay.DECODE
    assert state.select_run_way(has_prefill=True, has_decode=True) is RunWay.PREFILL
    assert state.left_decode_num == 1
