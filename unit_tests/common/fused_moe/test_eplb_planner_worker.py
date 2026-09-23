import os
import signal
import subprocess
import sys
import time

import pytest
import torch

from lightllm.common.eplb_planner_worker import EPLBPlannerWorker
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_planner import plan_eplb_candidate

FULL_LOAD = torch.ones((1, 1, 1, 8), dtype=torch.int64)
FULL_CURRENT = torch.arange(8, dtype=torch.int64).view(1, 8, 1)
FULL = dict(
    full_layout=True,
    world_size=8,
    node_world_size=8,
    num_redundant_experts_per_rank=0,
    expert_alignment=128,
    stickiness=0.1,
)
RED_LOAD = torch.ones((1, 1, 1, 8), dtype=torch.int64)
RED_CURRENT = torch.tensor([[[1], [2], [3], [4], [5], [6], [7], [0]]])
RED = dict(
    full_layout=False,
    world_size=8,
    node_world_size=8,
    num_redundant_experts_per_rank=1,
    expert_alignment=128,
    stickiness=0.1,
)


def dead(pid):
    try:
        return open(f"/proc/{pid}/stat").read().split()[2] == "Z"
    except FileNotFoundError:
        return True


def test_worker_persistent_full_redundant_and_cpu():
    worker = EPLBPlannerWorker(timeout_s=30)
    pid = worker.info["pid"]
    try:
        for load, current, settings in ((FULL_LOAD, FULL_CURRENT, FULL), (RED_LOAD, RED_CURRENT, RED)):
            direct = plan_eplb_candidate(load, current, **settings)
            actual, _ = worker.plan(load, current, **settings)
            assert torch.equal(direct, actual)
        assert worker.info["pid"] == pid and not worker.info["cuda_initialized"]
    finally:
        worker.close()
    assert dead(pid)


def test_worker_rejects_concurrent_and_timeout_cleans_child():
    worker = EPLBPlannerWorker(timeout_s=30)
    pid = worker.info["pid"]
    try:
        assert worker._rpc_lock.acquire(False)
        with pytest.raises(RuntimeError, match="one in-flight"):
            worker.plan(FULL_LOAD, FULL_CURRENT, **FULL)
        worker._rpc_lock.release()
        assert worker._process.is_alive()
        os.kill(pid, signal.SIGSTOP)
        worker._timeout_s = 0.05
        with pytest.raises(TimeoutError):
            worker.plan(FULL_LOAD, FULL_CURRENT, **FULL)
    finally:
        worker.close()
    assert dead(pid)


def test_worker_error_and_parent_watchdog_exit():
    worker = EPLBPlannerWorker(timeout_s=30)
    pid = worker.info["pid"]
    try:
        with pytest.raises(RuntimeError):
            worker.plan(FULL_LOAD, FULL_CURRENT.repeat(1, 1, 2), **FULL)
    finally:
        worker.close()
    assert dead(pid)
    code = """import os
from lightllm.common.eplb_planner_worker import EPLBPlannerWorker
worker = EPLBPlannerWorker()
print(worker.info["pid"], flush=True)
os._exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=45, check=True)
    child = next(int(line) for line in result.stdout.splitlines() if line.strip().isdigit())
    until = time.monotonic() + 5
    while not dead(child) and time.monotonic() < until:
        time.sleep(0.1)
    assert dead(child)
