"""An unrecoverable inference exception must reach process supervision."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("exception", ["RuntimeError", "torch.OutOfMemoryError"])
def test_inference_loop_exception_exits_worker_process(exception):
    source = f'''
import types
import torch
import lightllm.server.router.model_infer.mode_backend.dp_backend.impl as module
module.torch.cuda.set_device = lambda *_: None
module.get_current_device_id = lambda: 0
def fail():
    raise {exception}("injected unrecoverable inference failure")
backend = object.__new__(module.DPChunkedPrefillBackend)
backend.overlap_event_manager = types.SimpleNamespace(get_overlap_event_pack=fail)
backend.logger = types.SimpleNamespace(exception=lambda *_: None)
print("entered-worker", flush=True)
backend.infer_loop()
print("incorrectly-survived", flush=True)
'''
    child = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=45)
    assert "entered-worker" in child.stdout, child.stderr
    assert child.returncode == 1
    assert "incorrectly-survived" not in child.stdout
