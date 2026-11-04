import subprocess
import sys

import pytest


WORKER = """
import sys
import threading
from types import SimpleNamespace
from lightllm.server.router.model_infer import model_rpc

model_rpc.start_parent_check_thread = lambda: None
model_rpc.get_unique_server_name = lambda: 'failure-test'
model_rpc.ModelRpcServer = lambda *args: SimpleNamespace(close=lambda: print('model-closed', flush=True))

class Server:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        if sys.argv[1] == 'failure':
            def fail():
                raise RuntimeError('injected inference failure')
            threading.Thread(target=threading.Event().wait, daemon=True).start()
            threading.Thread(target=fail, name='inference-test', daemon=True).start()
            threading.Event().wait(60)

    def close(self):
        print('server-closed', flush=True)

model_rpc.ThreadedServer = Server
model_rpc._init_env(None, 0, 0, 1, None, '/unused', SimpleNamespace(set=lambda: None))
"""


@pytest.mark.parametrize("mode", ["normal", "failure"])
def test_rpc_worker_thread_failure_reaches_process_supervision(mode):
    result = subprocess.run([sys.executable, "-c", WORKER, mode], capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    if mode == "failure":
        assert result.returncode == 1, output
        assert "fatal background thread failure in inference-test" in output
        assert "injected inference failure" in output
        # A failed CUDA worker cannot safely run normal Graph/collective cleanup.
        assert "model-closed" not in output
    else:
        assert result.returncode == 0, output
        assert "server-closed" in output and "model-closed" in output
