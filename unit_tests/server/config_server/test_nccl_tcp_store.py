"""The config HTTP worker must be able to start NCCL rendezvous stores."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import pytest


@pytest.mark.parametrize("bind_host", ["127.0.0.1", "0.0.0.0"])
def test_tcp_store_can_start_from_daemonic_http_worker(bind_host):
    code = textwrap.dedent(
        """
        import datetime
        import multiprocessing
        import socket
        import torch.distributed as dist
        from lightllm.server.config_server.nccl_tcp_store import start_tcp_store_server

        # Hypercorn's worker uses this multiprocessing daemon flag. The former
        # multiprocessing.Process launch raised before creating a store here.
        multiprocessing.current_process().daemon = True
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        process = start_tcp_store_server({bind_host!r}, port)
        try:
            store = dist.TCPStore(
                host_name='127.0.0.1', port=port, world_size=None, is_master=False,
                timeout=datetime.timedelta(seconds=25), use_libuv=True,
            )
            store.set('config-server-test', 'ready')
            assert store.get('config-server-test') == b'ready'
            assert process.poll() is None
        finally:
            process.kill()
            process.wait(timeout=10)
        """
    ).format(bind_host=bind_host)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
