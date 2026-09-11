import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import lightllm.server.generationserver.worker as model_rpc_client


class _FakeProcess:
    def __init__(self, *, alive=True, pid=1234, exitcode=None):
        self.pid = pid
        self.exitcode = exitcode
        self.alive = alive
        self.terminated = 0
        self.killed = 0
        self.joins = 0
        self.started = 0

    def start(self):
        self.started += 1

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated += 1
        self.alive = False

    def kill(self):
        self.killed += 1
        self.alive = False

    def join(self, timeout=None):
        self.joins += 1


def test_model_start_wait_rejects_a_dead_worker_without_waiting_for_timeout():
    event = SimpleNamespace(wait=lambda timeout: False)
    process = _FakeProcess(alive=False, exitcode=9)

    with pytest.raises(RuntimeError, match="exited before initialization"):
        model_rpc_client.wait_for_model_process(
            event, process, timeout=40, worker_name="generation model"
        )


def test_generation_rpc_client_close_owns_worker_and_socket(monkeypatch, tmp_path):
    class Connection:
        def __init__(self):
            self.root = SimpleNamespace(init_model=lambda: None, run_task=lambda: None)
            self.closed = 0

        def close(self):
            self.closed += 1

    monkeypatch.setattr(model_rpc_client.rpyc, "async_", lambda func: func)
    connection = Connection()
    process = _FakeProcess()
    socket_path = tmp_path / "generation.sock"
    socket_path.write_text("socket placeholder", encoding="utf-8")
    client = model_rpc_client.ModelRpcClient(connection, process, str(socket_path))

    async def scenario():
        await client.close()
        await client.close()

    asyncio.run(scenario())

    assert client.process is process
    assert client.socket_path == str(socket_path)
    assert connection.closed == 1
    assert process.terminated == 1
    assert process.killed == 0
    assert process.joins == 1
    assert not socket_path.exists()


def test_model_connect_failure_terminates_worker_and_unlinks_socket(monkeypatch):
    events = []
    process = _FakeProcess()
    original_start = process.start

    def start():
        events.append("start")
        original_start()

    process.start = start
    event = SimpleNamespace(wait=lambda timeout: True)
    monkeypatch.setattr(model_rpc_client.mp, "Event", lambda: event)
    monkeypatch.setattr(model_rpc_client.mp, "Process", lambda **_kwargs: process)
    monkeypatch.setattr(
        model_rpc_client,
        "get_env_start_args",
        lambda: SimpleNamespace(enable_torch_memory_saver=True),
    )

    class MemorySaver:
        @contextmanager
        def configure_subprocess(self):
            events.append("configure_enter")
            yield
            events.append("configure_exit")

    monkeypatch.setattr(
        model_rpc_client,
        "TorchMemorySaverWrapper",
        lambda enabled: events.append(("enabled", enabled)) or MemorySaver(),
    )
    monkeypatch.setattr(
        model_rpc_client,
        "retry",
        lambda **_kwargs: lambda func: func,
    )
    monkeypatch.setattr(
        model_rpc_client,
        "unix_connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("connect failed")),
    )

    with pytest.raises(OSError, match="connect failed"):
        asyncio.run(
            model_rpc_client.start_model_worker(
                lambda *_args: None,
                socket_prefix="test_generation",
                worker_name="generation model",
            )
        )

    assert process.started == 1
    assert events == [
        ("enabled", True),
        "configure_enter",
        "start",
        "configure_exit",
    ]
    assert process.terminated == 1
    assert process.joins == 1
