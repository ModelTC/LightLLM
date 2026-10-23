import time
import threading
import socket
import zlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server.router.model_infer.mode_backend.pd import nccl_kv_transporter as nccl


class FakeFrameStream:
    def __init__(self, payload, compressed=0):
        self.parts = [
            nccl._BoundedControlRPCChannel.FRAME_HEADER.pack(len(payload), compressed),
            payload + nccl._BoundedControlRPCChannel.FLUSHER,
        ]
        self.read_sizes = []

    def read(self, size):
        self.read_sizes.append(size)
        return self.parts.pop(0)


def make_channel():
    channel = nccl._NcclControlChannel.__new__(nccl._NcclControlChannel)
    channel._store_values = {}
    channel._store_value_expirations = {}
    channel._store_value_bytes = 0
    channel._store_cond = threading.Condition()
    channel._store_cleanup_stop = threading.Event()
    return channel


def test_control_rpc_frame_accepts_bounded_payload():
    stream = FakeFrameStream(zlib.compress(b"metadata"), compressed=1)
    channel = nccl._BoundedControlRPCChannel(stream)

    assert channel.recv() == b"metadata"


def test_control_rpc_frame_rejects_oversized_wire_payload_before_read(monkeypatch):
    monkeypatch.setattr(nccl, "_MAX_CONTROL_RPC_FRAME_BYTES", 4)
    stream = FakeFrameStream(b"large")
    channel = nccl._BoundedControlRPCChannel(stream)

    with pytest.raises(ValueError, match="frame exceeds 4 bytes"):
        channel.recv()

    assert stream.read_sizes == [nccl._BoundedControlRPCChannel.FRAME_HEADER.size]


def test_control_rpc_frame_rejects_oversized_decompressed_payload(monkeypatch):
    monkeypatch.setattr(nccl, "_MAX_CONTROL_RPC_FRAME_BYTES", 1024)
    stream = FakeFrameStream(zlib.compress(b"x" * 10000), compressed=1)
    channel = nccl._BoundedControlRPCChannel(stream)

    with pytest.raises(ValueError, match="frame exceeds 1024 bytes"):
        channel.recv()


def test_control_server_handles_store_and_delete_rpcs():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    control = nccl._NcclControlChannel("127.0.0.1", port, port)
    try:
        deadline = time.monotonic() + 2
        while not control._server.active and time.monotonic() < deadline:
            time.sleep(0.01)
        assert control._server.active

        control.send_store_value("peer", "127.0.0.1", port, "key", b"metadata")
        assert control._store_values == {"key": b"metadata"}

        control.send_delete_value("peer", "127.0.0.1", port, "key")
        assert control._store_values == {}
        assert control._store_value_bytes == 0
    finally:
        control.close()


def test_store_consumption_releases_accounting():
    channel = make_channel()

    channel.add_store_value("agent->peer:key", b"metadata")

    assert channel.wait_store_value("agent->peer:key") == b"metadata"
    assert channel._store_values == {}
    assert channel._store_value_bytes == 0


def test_store_rejects_oversized_key_and_value_without_mutation(monkeypatch):
    monkeypatch.setattr(nccl, "_MAX_CONTROL_STORE_KEY_BYTES", 4)
    monkeypatch.setattr(nccl, "_MAX_CONTROL_STORE_VALUE_BYTES", 3)
    channel = make_channel()
    channel.add_store_value("safe", b"ok")

    with pytest.raises(ValueError, match="key exceeds"):
        channel.add_store_value("unsafe", b"x")
    with pytest.raises(ValueError, match="value exceeds"):
        channel.add_store_value("next", b"long")

    assert channel._store_values == {"safe": b"ok"}
    assert channel._store_value_bytes == 2


def test_store_rejects_entry_limit_without_mutation(monkeypatch):
    monkeypatch.setattr(nccl, "_MAX_CONTROL_STORE_ENTRIES", 1)
    channel = make_channel()
    channel.add_store_value("first", b"")

    with pytest.raises(ValueError, match="exceeds 1 entries"):
        channel.add_store_value("second", b"")

    assert channel._store_values == {"first": b""}
    assert channel._store_value_bytes == 0


def test_store_rejects_aggregate_overflow_and_preserves_existing_value(monkeypatch):
    monkeypatch.setattr(nccl, "_MAX_CONTROL_STORE_BYTES", 4)
    monkeypatch.setattr(nccl, "_MAX_CONTROL_STORE_VALUE_BYTES", 10)
    channel = make_channel()
    channel.add_store_value("first", b"abc")
    channel.add_store_value("second", b"d")

    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        channel.add_store_value("first", b"abcd")

    assert channel._store_values == {"first": b"abc", "second": b"d"}
    assert channel._store_value_bytes == 4

    channel.add_store_value("first", b"a")
    assert channel._store_value_bytes == 2


def test_concurrent_store_writes_cannot_exceed_aggregate_limit(monkeypatch):
    monkeypatch.setattr(nccl, "_MAX_CONTROL_STORE_BYTES", 10)
    channel = make_channel()
    barrier = threading.Barrier(10)

    def add_value(index):
        barrier.wait()
        try:
            channel.add_store_value(f"key-{index}", b"xx")
            return True
        except ValueError:
            return False

    threads = [threading.Thread(target=add_value, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(channel._store_values) == 5
    assert channel._store_value_bytes == 10


def test_expired_values_release_capacity(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(nccl, "_control_store_now", lambda: now[0])
    monkeypatch.setattr(nccl, "_MAX_CONTROL_STORE_ENTRIES", 1)
    monkeypatch.setattr(nccl, "_CONTROL_STORE_CLEANUP_INTERVAL_SECONDS", 0.01)
    channel = make_channel()
    channel.add_store_value("old", b"old")

    now[0] += nccl._CONTROL_STORE_VALUE_TTL_SECONDS + 1
    cleanup_thread = threading.Thread(target=channel._expire_store_values_loop)
    cleanup_thread.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with channel._store_cond:
            if not channel._store_values:
                break
        threading.Event().wait(timeout=0.01)
    channel._store_cleanup_stop.set()
    cleanup_thread.join(timeout=1)

    assert not cleanup_thread.is_alive()
    assert channel._store_values == {}
    assert channel._store_value_bytes == 0


def test_explicit_delete_releases_capacity():
    channel = make_channel()
    channel.add_store_value("key", b"value")

    channel.delete_store_value("key")

    assert channel._store_values == {}
    assert channel._store_value_bytes == 0


def test_exposed_control_service_validates_and_deletes_remote_values():
    channel = make_channel()
    service = nccl._NcclControlService(channel)

    service.exposed_set_value("agent->peer:key", b"metadata")
    assert channel._store_values == {"agent->peer:key": b"metadata"}

    service.exposed_delete_value("agent->peer:key")
    assert channel._store_values == {}
    assert channel._store_value_bytes == 0


def test_store_delete_key_targets_remote_rendezvous_value():
    channel = Mock()
    transporter = SimpleNamespace(
        agent_name="local",
        control_channel=channel,
        _get_remote_metadata=Mock(return_value=SimpleNamespace(host_ip="127.0.0.2", control_port=4321)),
    )
    store = nccl._NcclControlStore(transporter, "peer")

    store.delete_key("send_to/peer/0")

    channel.send_delete_value.assert_called_once_with("peer", "127.0.0.2", 4321, "local->peer:send_to/peer/0")
