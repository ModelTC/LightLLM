import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import zmq

from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.server.multi_level_kv_cache import manager as manager_impl
from lightllm.server.multi_level_kv_cache.manager import MultiLevelKVCacheManager


class StopLoop(BaseException):
    pass


@pytest.fixture
def manager(monkeypatch):
    stop = object()
    socket_threads = []
    sent = Queue()
    errors = Queue()
    send_started = threading.Event()
    send_allowed = threading.Event()
    send_allowed.set()
    send_socket = MagicMock()
    send_socket.__enter__.return_value = send_socket
    send_socket.connect.side_effect = lambda address: socket_threads.append(threading.get_ident())
    send_socket.__exit__.side_effect = lambda *args: socket_threads.append(threading.get_ident())
    recv_socket = Mock()
    context = Mock()

    def socket(kind):
        if kind == zmq.PULL:
            return recv_socket
        assert kind == zmq.PUSH
        socket_threads.append(threading.get_ident())
        return send_socket

    def send(group, protocol):
        socket_threads.append(threading.get_ident())
        assert protocol == pickle.HIGHEST_PROTOCOL
        send_started.set()
        assert send_allowed.wait(5)
        sent.put(group)

    class StoppableQueue(Queue):
        def get(self, *args, **kwargs):
            item = super().get(*args, **kwargs)
            if item is stop:
                raise StopLoop()
            return item

    send_loop = MultiLevelKVCacheManager.send_loop

    def run_sender(self):
        try:
            send_loop(self)
        except StopLoop:
            pass
        except BaseException as exc:
            errors.put(exc)

    context.socket.side_effect = socket
    send_socket.send_pyobj.side_effect = send
    monkeypatch.setattr(manager_impl.zmq, "Context", lambda _: context)
    monkeypatch.setattr(manager_impl, "Queue", StoppableQueue)
    monkeypatch.setattr(
        manager_impl, "get_shm_port_args", lambda: SimpleNamespace(multi_level_kv_cache_port=1, router_port=2)
    )
    monkeypatch.setattr(manager_impl, "CpuKvCacheClient", Mock())
    monkeypatch.setattr(manager_impl, "ShmReqManager", Mock())
    monkeypatch.setattr(MultiLevelKVCacheManager, "cpu_cache_hanle_loop", lambda self: None)
    monkeypatch.setattr(MultiLevelKVCacheManager, "send_loop", run_sender)
    manager = MultiLevelKVCacheManager(
        SimpleNamespace(zmq_mode="tcp://", enable_cpu_cache=True, enable_disk_cache=False, running_max_req_size=2)
    )
    try:
        yield manager, sent, send_started, send_allowed
    finally:
        send_allowed.set()
        manager.send_to_router_queue.put(stop, timeout=5)
        manager.send_to_router_thread.join(timeout=5)
        manager.cpu_cache_thread.join(timeout=5)
        manager.executor.shutdown()
        assert not manager.send_to_router_thread.is_alive()
        assert errors.empty(), list(errors.queue)
        assert set(socket_threads) == {manager.send_to_router_thread.ident}
        send_socket.connect.assert_called_once_with("tcp://127.0.0.1:2")
        send_socket.__exit__.assert_called_once()


@pytest.mark.parametrize("expired", [False, True])
def test_concurrent_workers_leave_socket_sends_to_owner_thread(manager, expired):
    manager, sent, _, _ = manager
    groups = [GroupReqIndexes(i, None, [], time.time()) for i in range(100)]
    start_time = time.time() - 10 if expired else time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda group: manager._handle_group_req_multi_cache_match(group, start_time), groups))

    assert sorted(sent.get(timeout=5).group_req_id for _ in groups) == list(range(100))
    assert sent.empty()
    assert manager.send_to_router_thread.ident != threading.get_ident()


def test_send_while_receiver_sleeps(manager, monkeypatch):
    manager, sent, _, _ = manager
    group = GroupReqIndexes(0, None, [], time.time())
    receiver_sleeping = threading.Event()
    manager.zmq_recv_socket.recv_pyobj.side_effect = zmq.Again()

    def worker():
        assert receiver_sleeping.wait(5)
        manager._handle_group_req_multi_cache_match(group, time.time())

    def sleep(interval):
        assert interval == 0.01
        receiver_sleeping.set()
        assert sent.get(timeout=5) is group
        raise StopLoop()

    monkeypatch.setattr(manager_impl.time, "sleep", sleep)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        with pytest.raises(StopLoop):
            manager.recv_loop()
        future.result(timeout=5)


def test_receive_under_send_backpressure(manager, monkeypatch):
    manager, sent, send_started, send_allowed = manager
    groups = [GroupReqIndexes(i, None, [], time.time()) for i in range(5)]
    send_allowed.clear()
    manager.send_to_router_queue.put(groups[0])
    assert send_started.wait(5)
    assert manager.send_to_router_queue.maxsize == manager.args.running_max_req_size
    manager.send_to_router_queue.put(groups[1])
    manager.send_to_router_queue.put(groups[2])
    worker_publishing = threading.Event()
    put = manager.send_to_router_queue.put

    def publish(group, *args, **kwargs):
        worker_publishing.set()
        put(group, *args, **kwargs)

    monkeypatch.setattr(manager.send_to_router_queue, "put", publish)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(manager._handle_group_req_multi_cache_match, groups[3], time.time() - 10)
        try:
            assert worker_publishing.wait(5)
            assert manager.send_to_router_queue.full()
            assert not future.done()
            manager.zmq_recv_socket.recv_pyobj.side_effect = [groups[4], zmq.Again(), StopLoop()]
            with pytest.raises(StopLoop):
                manager.recv_loop()
            assert manager.recv_queue.get_nowait() is groups[4]
            assert sent.empty()
        finally:
            send_allowed.set()
        future.result(timeout=5)
    assert [sent.get(timeout=5).group_req_id for _ in range(4)] == list(range(4))
