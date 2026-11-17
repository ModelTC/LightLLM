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
    socket_threads = []
    sent = Queue()
    send_started = threading.Event()
    send_allowed = threading.Event()
    send_allowed.set()
    send_socket = MagicMock()
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

    context.socket.side_effect = socket
    send_socket.connect.side_effect = lambda address: socket_threads.append(threading.get_ident())
    send_socket.send_pyobj.side_effect = send
    monkeypatch.setattr(manager_impl.zmq, "Context", lambda _: context)
    monkeypatch.setattr(
        manager_impl, "get_shm_port_args", lambda: SimpleNamespace(multi_level_kv_cache_port=1, router_port=2)
    )
    monkeypatch.setattr(manager_impl, "CpuKvCacheClient", Mock())
    monkeypatch.setattr(manager_impl, "ShmReqManager", Mock())
    monkeypatch.setattr(MultiLevelKVCacheManager, "cpu_cache_hanle_loop", lambda self: None)
    cache_manager = MultiLevelKVCacheManager(
        SimpleNamespace(zmq_mode="tcp://", enable_cpu_cache=True, enable_disk_cache=False, running_max_req_size=2)
    )
    try:
        yield cache_manager, sent, send_started, send_allowed, socket_threads
    finally:
        send_allowed.set()
        cache_manager.cpu_cache_thread.join(timeout=5)
        cache_manager.executor.shutdown(wait=False)
        assert set(socket_threads) == {threading.get_ident()}
        send_socket.connect.assert_called_once_with("tcp://127.0.0.1:2")


@pytest.mark.parametrize("expired", [False, True])
def test_concurrent_workers_send_on_task_loop_thread(manager, expired):
    cache_manager, sent, _, _, socket_threads = manager
    groups = [GroupReqIndexes(i, None, [], time.time()) for i in range(100)]
    start_time = time.time() - 10 if expired else time.time()
    worker_threads = []
    real_sleep = time.sleep

    def match(group):
        worker_threads.append(threading.get_ident())
        cache_manager._handle_group_req_multi_cache_match(group, start_time)

    def run_workers():
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(match, groups))

    def sleep(interval):
        assert interval == 0.005
        if sent.qsize() >= len(groups):
            raise StopLoop()
        real_sleep(0.001)

    cache_manager.zmq_recv_socket.recv_pyobj.side_effect = zmq.Again
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(manager_impl.time, "sleep", sleep)
        with ThreadPoolExecutor(max_workers=1) as executor:
            workers = executor.submit(run_workers)
            with pytest.raises(StopLoop):
                cache_manager.task_loop()
            workers.result(timeout=5)
    assert sorted(sent.get(timeout=5).group_req_id for _ in groups) == list(range(100))
    assert sent.empty()
    assert set(socket_threads).isdisjoint(worker_threads)


def test_send_before_recv(manager):
    cache_manager, sent, _, _, _ = manager
    pending = GroupReqIndexes(1, None, [], time.time())
    incoming = GroupReqIndexes(2, None, [], time.time())
    cache_manager.send_to_router_queue.put(pending)
    order = []
    recv_count = {"n": 0}

    def send(group, protocol):
        order.append(("send", group.group_req_id))
        sent.put(group)

    def recv(*args, **kwargs):
        order.append(("recv",))
        recv_count["n"] += 1
        if recv_count["n"] == 1:
            return incoming
        raise zmq.Again()

    cache_manager.send_to_router.send_pyobj.side_effect = send
    cache_manager.zmq_recv_socket.recv_pyobj.side_effect = recv

    def sleep(interval):
        assert interval == 0.005
        raise StopLoop()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(manager_impl.time, "sleep", sleep)
        with pytest.raises(StopLoop):
            cache_manager.task_loop()
    assert order[0] == ("send", 1)
    assert ("recv",) in order
    assert order.index(("send", 1)) < order.index(("recv",))
    assert cache_manager.recv_queue.get_nowait() is incoming
    assert sent.get_nowait() is pending


def test_blocked_send_holds_recv_on_the_same_thread(manager):
    cache_manager, sent, send_started, send_allowed, _ = manager
    groups = [GroupReqIndexes(i, None, [], time.time()) for i in range(4)]
    incoming = GroupReqIndexes(9, None, [], time.time())
    send_allowed.clear()
    cache_manager.send_to_router_queue.put(groups[0])

    def recv(*args, **kwargs):
        if recv.calls == 0:
            recv.calls += 1
            return incoming
        raise zmq.Again()

    recv.calls = 0
    cache_manager.zmq_recv_socket.recv_pyobj.side_effect = recv
    worker_blocked = threading.Event()
    put = cache_manager.send_to_router_queue.put

    def publish(group, *args, **kwargs):
        worker_blocked.set()
        put(group, *args, **kwargs)

    def fill_queue():
        assert send_started.wait(5)
        assert cache_manager.recv_queue.empty()
        cache_manager.send_to_router_queue.put(groups[1])
        cache_manager.send_to_router_queue.put(groups[2])
        cache_manager.send_to_router_queue.put = publish
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(cache_manager._handle_group_req_multi_cache_match, groups[3], time.time() - 10)
            try:
                assert worker_blocked.wait(5)
                assert cache_manager.send_to_router_queue.full()
                assert not future.done()
                assert cache_manager.recv_queue.empty()
            finally:
                send_allowed.set()
            future.result(timeout=5)

    def sleep(interval):
        assert interval == 0.005
        if sent.qsize() < len(groups):
            return
        raise StopLoop()

    filler = threading.Thread(target=fill_queue)
    filler.start()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(manager_impl.time, "sleep", sleep)
        with pytest.raises(StopLoop):
            cache_manager.task_loop()
    filler.join(timeout=5)
    assert not filler.is_alive()
    assert [sent.get(timeout=5).group_req_id for _ in range(4)] == list(range(4))
    assert cache_manager.recv_queue.get_nowait() is incoming
