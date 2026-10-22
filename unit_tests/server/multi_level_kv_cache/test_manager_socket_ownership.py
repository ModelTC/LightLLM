import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server.multi_level_kv_cache.manager import MultiLevelKVCacheManager


@pytest.mark.parametrize("expired", [False, True])
def test_concurrent_workers_leave_socket_sends_to_owner_thread(expired):
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.cpu_cache_time_out = 0.5
    manager.send_to_router_queue = Queue()
    manager.send_to_router = Mock()
    manager.shm_req_manager = Mock()
    groups = [SimpleNamespace(group_req_id=i, shm_req_indexes=[]) for i in range(100)]
    start_time = time.time() - 10 if expired else time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda group: manager._handle_group_req_multi_cache_match(group, start_time), groups))

    manager.send_to_router.send_pyobj.assert_not_called()
    owner_thread = threading.get_ident()
    sent_groups = []

    def send(group, protocol):
        assert threading.get_ident() == owner_thread
        assert protocol == pickle.HIGHEST_PROTOCOL
        sent_groups.append(group.group_req_id)

    manager.send_to_router.send_pyobj.side_effect = send
    manager._send_finished_group_reqs()
    assert sorted(sent_groups) == list(range(100))
    assert manager.send_to_router_queue.empty()
    manager._send_finished_group_reqs()
    assert len(sent_groups) == 100
