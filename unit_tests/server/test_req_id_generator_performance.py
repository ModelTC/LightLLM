"""Opt-in microbenchmark for request ID generation through RPyC.

Run with::

    LIGHTLLM_RUN_REQ_ID_PERF=1 pytest -q -s \
        unit_tests/server/test_req_id_generator_performance.py
"""

import os
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from rpyc.utils.server import ThreadedServer

from lightllm.server.httpserver import req_id_generator_client
from lightllm.server.httpserver.req_id_generator_client import ReqIDGenerator
from lightllm.server.httpserver.req_id_generator_server import (
    CLIENT_REQUEST_ID_RANGE_SIZE,
    MAX_BEST_OF,
    StandaloneReqIDGeneratorState,
)
from lightllm.utils.net_utils import get_abstract_unix_socket_address

RUN_PERFORMANCE_TEST = os.getenv("LIGHTLLM_RUN_REQ_ID_PERF") == "1"


class _LocalIDGenerator:
    def __init__(self):
        self.current_id = MAX_BEST_OF
        self.lock = threading.Lock()

    def generate_id(self):
        with self.lock:
            request_id = self.current_id
            self.current_id += MAX_BEST_OF
            return request_id


def _client(ipc_name):
    socket_address = get_abstract_unix_socket_address(
        "req_id_generator", ipc_name
    )
    with patch.object(
        req_id_generator_client,
        "get_req_id_generator_socket_address",
        return_value=socket_address,
    ):
        return ReqIDGenerator()


def _measure_calls(generate_id, count: int):
    latencies_ns = []
    request_ids = []
    start = time.perf_counter_ns()
    for _ in range(count):
        call_start = time.perf_counter_ns()
        request_ids.append(generate_id())
        latencies_ns.append(time.perf_counter_ns() - call_start)
    elapsed_ns = time.perf_counter_ns() - start
    return request_ids, latencies_ns, elapsed_ns


@pytest.mark.skipif(
    not RUN_PERFORMANCE_TEST,
    reason="set LIGHTLLM_RUN_REQ_ID_PERF=1 to run the request-ID microbenchmark",
)
def test_req_id_generator_rpc_performance():
    iterations = int(os.getenv("LIGHTLLM_REQ_ID_PERF_ITERATIONS", "10000"))
    client_count = 4
    assert iterations >= client_count

    ipc_name = uuid.uuid4().hex
    state = StandaloneReqIDGeneratorState()
    server = ThreadedServer(
        state,
        socket_path=get_abstract_unix_socket_address(
            "req_id_generator", ipc_name
        ),
        protocol_config={"allow_pickle": True},
    )
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    clients = [_client(ipc_name) for _ in range(client_count)]
    try:
        for _ in range(100):
            clients[0].generate_id()

        local_generator = _LocalIDGenerator()
        _, _, local_elapsed_ns = _measure_calls(local_generator.generate_id, iterations)
        request_ids, rpc_latencies_ns, rpc_elapsed_ns = _measure_calls(
            clients[0].generate_id, iterations
        )

        calls_per_client = iterations // client_count
        concurrent_call_count = calls_per_client * client_count
        concurrent_start_ns = time.perf_counter_ns()
        with ThreadPoolExecutor(max_workers=client_count) as executor:
            futures = [
                executor.submit(_measure_calls, client.generate_id, calls_per_client)
                for client in clients
            ]
            concurrent_ids = [
                request_id for future in futures for request_id in future.result()[0]
            ]
        concurrent_elapsed_ns = time.perf_counter_ns() - concurrent_start_ns

        assert len(set(request_ids)) == len(request_ids)
        assert len(set(concurrent_ids)) == len(concurrent_ids)
        assert set(request_ids).isdisjoint(concurrent_ids)
        assert all(request_id % MAX_BEST_OF == 0 for request_id in request_ids)

        local_ns_per_call = local_elapsed_ns / iterations
        rpc_ns_per_call = rpc_elapsed_ns / iterations
        rpc_calls_per_second = iterations * 1e9 / rpc_elapsed_ns
        concurrent_calls_per_second = (
            concurrent_call_count * 1e9 / concurrent_elapsed_ns
        )
        rpc_latency_us = sorted(rpc_latencies_ns)
        rpc_p50_us = statistics.median(rpc_latency_us) / 1e3
        rpc_p95_us = rpc_latency_us[int(len(rpc_latency_us) * 0.95)] / 1e3
        print(
            "request ID performance: "
            f"local={local_ns_per_call:.0f} ns/call, "
            f"ranged_client={rpc_calls_per_second:.0f} calls/s, "
            f"client_p50={rpc_p50_us:.1f} us, client_p95={rpc_p95_us:.1f} us, "
            f"client_overhead={rpc_ns_per_call / local_ns_per_call:.1f}x, "
            f"ranged_{client_count}_clients={concurrent_calls_per_second:.0f} calls/s, "
            f"range_size={CLIENT_REQUEST_ID_RANGE_SIZE}"
        )

        minimum_calls_per_second = float(
            os.getenv("LIGHTLLM_REQ_ID_MIN_CALLS_PER_SECOND", "0")
        )
        if minimum_calls_per_second:
            assert rpc_calls_per_second >= minimum_calls_per_second
    finally:
        for client in clients:
            client.connection.close()
        server.close()
        server_thread.join(timeout=5)
        assert not server_thread.is_alive()
