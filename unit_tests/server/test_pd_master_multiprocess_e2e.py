"""Real PD master service tests with CPU-only P/D protocol peers.

The functional cases cover worker discovery, HTTP generation, SSE, busy replies,
cancellation, shared IDs and metrics, health aggregation, and private channel
isolation. The opt-in performance case compares the same end-to-end protocol path
with one worker and multiple workers without loading model weights.
"""

import asyncio
from contextlib import AsyncExitStack
import json
import os
import pickle
import statistics
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import websockets
from prometheus_client.parser import text_string_to_metric_families

from lightllm.server.core.objs import FinishStatus
from lightllm.server.httpserver import pd_loop
from lightllm.server.pd_io_struct import ObjType, PDDecodeNodeInfo, PDUpKVStatus
from unit_tests.server.pd_master_test_utils import running_pd_master


class FakePDPeers:
    """Drive the actual registration, prompt-ID, KV status, and token channels."""

    def __init__(self, workers, decode_delay=0.02):
        self.workers = workers
        self.decode_delay = decode_delay
        self.stack = AsyncExitStack()
        self.tasks = []
        self.requests = []
        self.prefill_done = {}
        self.generations = {}
        self.generation_errors = []
        self.aborted = set()
        self.blocked_prefills = []
        self.hold_prompt_ids = False
        self.hold_decode = False
        self.busy_prefill_requests = 0
        self.node_addresses = {"prefill": "prefill.internal:18001", "decode": "decode.internal:18002"}

    async def __aenter__(self):
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            for endpoint in self.workers:
                owner = endpoint["node_id"]
                response = await client.get("http://" + endpoint["host_ip_port"] + "/get_server_info")
                response.raise_for_status()
                start_args = response.json()
                kv = await self.stack.enter_async_context(
                    websockets.connect("ws://" + endpoint["host_ip_port"] + "/kv_move_status", proxy=None)
                )
                for mode, node_id in (("prefill", 501), ("decode", 502)):
                    websocket = await self.stack.enter_async_context(
                        websockets.connect("ws://" + endpoint["host_ip_port"] + "/pd_register", proxy=None)
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "node_id": node_id,
                                "client_ip_port": self.node_addresses[mode],
                                "mode": mode,
                                "start_args": dict(start_args, run_mode=mode),
                                "connection_id": f"{mode}-{owner}",
                            }
                        )
                    )
                    self.tasks.append(asyncio.create_task(self._run(mode, owner, websocket, kv)))
        return self

    async def __aexit__(self, *exc):
        tasks = [*self.tasks, *self.generations.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.stack.aclose()

    def _event(self, request_id):
        return self.prefill_done.setdefault(request_id, asyncio.Event())

    async def _token(self, websocket, mode, owner, request_id, index, text, finish):
        metadata = {
            "id": 2 if index == 1 else 3,
            "logprob": -0.1,
            "logprobs": None,
            "prompt_tokens": 3,
            "count_output_tokens": index,
            "node_mode": mode,
            "prompt_cache_len": 0,
        }
        await websocket.send(
            pickle.dumps(
                (
                    ObjType.TOKEN_PACKS,
                    [(request_id, text, metadata, FinishStatus(finish))],
                    {
                        "connection_key": self.node_addresses[mode] + "/dpNone",
                        "total_token_usage_rate": 0.1,
                    },
                )
            )
        )

    async def _run(self, mode, owner, websocket, kv):
        async for data in websocket:
            message = pickle.loads(data)
            if message[0] == ObjType.ABORT:
                request_id = message[1]
                self.aborted.add((mode, owner, request_id))
                task = self.generations.get((owner, request_id))
                if mode == "decode" and task is not None:
                    task.cancel()
                continue
            if mode == "prefill":
                if message[0] == ObjType.REQ:
                    _, params, _ = message[1]
                    assert params.pd_master_node_id.get() == owner
                    if self.busy_prefill_requests:
                        self.busy_prefill_requests -= 1
                        await websocket.send(
                            pickle.dumps(
                                (ObjType.PD_UPLOAD_SERVER_BUSY, params.group_request_id, "test capacity exhausted")
                            )
                        )
                        continue
                    if self.hold_prompt_ids:
                        self.blocked_prefills.append((owner, params.group_request_id))
                        continue
                    await websocket.send(
                        pickle.dumps((ObjType.PD_UPLOAD_PREFILL_PROMPT_IDS, params.group_request_id, [2, 3, 2]))
                    )
                elif message[0] == ObjType.PD_REQ_DECODE_NODE_INFO:
                    request_id = message[1]
                    await self._token(websocket, mode, owner, request_id, 1, "hello", FinishStatus.FINISHED_LENGTH)
                    self._event(request_id).set()
            elif message[0] == ObjType.REQ:
                task = asyncio.create_task(self._decode(owner, websocket, kv, message[1]))
                self.generations[(owner, message[1][1].group_request_id)] = task
                task.add_done_callback(self._generation_finished)

    def _generation_finished(self, task):
        if not task.cancelled() and task.exception() is not None:
            self.generation_errors.append(task.exception())

    async def _decode(self, owner, websocket, kv, request):
        prompt_ids, params, _ = request
        request_id = params.group_request_id
        assert params.pd_master_node_id.get() == owner
        self.requests.append((owner, request_id))
        info = PDDecodeNodeInfo(
            decode_node_id=502,
            pd_master_node_id=owner,
            agent_name="test",
            agent_metadata=b"",
            num_pages=0,
            page_reg_desc=b"",
            request_id=request_id,
            ready_kv_len=0,
        )
        assert len(prompt_ids) == 3
        await kv.send(pickle.dumps(PDUpKVStatus(request_id, owner, pickle.dumps(info))))
        await asyncio.wait_for(self._event(request_id).wait(), timeout=5)
        # Decode also reports the transferred first token. P/D WebSocket delivery
        # order is independent; the master deduplicates their index-one outputs.
        await self._token(websocket, "decode", owner, request_id, 1, "hello", FinishStatus.NO_FINISH)
        if self.hold_decode:
            await asyncio.Event().wait()
        # Keep enough requests in flight to exercise load balancing.
        if self.decode_delay:
            await asyncio.sleep(self.decode_delay)
        await self._token(websocket, "decode", owner, request_id, 2, " world", FinishStatus.FINISHED_STOP)


async def _wait_until(predicate, timeout=5):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


async def _wait_http_status(client, url, expected_status, timeout=20):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        response = await client.get(url)
        if response.status_code == expected_status:
            return response
        assert asyncio.get_running_loop().time() < deadline, response.text
        await asyncio.sleep(0.1)


@pytest.mark.parametrize("worker_count,strategy", [(1, "cache_aware"), (2, "round_robin"), (4, "cache_aware")])
def test_real_pd_master_generation_sse_busy_and_cancellation(tmp_path, monkeypatch, worker_count, strategy):
    monkeypatch.setenv("LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS", "0")

    async def run(master):
        public = urlsplit(master.public_url)
        monkeypatch.setattr(
            pd_loop,
            "get_shm_port_args",
            lambda: SimpleNamespace(config_server_port=public.port, pd_master_port=public.port),
        )
        discovery_args = SimpleNamespace(
            config_server_host=None,
            config_server_port=None,
            pd_master_ip=public.hostname,
        )
        discovered = await pd_loop._get_pd_master_objs(discovery_args)
        assert discovered is not None
        assert {(node.node_id, node.host_ip_port) for node in discovered.values()} == {
            (worker["node_id"], worker["host_ip_port"]) for worker in master.workers
        }
        async with FakePDPeers(master.workers) as peers:
            async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
                deadline = asyncio.get_running_loop().time() + 10
                while True:
                    response = await client.get(master.public_url + "/readiness")
                    if response.status_code == 200:
                        break
                    assert asyncio.get_running_loop().time() < deadline, response.text
                    await asyncio.sleep(0.1)

                async def generate():
                    response = await client.post(
                        master.public_url + "/generate",
                        json={
                            "inputs": "hello world",
                            "parameters": {"max_new_tokens": 2, "n": 2, "best_of": 2},
                        },
                    )
                    assert response.status_code == 200, response.text
                    assert response.json()["generated_text"] == ["hello world", "hello world"]

                await asyncio.gather(*(generate() for _ in range(32)))
                assert len(peers.requests) == 64
                assert {owner for owner, _ in peers.requests} == {worker["node_id"] for worker in master.workers}
                ids = [request_id for _, request_id in peers.requests]
                assert len(set(ids)) == len(ids)
                assert all(request_id % 8 == 0 for request_id in ids)

                # Scraping through each worker shares its manager's metrics client
                # and exposes the same service-wide request counter.
                for worker in master.workers:
                    deadline = asyncio.get_running_loop().time() + 5
                    while True:
                        response = await client.get("http://" + worker["host_ip_port"] + "/metrics")
                        assert response.status_code == 200
                        counts = [
                            sample.value
                            for family in text_string_to_metric_families(response.text)
                            for sample in family.samples
                            if sample.name == "lightllm_request_count_total"
                        ]
                        if counts == [32]:
                            break
                        assert asyncio.get_running_loop().time() < deadline, response.text
                        await asyncio.sleep(0.05)

                async with client.stream(
                    "POST",
                    master.public_url + "/generate_stream",
                    json={
                        "inputs": "hello world",
                        "parameters": {"max_new_tokens": 2},
                    },
                ) as response:
                    assert response.status_code == 200
                    events = [json.loads(line[5:]) async for line in response.aiter_lines() if line.startswith("data:")]
                assert [event["token"]["text"] for event in events] == ["hello", " world"]
                assert [event["finished"] for event in events] == [False, True]

                # A rejected Prefill must release its reservation and preserve the
                # public busy envelope, without killing a control connection.
                peers.busy_prefill_requests = 1
                response = await client.post(
                    master.public_url + "/generate",
                    json={
                        "inputs": "hello world",
                        "parameters": {"max_new_tokens": 2},
                    },
                )
                assert response.status_code == 429, response.text
                assert response.json()["error"]["type"] == "RateLimitError"

                # Closing SSE after the first token must reach both P/D actors.
                peers.hold_decode = True
                async with client.stream(
                    "POST",
                    master.public_url + "/generate_stream",
                    json={
                        "inputs": "hello world",
                        "parameters": {"max_new_tokens": 2},
                    },
                ) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            assert json.loads(line[5:])["token"]["text"] == "hello"
                            break
                    owner, request_id = peers.requests[-1]
                await _wait_until(
                    lambda: all((mode, owner, request_id) in peers.aborted for mode in ("prefill", "decode"))
                )
                peers.hold_decode = False

                # Cancellation before prompt IDs must clean up a still-held
                # Prefill reservation, rather than only the output iterator.
                peers.hold_prompt_ids = True
                pending = asyncio.create_task(
                    client.post(
                        master.public_url + "/generate",
                        json={
                            "inputs": "hello world",
                            "parameters": {"max_new_tokens": 2},
                        },
                    )
                )
                await _wait_until(lambda: bool(peers.blocked_prefills))
                owner, request_id = peers.blocked_prefills[-1]
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                await _wait_until(
                    lambda: all((mode, owner, request_id) in peers.aborted for mode in ("prefill", "decode"))
                )
                peers.hold_prompt_ids = False

                await generate()
                assert not peers.generation_errors
                for task in peers.tasks:
                    assert not task.done(), task.exception()

    with running_pd_master(
        tmp_path, extra_args=["--httpserver_workers", str(worker_count), "--select_p_d_node_strategy", strategy]
    ) as master:
        assert len(master.workers) == worker_count
        try:
            asyncio.run(run(master))
        except BaseException:
            print(master.log_path.read_text()[-12000:])
            raise


def test_real_health_aggregation_and_private_channel_isolation(tmp_path):
    """Public health succeeds only while every real worker has usable P/D peers."""

    async def run(master):
        async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
            assert (await client.get(master.public_url + "/liveness")).status_code == 200
            assert (await client.get(master.public_url + "/pd_register")).status_code == 426
            await _wait_http_status(client, master.public_url + "/readiness", 503)
            await _wait_http_status(client, master.public_url + "/health", 503)

            async with FakePDPeers(master.workers):
                readiness = await _wait_http_status(client, master.public_url + "/readiness", 200)
                health = await _wait_http_status(client, master.public_url + "/health", 200)
                assert readiness.json() == {"status": "ok"}
                assert health.json() == {"message": "Ok"}

            readiness = await _wait_http_status(client, master.public_url + "/readiness", 503)
            health = await _wait_http_status(client, master.public_url + "/health", 503)
            assert readiness.json() == {"status": "not ready"}
            assert health.json() == {"message": "Error"}

    with running_pd_master(tmp_path, extra_args=["--httpserver_workers", "2"]) as master:
        try:
            asyncio.run(run(master))
        except BaseException:
            print(master.log_path.read_text()[-12000:])
            raise


def _service_cpu_seconds(master):
    processes = [psutil.Process(master.process.pid), *master.descendants()]
    cpu_seconds = 0.0
    for process in {process.pid: process for process in processes}.values():
        try:
            cpu_time = process.cpu_times()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        cpu_seconds += cpu_time.user + cpu_time.system
    return cpu_seconds


async def _run_high_load(master, request_count, concurrency):
    prompt = "hello world " * 48
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 2},
    }
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(60)

    async with FakePDPeers(master.workers, decode_delay=0) as peers, httpx.AsyncClient(
        trust_env=False,
        timeout=timeout,
        limits=limits,
    ) as client:
        await _wait_http_status(client, master.public_url + "/readiness", 200)
        semaphore = asyncio.Semaphore(concurrency)

        async def generate(latencies=None):
            async with semaphore:
                start = time.perf_counter()
                response = await client.post(master.public_url + "/generate", json=payload)
                if latencies is not None:
                    latencies.append(time.perf_counter() - start)
                assert response.status_code == 200, response.text
                assert response.json()["generated_text"] == ["hello world"]

        # 预热连接池、tokenizer 和请求 ID 客户端，避免一次性初始化污染结果。
        await asyncio.gather(*(generate() for _ in range(concurrency)))
        peers.requests.clear()
        cpu_start = _service_cpu_seconds(master)
        latencies = []
        start = time.perf_counter()
        await asyncio.gather(*(generate(latencies) for _ in range(request_count)))
        elapsed = time.perf_counter() - start
        cpu_seconds = _service_cpu_seconds(master) - cpu_start

        assert len(peers.requests) == request_count
        request_ids = [request_id for _, request_id in peers.requests]
        assert len(set(request_ids)) == request_count
        assert {owner for owner, _ in peers.requests} == {worker["node_id"] for worker in master.workers}

        sorted_latencies = sorted(latencies)
        return {
            "requests": request_count,
            "concurrency": concurrency,
            "elapsed_seconds": elapsed,
            "requests_per_second": request_count / elapsed,
            "p50_ms": statistics.median(sorted_latencies) * 1000,
            "p95_ms": sorted_latencies[int((len(sorted_latencies) - 1) * 0.95)] * 1000,
            "service_cpu_cores": cpu_seconds / elapsed,
        }


@pytest.mark.skipif(
    os.getenv("LIGHTLLM_RUN_PD_MASTER_PERF") != "1",
    reason="set LIGHTLLM_RUN_PD_MASTER_PERF=1 to run the real PD master load benchmark",
)
def test_multiprocess_pd_master_high_load_speedup(tmp_path):
    """Compare the complete HTTP -> PD master -> fake P/D protocol path."""
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if available_cpus < 2:
        pytest.skip("the multiprocess benchmark requires at least two available CPUs")

    request_count = int(os.getenv("LIGHTLLM_PD_MASTER_PERF_REQUESTS", "2000"))
    concurrency = int(os.getenv("LIGHTLLM_PD_MASTER_PERF_CONCURRENCY", "128"))
    multi_worker_count = int(os.getenv("LIGHTLLM_PD_MASTER_PERF_WORKERS", str(min(4, available_cpus))))
    if not 2 <= multi_worker_count <= available_cpus:
        raise ValueError("LIGHTLLM_PD_MASTER_PERF_WORKERS must be between 2 and the available CPU count")
    results = {}

    for worker_count in (1, multi_worker_count):
        runtime_dir = tmp_path / f"workers-{worker_count}"
        with running_pd_master(
            runtime_dir,
            extra_args=[
                "--httpserver_workers",
                str(worker_count),
                "--select_p_d_node_strategy",
                "round_robin",
            ],
        ) as master:
            try:
                results[worker_count] = asyncio.run(_run_high_load(master, request_count, concurrency))
            except BaseException:
                print(master.log_path.read_text()[-12000:])
                raise

    single_qps = results[1]["requests_per_second"]
    multi_qps = results[multi_worker_count]["requests_per_second"]
    speedup = multi_qps / single_qps
    print(
        "PD master high-load benchmark: "
        + json.dumps(
            {
                "single_worker": results[1],
                f"{multi_worker_count}_workers": results[multi_worker_count],
                "throughput_speedup": speedup,
            },
            sort_keys=True,
        )
    )

    minimum_speedup = float(os.getenv("LIGHTLLM_PD_MASTER_MIN_SPEEDUP", "1.0"))
    if minimum_speedup:
        assert speedup >= minimum_speedup
