"""Asynchronous metadata replication for locally selected cache-aware routes."""

import asyncio
import hashlib
import json
import time
import uuid

import websockets

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class SharedCacheState:
    # Snapshots are estimates of pending Prefill, not admission reservations.
    load_interval = 0.25
    load_max_age = 3.0

    def __init__(self, selector, args):
        self.selector = selector
        self.owner = uuid.uuid4().hex  # HTTP-worker incarnation, not just the master address.
        self.master_id = args.pd_node_id
        self.namespace = hashlib.sha256(
            json.dumps(
                [args.model_dir, args.model_name, "sampled_chars_v1", selector.policy.config.sample_stride]
            ).encode()
        ).hexdigest()
        self.enabled_for_single_worker = not args.disable_pd_cache_high_priority
        self.uri = f"ws://{args.config_server_host}:{args.config_server_port}/pd_cache_metadata"
        self.epoch = None
        self.cursor = 0
        self.ready = False
        self.synchronized = asyncio.Event()
        self.remote_loads = {}
        self.outgoing = asyncio.Queue(maxsize=256)
        self.local_sequence = 0
        self.dropped_hints = 0

    def enabled(self):
        return len(self.selector.prefill_nodes) > 1 or self.enabled_for_single_worker

    def publish_hint(self, prompt, worker):
        if not self.enabled() or not self.ready:
            return
        self.local_sequence += 1
        message = {
            "kind": "hint",
            "id": self.local_sequence,
            "key": prompt[:: self.selector.policy.config.sample_stride],
            "worker": worker.connection_key,
            "node_id": worker.node_id,
            "observed_at": time.time(),
        }
        # A lost hint only reduces affinity. Never wait for a slow relay in the request path.
        if len(message["key"].encode("utf-8")) > 64 * 1024 or self.outgoing.full():
            self.dropped_hints += 1
            return
        self.outgoing.put_nowait(message)

    def apply(self, message):
        kind = message["kind"]
        if kind in ("reset", "resume"):
            if kind == "reset":
                self.selector.policy.prompt_cache_tree.clear()
                # Unsent hints predate the reset and could overwrite newer retained history.
                while not self.outgoing.empty():
                    self.outgoing.get_nowait()
            self.epoch = message["epoch"]
            self.cursor = message["seq"]
            self.remote_loads.clear()
            self.ready = False
            self.synchronized.clear()
        elif kind == "hint":
            if message["seq"] <= self.cursor:
                return
            assert message["seq"] == self.cursor + 1, "Cache metadata replay gap"
            age = time.time() - message["observed_at"]
            # Cross-host monotonic clocks cannot be compared. A future wall-clock
            # observation is retained for affinity, but is ineligible for freshness priority.
            inserted_at = time.monotonic() - age if age >= 0 else None
            self.selector.policy.prompt_cache_tree.insert_key(
                message["key"], message["worker"], inserted_at, message["node_id"]
            )
            self.cursor = message["seq"]
        elif kind == "loads":
            self.remote_loads = message["owners"]
            self.ready = True
            self.synchronized.set()
        elif kind == "load":
            self.remote_loads[message["owner"]] = {"observed_at": message["observed_at"], "workers": message["workers"]}
        else:
            raise ValueError(f"Unknown cache metadata message: {kind}")

    def cluster_loads(self, workers):
        """None means unknown, never idle. Own counters are always read directly."""
        if not self.ready:
            return None
        loads = {
            worker.connection_key: [worker.dispatched_req_num, worker.dispatched_prompt_chars] for worker in workers
        }
        now = time.time()
        for owner, snapshot in self.remote_loads.items():
            if owner == self.owner:
                continue
            if snapshot is None or not 0 <= now - snapshot["observed_at"] <= self.load_max_age:
                return None
            for worker in workers:
                remote = snapshot["workers"].get(worker.connection_key)
                if remote is None or remote[0] != worker.node_id:
                    return None
                loads[worker.connection_key][0] += remote[1]
                loads[worker.connection_key][1] += remote[2]
        return loads

    async def run(self):
        while True:
            if not self.enabled():
                await asyncio.sleep(1)
                continue
            tasks = []
            try:
                async with websockets.connect(self.uri, max_size=128 * 1024, max_queue=256) as websocket:
                    await websocket.send(
                        json.dumps(
                            {
                                "version": 1,
                                "namespace": self.namespace,
                                "owner": self.owner,
                                "master_id": self.master_id,
                                "epoch": self.epoch,
                                "cursor": self.cursor,
                            }
                        )
                    )

                    async def receive_loop():
                        last_yield = time.monotonic()
                        async for payload in websocket:
                            self.apply(json.loads(payload))
                            # Catch-up must yield to token forwarding on this same event loop.
                            if time.monotonic() - last_yield >= 0.002:
                                await asyncio.sleep(0)
                                last_yield = time.monotonic()

                    async def send_loop():
                        await self.synchronized.wait()
                        next_load = 0.0
                        while self.enabled():
                            now = time.monotonic()
                            if now >= next_load:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "kind": "load",
                                            "observed_at": time.time(),
                                            "workers": {
                                                worker.connection_key: [
                                                    worker.node_id,
                                                    worker.dispatched_req_num,
                                                    worker.dispatched_prompt_chars,
                                                ]
                                                for worker in self.selector.prefill_nodes
                                            },
                                        }
                                    )
                                )
                                next_load = now + self.load_interval
                            try:
                                hint = await asyncio.wait_for(self.outgoing.get(), max(0, next_load - time.monotonic()))
                            except asyncio.TimeoutError:
                                continue
                            payload = json.dumps(hint, ensure_ascii=False)
                            # JSON escaping can expand an otherwise small sampled key.
                            if len(payload.encode("utf-8")) > 64 * 1024:
                                self.dropped_hints += 1
                                continue
                            await websocket.send(payload)

                    tasks = [asyncio.create_task(receive_loop()), asyncio.create_task(send_loop())]
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
            except Exception:
                logger.exception("PD cache metadata disconnected; using worker-reported load")
            finally:
                self.ready = False
                self.synchronized.clear()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(1)
