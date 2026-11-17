"""Bounded, in-memory routing-hint relay; never participates in token forwarding."""

import asyncio
import json
import uuid
from collections import deque
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)
router = APIRouter()


@dataclass(eq=False)
class MetadataSubscriber:
    owner: str
    master_id: int
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    overflow: asyncio.Event = field(default_factory=asyncio.Event)


class CacheMetadataRelay:
    """One ordered hint log per model/schema namespace, owned by the ASGI loop."""

    def __init__(self, max_log_bytes=16 * 1024 * 1024):
        self.epoch = uuid.uuid4().hex
        self.sequence = 0
        self.history = deque()
        self.history_bytes = 0
        self.event_ids = set()
        self.subscribers = {}
        self.owners = {}
        self.max_log_bytes = max_log_bytes

    def subscribe(self, owner, master_id, epoch, cursor):
        subscriber = MetadataSubscriber(owner, master_id)
        previous = self.subscribers.get(owner)
        if previous is not None:
            previous.overflow.set()
        self.subscribers[owner] = subscriber
        self.owners.setdefault(owner, {"master_id": master_id, "load": None})
        oldest = self.history[0][0] if self.history else self.sequence + 1
        resume = epoch == self.epoch and oldest - 1 <= cursor <= self.sequence
        start = cursor if resume else oldest - 1
        initial = [json.dumps({"kind": "resume" if resume else "reset", "epoch": self.epoch, "seq": start})]
        initial.extend(payload for seq, _, payload in self.history if seq > start)
        # Empty snapshots advertise membership, not an idle worker.
        initial.append(json.dumps({"kind": "loads", "owners": self._loads()}))
        self._broadcast({"kind": "loads", "owners": self._loads()})
        return subscriber, initial

    def unsubscribe(self, subscriber):
        if self.subscribers.get(subscriber.owner) is subscriber:
            self.subscribers.pop(subscriber.owner)

    def forget_master(self, master_id):
        # A metadata disconnect alone does not mean that its requests stopped.
        for owner, state in list(self.owners.items()):
            if state["master_id"] == master_id and owner not in self.subscribers:
                self.owners.pop(owner)
        self._broadcast({"kind": "loads", "owners": self._loads()})

    def _loads(self):
        return {owner: state["load"] for owner, state in self.owners.items()}

    def _broadcast(self, message):
        payload = json.dumps(message, ensure_ascii=False)
        for subscriber in self.subscribers.values():
            try:
                subscriber.queue.put_nowait(payload)
            except asyncio.QueueFull:
                # The endpoint closes this metadata connection; token sockets are independent.
                subscriber.overflow.set()

    def publish(self, subscriber, message):
        if self.subscribers.get(subscriber.owner) is not subscriber:
            return
        if message["kind"] == "hint":
            event_id = (subscriber.owner, message["id"])
            if event_id in self.event_ids:
                return
            self.sequence += 1
            event = {
                "kind": "hint",
                "seq": self.sequence,
                "owner": subscriber.owner,
                "id": message["id"],
                "key": message["key"],
                "worker": message["worker"],
                "node_id": message["node_id"],
                "observed_at": message["observed_at"],
            }
            payload = json.dumps(event, ensure_ascii=False)
            self.history.append((self.sequence, event_id, payload))
            self.event_ids.add(event_id)
            self.history_bytes += len(payload.encode("utf-8"))
            while self.history_bytes > self.max_log_bytes:
                _, old_id, old_payload = self.history.popleft()
                self.event_ids.remove(old_id)
                self.history_bytes -= len(old_payload.encode("utf-8"))
            self._broadcast(event)
        elif message["kind"] == "load":
            load = {"observed_at": message["observed_at"], "workers": message["workers"]}
            self.owners[subscriber.owner]["load"] = load
            self._broadcast({"kind": "load", "owner": subscriber.owner, **load})
        else:
            raise ValueError(f"Unknown cache metadata message: {message['kind']}")


relays = {}


@router.websocket("/pd_cache_metadata")
async def cache_metadata_endpoint(websocket: WebSocket):
    await websocket.accept()
    subscriber = None
    tasks = []
    try:
        hello = await websocket.receive_json()
        assert hello["version"] == 1
        namespace = hello["namespace"]
        if namespace not in relays:
            relays[namespace] = CacheMetadataRelay()
        relay = relays[namespace]
        subscriber, initial = relay.subscribe(hello["owner"], hello["master_id"], hello["epoch"], hello["cursor"])

        async def send_loop():
            for index, payload in enumerate(initial):
                await websocket.send_text(payload)
                if index % 16 == 15:
                    await asyncio.sleep(0)
            while True:
                await websocket.send_text(await subscriber.queue.get())

        async def receive_loop():
            while True:
                payload = await websocket.receive_text()
                if len(payload.encode("utf-8")) > 128 * 1024:
                    raise ValueError("Cache metadata message exceeds 128 KiB")
                relay.publish(subscriber, json.loads(payload))

        tasks = [
            asyncio.create_task(send_loop()),
            asyncio.create_task(receive_loop()),
            asyncio.create_task(subscriber.overflow.wait()),
        ]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("PD cache metadata connection failed")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if subscriber is not None:
            relay.unsubscribe(subscriber)
            if subscriber.master_id not in websocket.app.state.registered_pd_master_objs:
                relay.forget_master(subscriber.master_id)
        try:
            await websocket.close(code=1013)
        except (RuntimeError, WebSocketDisconnect):
            pass
