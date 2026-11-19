"""Read-only prefix residency probes for split-rank hybrid Prefill workers."""

import asyncio
import os
import pickle
import threading
import uuid

import zmq
import zmq.asyncio
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from lightllm.server.pd_io_struct import ObjType
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
MAX_HASHES = 4096
TIERS = {b"gpu", b"cpu", b"disk", b"miss", b"unknown"}


def probe_enabled(args):
    return (
        os.getenv("LIGHTLLM_PD_TIER_AWARE_RANK") == "1"
        and args.run_mode == "prefill"
        and args.use_dp_split_mode_connect_pd_master
        and args.enable_cpu_cache
        and not args.disable_dynamic_prompt_cache
    )


def _address(name):
    # Linux abstract sockets disappear with the owner, including on a failed start.
    return f"ipc://@lightllm-{get_unique_server_name()}-cache-tier-{name}"


def start_probe_reader(name, lookup):
    """The cache owner serves bounded reads on its CPU thread, outside the inference loop."""
    def serve():
        with zmq.Context.instance().socket(zmq.REP) as sock:
            sock.setsockopt(zmq.MAXMSGSIZE, 16 * MAX_HASHES)
            sock.bind(_address(name))
            while True:
                raw = sock.recv()
                try:
                    if not raw or len(raw) % 16 or len(raw) > 16 * MAX_HASHES:
                        answer = b"unknown"
                    else:
                        hashes = [int.from_bytes(raw[i : i + 16], "big") for i in range(0, len(raw), 16)]
                        answer = lookup(hashes)
                except Exception:
                    logger.exception("Failed to read Prefill cache residency")
                    answer = b"unknown"
                sock.send(answer)

    threading.Thread(target=serve, name=f"cache-tier-{name}", daemon=True).start()


async def _read_tier(name, raw):
    with zmq.asyncio.Context.instance().socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(_address(name))
        await sock.send(raw)
        return await sock.recv()


async def reply_to_probe(websocket, probe_id, rank, raw, big_page_num):
    async def lookup():
        # Any reachable big checkpoint implies a hit on the first big page.
        gpu = await _read_tier(rank, raw[: 16 * big_page_num])
        if gpu != b"miss":
            return gpu
        return await _read_tier("host", raw)

    tier = b"unknown"
    if rank is not None and isinstance(raw, bytes) and 0 < len(raw) <= 16 * MAX_HASHES and len(raw) % 16 == 0:
        try:
            tier = await asyncio.wait_for(lookup(), timeout=0.08)
        except (asyncio.TimeoutError, zmq.ZMQError):
            pass
    try:
        await websocket.send(pickle.dumps((ObjType.CACHE_TIER_RESULT, probe_id, tier)))
    except ConnectionClosed:
        return


async def query_prefill_tier(worker, raw):
    probe_id = uuid.uuid4().hex
    reply = asyncio.get_running_loop().create_future()
    worker.cache_tier_queries[probe_id] = reply

    async def query():
        await worker.websocket.send_bytes(pickle.dumps((ObjType.CACHE_TIER_QUERY, probe_id, raw)))
        return await reply

    try:
        return await asyncio.wait_for(query(), timeout=0.15)
    except (asyncio.TimeoutError, OSError, RuntimeError, WebSocketDisconnect):
        return b"unknown"
    finally:
        worker.cache_tier_queries.pop(probe_id, None)


def finish_probe(worker, probe_id, tier):
    reply = worker.cache_tier_queries.get(probe_id)
    if reply is not None and not reply.done():
        reply.set_result(tier if tier in TIERS else b"unknown")
