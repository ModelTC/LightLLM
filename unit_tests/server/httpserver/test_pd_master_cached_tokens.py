import asyncio
import copy
import pickle
from types import SimpleNamespace

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver_for_pd_master import manager as pd_master_manager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster


def _make_manager(monkeypatch):
    monkeypatch.setattr(
        "lightllm.server.httpserver.manager.HttpServerManager._check_and_repair_length",
        classmethod(lambda cls, *a, **k: asyncio.sleep(0)),
    )
    monkeypatch.setattr(SamplingParams, "from_buffer_copy", classmethod(lambda cls, other: copy.copy(other)))
    mgr = object.__new__(HttpServerManagerForPDMaster)
    mgr.log_pd_routes = False
    counter = [0]

    def gen_id():
        counter[0] += 1
        return counter[0]

    mgr.id_gen = SimpleNamespace(generate_id=gen_id)
    mgr.metric_client = SimpleNamespace(counter_inc=lambda *a, **k: None, histogram_observe=lambda *a, **k: None)
    mgr.tokens = lambda *a, **k: 10
    mgr._log_req_header = lambda *a, **k: asyncio.sleep(0)
    mgr.select_p_d_node = lambda *a, **k: asyncio.sleep(0, result=(1, 1))
    mgr.remove_req = lambda *a, **k: asyncio.sleep(0)
    return mgr


def _collect(mgr, sampling_params, monkeypatch, split):
    mgr._split_max_new_tokens = lambda *a, **k: list(split)

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        sub_req_id = sp.group_request_id
        # 首个 token 由 prefill 节点产出，携带真实缓存命中；后续 decode token 恒为 0。
        hit = sp.max_new_tokens * 10
        yield sub_req_id, "x", {"prompt_tokens": 10, "prompt_cache_len": hit}, FinishStatus()
        for _ in range(2):
            yield sub_req_id, "y", {"prompt_tokens": 10, "prompt_cache_len": 0}, FinishStatus()
        yield sub_req_id, "z", {"prompt_tokens": 10, "prompt_cache_len": 0}, FinishStatus(FinishStatus.FINISHED_STOP)

    monkeypatch.setattr(mgr, "_wait_to_token_package", fake_wait)

    async def run():
        out = []
        async for sub_id, out_str, metadata, finish in mgr.generate(
            prompt="hello",
            sampling_params=sampling_params,
            multimodal_params=SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
            request=None,
        ):
            out.append(metadata.get("prompt_cache_len", -1))
        return out

    return asyncio.run(run())


def test_single_block_prefill_hit_persists_past_decode_zeros(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sp = SamplingParams()
    sp.max_new_tokens = 3
    sp.best_of = 1
    sp.group_request_id = 0
    cached = _collect(mgr, sp, monkeypatch, split=[3])
    # prefill reported 30; decode chunks reported 0 — all must read 30, not collapse to 0.
    assert cached and all(c == 30 for c in cached), cached


def test_multi_block_keeps_first_block_hit(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sp = SamplingParams()
    sp.max_new_tokens = 5
    sp.best_of = 1
    sp.group_request_id = 0
    # 两段：block0 命中 30，block1 命中 50。最终 usage 应保留首块 30，不被 50 覆盖。
    cached = _collect(mgr, sp, monkeypatch, split=[3, 2])
    assert cached[-1] == 30, cached


def test_decode_first_token_uses_later_prefill_cache_metadata(monkeypatch):
    class FakeReqStatus:
        def __init__(self, req_id, p_node, d_node):
            self.prefill_prompt_ids_event = asyncio.Event()
            self.prefill_prompt_ids_event.prompt_ids = [1, 2, 3]
            self.prefill_prompt_ids_event.set()
            self.up_status_event = asyncio.Event()
            self.up_status_event.upkv_status = SimpleNamespace(pd_kv_trans_params=pickle.dumps(SimpleNamespace()))
            self.up_status_event.set()
            self.token_batches = [
                [
                    (
                        req_id,
                        "decode-first",
                        {"count_output_tokens": 1, "node_mode": "decode", "prompt_cache_len": 0},
                        FinishStatus(),
                    )
                ],
                [(req_id, "decode-second", {"count_output_tokens": 2, "node_mode": "decode"}, FinishStatus())],
                [
                    (
                        req_id,
                        "prefill-first",
                        {"count_output_tokens": 1, "node_mode": "prefill", "prompt_cache_len": 7},
                        FinishStatus(FinishStatus.FINISHED_LENGTH),
                    )
                ],
            ]

        async def wait_to_ready(self):
            await asyncio.sleep(0)

        async def can_read(self, req_id_to_out_inf):
            return bool(self.token_batches)

        async def pop_all_tokens(self):
            return self.token_batches.pop(0)

    class FakeWebSocket:
        async def send_bytes(self, data):
            pass

    monkeypatch.setattr(pd_master_manager, "ReqStatus", FakeReqStatus)
    mgr = object.__new__(HttpServerManagerForPDMaster)
    mgr.args = SimpleNamespace(pd_node_id=1)
    mgr.req_id_to_out_inf = {}
    sampling_params = SamplingParams()
    sampling_params.group_request_id = 11
    sampling_params.max_new_tokens = 2
    node = SimpleNamespace(websocket=FakeWebSocket())
    request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))

    async def run():
        stream = mgr.fetch_pd_stream(node, node, "hello", sampling_params, SimpleNamespace(), request)
        first = await anext(stream)
        second = await anext(stream)
        await stream.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert first[1] == "decode-first"
    assert first[2]["prompt_cache_len"] == 7
    assert second[1] == "decode-second"
