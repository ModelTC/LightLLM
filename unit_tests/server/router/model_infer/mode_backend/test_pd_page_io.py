from types import SimpleNamespace

import torch
import pytest


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_hybrid_pd_uses_the_shared_att_state_task_without_model_flags(monkeypatch, role):
    from lightllm.server.router.model_infer.infer_batch import g_infer_context
    from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl import (
        PDChunkedPrefillForPrefillNode,
    )
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_impl import PDDecodeNode

    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    cls = PDChunkedPrefillForPrefillNode if role == "prefill" else PDDecodeNode
    backend = cls.__new__(cls)
    backend.args = SimpleNamespace(pd_kv_page_size=256, page_size=256)
    backend.is_master_in_dp = role == "decode"
    groups, tasks = [], []
    backend.info_queue = SimpleNamespace(put=groups.append)
    req = SimpleNamespace(
        shm_req=SimpleNamespace(input_len=512),
        cur_kv_len=512 if role == "prefill" else 0,
        hold_kv_len=0,
        pd_trans_kv_start_index=0,
        _kv_cache_alloc_need=lambda end: end,
    )

    def create(**kwargs):
        task = SimpleNamespace(
            page_kind=kwargs.get("page_kind", "kv"), first_gen_token_id=None, first_gen_token_logprob=None
        )
        tasks.append(task)
        if "group" in kwargs:
            kwargs["group"].task_list.append(task)
        return task

    backend._create_pd_trans_task = create
    if role == "prefill":
        backend._prefill_chuncked_handle_func(req, next_token_id=17, next_token_prob=-0.2, output_len=1)
        assert tasks[-1].first_gen_token_id == 17
    else:
        backend._alloc_req_kv_mem = lambda req, count: torch.arange(count)
        backend._decode_node_gen_trans_tasks(req)
        assert len(groups) == 1
    assert [task.page_kind for task in tasks] == ["kv", "kv", "att_state"]


@pytest.mark.parametrize("seq_len", [1, 2, 3, 129, 256, 257])
def test_pd_request_state_rebuilds_engram_locally(monkeypatch, seq_len):
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DeepseekV4AttStatePageHelper
    from lightllm.models.deepseek_v41.engram import EngramHasher

    slots = seq_len + 5
    source_ids = torch.arange(seq_len, dtype=torch.int32).flip(0)
    decode_ids = torch.arange(seq_len, dtype=torch.int32) + slots

    def manager():
        return SimpleNamespace(
            req_to_swa=torch.zeros((1, 2, 4, 585), dtype=torch.uint8),
            req_to_c4_state=None,
            req_to_c4_indexer_state=None,
            req_to_c128_state=None,
            req_to_token_indexs=torch.stack((source_ids, decode_ids)),
        )

    source, destination = manager(), manager()
    source.req_to_swa[:, 0].random_(256)
    page = torch.empty(source.req_to_swa[:, 0].numel(), dtype=torch.uint8)
    helper = DeepseekV4AttStatePageHelper(source)
    helper._copy_req_state(page, source, 0, write=True)
    helper._copy_req_state(page, destination, 1, write=False)
    torch.testing.assert_close(destination.req_to_swa[:, 1], source.req_to_swa[:, 0], atol=0, rtol=0)

    hasher = EngramHasher.__new__(EngramHasher)
    hasher.has_vision = False
    hasher.token_map = torch.arange(64).flip(0)
    hasher.history = torch.full((2 * slots,), -99, dtype=torch.int64)
    tokens = (torch.arange(seq_len) % 64).tolist()
    tail = tokens[-3:]
    hasher.restore_history([(1, seq_len, tail)], destination)
    expected = torch.full_like(hasher.history, -99)
    expected[decode_ids[-len(tail) :].long()] = hasher.token_map[tail]
    torch.testing.assert_close(hasher.history, expected, atol=0, rtol=0)
