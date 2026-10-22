from types import SimpleNamespace

import torch
import pytest

from lightllm.server.router.model_infer.mode_backend.pd.cache_page_io import create_cache_page_io


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
def test_nccl_page_prefix_preserves_byte_count_and_legacy_full_pages(dtype):
    from lightllm.server.router.model_infer.mode_backend.pd.nccl_kv_transporter import _page_prefix

    page = torch.arange(32).to(dtype).view(2, 16)
    assert _page_prefix(page, page.nbytes) is page
    prefix = _page_prefix(page, 8 * page.element_size())
    assert prefix.nbytes == 8 * page.element_size()
    torch.testing.assert_close(prefix, page.reshape(-1)[:8])
    assert prefix.data_ptr() == page.data_ptr()


def test_nixl_partial_transfer_keeps_the_allocation_page_stride():
    from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import NixlKVTransporter

    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter.page_len = 256
    transporter.nixl_agent = SimpleNamespace(
        get_xfer_descs=lambda pages, memory_type: pages,
        prep_xfer_dlist=lambda agent, descs, memory_type: (agent, descs),
    )
    peer, descriptors = transporter._create_paged_xfer_handles([(4096, 512, 0, 0)], 2, 64, agent_name="peer")
    assert peer == "peer"
    assert descriptors == [(4096, 64, 0), (4352, 64, 0)]


def test_existing_page_io_needs_no_position_metadata():
    from lightllm.server.router.model_infer.mode_backend.pd.cache_page_io import KVCachePageIO

    calls = []
    page_io = KVCachePageIO.__new__(KVCachePageIO)
    page_io.buffer = torch.empty((1, 2, 3), dtype=torch.bfloat16)
    page_io.kv_views, page_io.dp_world_size = [], 1
    page_io.transfer = SimpleNamespace(
        write_mem_to_page_kv_move_buffer=lambda indexes, *args: calls.append(("pack", indexes)),
        read_page_kv_move_buffer_to_mem=lambda indexes, *args: calls.append(("unpack", indexes)),
    )
    task = SimpleNamespace(
        mem_indexes=[3, 4], src_page_index=0, dst_page_index=0, prefill_dp_index=0, decode_dp_index=0, page_kind="kv"
    )
    assert page_io.pack(task) == 12
    page_io.unpack(task)
    assert calls == [("pack", [3, 4]), ("unpack", [3, 4])]


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
