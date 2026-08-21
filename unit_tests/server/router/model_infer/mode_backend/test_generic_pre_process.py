from types import SimpleNamespace

import torch
from lightllm.server.router.model_infer.mode_backend import generic_padded_pre_process, generic_pre_process


def _patch_empty_input_context(monkeypatch):
    mem_manager = SimpleNamespace(
        HOLD_TOKEN_MEMINDEX=-1,
        alloc=lambda size: torch.empty((size,), dtype=torch.int32),
    )
    infer_context = SimpleNamespace(
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=-1, mem_manager=mem_manager),
        radix_cache=None,
    )
    monkeypatch.setattr(generic_pre_process, "g_infer_context", infer_context)
    return infer_context


def test_prepare_prefill_inputs_allows_empty_batch(monkeypatch):
    _patch_empty_input_context(monkeypatch)

    model_input, run_reqs = generic_pre_process.prepare_prefill_inputs([], is_chuncked_mode=True)

    assert run_reqs == []
    assert model_input.batch_size == 0
    assert model_input.input_ids.shape == (0,)
    assert model_input.mem_indexes_cpu.shape == (0,)
    assert model_input.b_req_idx.shape == (0,)
    assert model_input.b_prefill_start_loc.shape == (0,)
    assert model_input.b_prefill_has_output_cpu == []
    assert model_input.max_q_seq_len == 0
    assert model_input.max_kv_seq_len == 0


def test_prepare_decode_inputs_allows_empty_batch(monkeypatch):
    _patch_empty_input_context(monkeypatch)

    model_input, run_reqs = generic_pre_process.prepare_decode_inputs([])

    assert run_reqs == []
    assert model_input.batch_size == 0
    assert model_input.input_ids is None
    assert model_input.mem_indexes_cpu.shape == (0,)
    assert model_input.b_req_idx.shape == (0,)
    assert model_input.b_position_delta.shape == (0,)
    assert model_input.b_shared_seq_len.shape == (0,)
    assert model_input.b_shared_radix_node_id.shape == (0,)
    assert model_input.max_q_seq_len == 1
    assert model_input.max_kv_seq_len == 0


def test_padded_decode_builds_raw_shared_radix_metadata(monkeypatch):
    max_draft_step = 2
    mem_manager = SimpleNamespace(
        HOLD_TOKEN_MEMINDEX=-1,
        alloc=lambda size: torch.arange(size, dtype=torch.int32),
    )
    req_manager = SimpleNamespace(HOLD_REQUEST_ID=-1, mem_manager=mem_manager)
    monkeypatch.setattr(
        generic_padded_pre_process,
        "g_infer_context",
        SimpleNamespace(req_manager=req_manager, radix_cache=None),
    )
    monkeypatch.setattr(
        generic_padded_pre_process,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_step=max_draft_step, mtp_dynamic_verify=True),
    )

    shared_kv_node = SimpleNamespace(time_id=torch.iinfo(torch.int64).max + 42)
    req = SimpleNamespace(
        req_idx=7,
        cur_kv_len=4,
        mtp_step=max_draft_step,
        multimodal_params={"images": [], "audios": []},
        shared_kv_node=shared_kv_node,
        get_cur_total_len=lambda: 5,
        get_radix_cache_shared_len=lambda: 4,
    )
    model_input, _, padded_req_num = generic_padded_pre_process.padded_prepare_decode_inputs(
        req_objs=[req], dest_batch_size=3
    )

    assert padded_req_num == 2
    assert model_input.b_mtp_index.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert model_input.b_shared_seq_len.tolist() == [4, 4, 4, 0, 0, 0, 0, 0, 0]
    assert model_input.b_shared_radix_node_id.tolist() == [42, 42, 42, -1, -1, -1, -1, -1, -1]
