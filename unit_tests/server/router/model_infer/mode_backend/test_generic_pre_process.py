from types import SimpleNamespace

import torch
from lightllm.server.router.model_infer.mode_backend import generic_padded_pre_process


def test_padded_decode_leaves_mtp_attention_metadata_unset(monkeypatch):
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
    monkeypatch.setattr(generic_padded_pre_process, "enable_diverse_mode_gqa_decode_fast_kernel", lambda: False)

    req = SimpleNamespace(
        req_idx=7,
        cur_kv_len=4,
        mtp_step=max_draft_step,
        multimodal_params={"images": [], "audios": []},
        get_cur_total_len=lambda: 5,
    )
    model_input, _, padded_req_num = generic_padded_pre_process.padded_prepare_decode_inputs(
        req_objs=[req], dest_batch_size=3
    )

    assert padded_req_num == 2
    assert model_input.b_mtp_index.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert model_input.b_mark_shared_group is None
