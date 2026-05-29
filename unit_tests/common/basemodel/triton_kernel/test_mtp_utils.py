import json
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel import mtp_utils


class _FakeMemManager:
    def __init__(self):
        self.freed = []

    def free(self, mem_indexes):
        self.freed.append(mem_indexes.clone())


def test_trim_dynamic_mtp_model_input(monkeypatch):
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            {
                "diverse_mode": False,
                "llm_kv_type": "fp16",
                "mtp_dynamic_verify": True,
                "mtp_step": 3,
                "llm_decode_att_backend": "triton",
            }
        ),
    )
    monkeypatch.setenv("LIGHTLLM_MAX_BATCH_SHARED_GROUP_SIZE", "4")
    mtp_utils.get_env_start_args.cache_clear()
    mtp_utils.get_diverse_max_batch_shared_group_size.cache_clear()

    fake_mem_manager = _FakeMemManager()
    monkeypatch.setattr(
        mtp_utils,
        "g_infer_context",
        SimpleNamespace(req_manager=SimpleNamespace(mem_manager=fake_mem_manager)),
    )

    model_input = ModelInput(
        batch_size=12,
        total_token_num=54,
        max_q_seq_len=1,
        max_kv_seq_len=6,
        input_ids=torch.arange(12, dtype=torch.int64, device="cuda") + 1000,
        b_req_idx=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32, device="cuda"),
        b_mtp_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6], dtype=torch.int32, device="cuda"),
        b_shared_seq_len=torch.tensor([0, 0, 0, 0, 7, 7, 7, 7, 9, 9, 9, 9], dtype=torch.int32, device="cuda"),
        b_mark_shared_group=torch.tensor([0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4], dtype=torch.int32, device="cuda"),
        mem_indexes=torch.arange(12, dtype=torch.int32, device="cuda") + 100,
        mem_indexes_cpu=torch.arange(12, dtype=torch.int32, device="cpu") + 100,
        is_prefill=False,
        multimodal_params=[{"row": i, "images": [], "audios": []} for i in range(12)],
        mtp_draft_input_hiddens=(torch.arange(12 * 5, dtype=torch.float32, device="cuda").reshape(12, 5) + 0.5),
    )
    req_to_next_token_probs = torch.tensor(
        [
            [1.0, 0.95, 0.90, 0.10, 0.0, 0.0],
            [1.0, 0.20, 0.80, 0.80, 0.0, 0.0],
            [1.0, 0.99, 0.99, 0.99, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    trimmed_input, selected_mask = mtp_utils.prepare_dynamic_mtp_model_input(
        model_input=model_input,
        req_num=3,
        dynamic_batch_size=8,
        req_to_next_token_ids=torch.empty((0,), dtype=torch.int64, device="cuda"),
        req_to_next_token_probs=req_to_next_token_probs,
    )
    torch.cuda.synchronize()

    expected_selected_mask = torch.tensor([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    expected_selected_rows = torch.where(expected_selected_mask == 1)[0]

    assert torch.equal(selected_mask.cpu(), expected_selected_mask)
    assert trimmed_input.batch_size == 8
    assert trimmed_input.max_q_seq_len == 1
    assert trimmed_input.multimodal_params == [
        {"row": 0, "images": [], "audios": []},
        {"row": 1, "images": [], "audios": []},
        {"row": 2, "images": [], "audios": []},
        {"row": 3, "images": [], "audios": []},
        {"row": 4, "images": [], "audios": []},
        {"row": 5, "images": [], "audios": []},
        {"row": 6, "images": [], "audios": []},
        {"row": 7, "images": [], "audios": []},
    ]

    assert torch.equal(
        trimmed_input.input_ids.cpu(), torch.arange(12, dtype=torch.int64)[expected_selected_rows] + 1000
    )
    assert torch.equal(trimmed_input.b_req_idx.cpu(), torch.tensor([0, 0, 0, 1, 2, 2, 2, 2], dtype=torch.int32))
    assert torch.equal(trimmed_input.b_mtp_index.cpu(), torch.tensor([0, 1, 2, 0, 0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(trimmed_input.b_seq_len.cpu(), torch.tensor([3, 4, 5, 3, 3, 4, 5, 6], dtype=torch.int32))
    assert torch.equal(trimmed_input.b_shared_seq_len.cpu(), torch.tensor([0, 0, 0, 7, 9, 9, 9, 9], dtype=torch.int32))
    assert torch.equal(
        trimmed_input.b_mark_shared_group.cpu(), torch.tensor([0, 0, 3, 1, 0, 0, 0, 4], dtype=torch.int32)
    )
    assert torch.equal(
        trimmed_input.mem_indexes.cpu(), torch.tensor([100, 101, 102, 103, 104, 105, 106, 107], dtype=torch.int32)
    )
    assert torch.equal(
        trimmed_input.mem_indexes_cpu, torch.tensor([100, 101, 102, 103, 104, 105, 106, 107], dtype=torch.int32)
    )
    assert len(fake_mem_manager.freed) == 1
    assert torch.equal(fake_mem_manager.freed[0], torch.tensor([108, 109, 110, 111], dtype=torch.int32))

    expected_hiddens = (torch.arange(12 * 5, dtype=torch.float32).reshape(12, 5) + 0.5)[expected_selected_rows]
    assert torch.equal(trimmed_input.mtp_draft_input_hiddens.cpu(), expected_hiddens)


def test_trim_rebuilds_b_mark_shared_group_by_max_batch_shared_group_size(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_MAX_BATCH_SHARED_GROUP_SIZE", "3")
    mtp_utils.get_diverse_max_batch_shared_group_size.cache_clear()

    model_input = ModelInput(
        batch_size=5,
        total_token_num=36,
        max_q_seq_len=1,
        max_kv_seq_len=6,
        input_ids=None,
        b_req_idx=torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"),
        b_mtp_index=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([3, 4, 5, 6, 7], dtype=torch.int32, device="cuda"),
        b_shared_seq_len=None,
        b_mark_shared_group=torch.tensor([0, 0, 0, 0, 5], dtype=torch.int32, device="cuda"),
        mem_indexes=torch.arange(5, dtype=torch.int32, device="cuda"),
        mem_indexes_cpu=torch.arange(5, dtype=torch.int32, device="cpu"),
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []} for _ in range(5)],
    )
    selected_mask = torch.ones((5,), dtype=torch.int32, device="cuda")

    trimmed_input = mtp_utils._trim_decode_model_input_inplace(
        model_input=model_input,
        selected_mask_gpu=selected_mask,
        dynamic_batch_size=5,
    )
    torch.cuda.synchronize()

    assert torch.equal(trimmed_input.b_req_idx.cpu(), torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32))
    assert torch.equal(trimmed_input.b_mtp_index.cpu(), torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32))
    assert torch.equal(trimmed_input.b_mark_shared_group.cpu(), torch.tensor([0, 0, 3, 0, 2], dtype=torch.int32))
