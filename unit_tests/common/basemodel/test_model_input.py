import torch

import lightllm.common.basemodel.batch_objs as batch_objs_module
from lightllm.common.basemodel.batch_objs import ModelInput


def _create_model_input(*, is_prefill=False):
    batch_size = 2
    return ModelInput(
        batch_size=batch_size,
        total_token_num=batch_size,
        max_q_seq_len=1,
        max_kv_seq_len=1,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32),
        b_mtp_index=torch.zeros(batch_size, dtype=torch.int32),
        b_seq_len=torch.ones(batch_size, dtype=torch.int32),
        is_prefill=is_prefill,
        multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
    )


def _mock_diverse_mode(monkeypatch, *, enabled=False):
    monkeypatch.setattr(
        batch_objs_module,
        "enable_diverse_mode_gqa_decode_fast_kernel",
        lambda: enabled,
    )


def test_diverse_decode_defaults_to_independent_groups(monkeypatch):
    _mock_diverse_mode(monkeypatch, enabled=True)
    model_input = _create_model_input()

    model_input._ensure_decode_group_metadata()

    assert torch.equal(model_input.b_mark_shared_group, torch.ones(2, dtype=torch.int32))
    assert torch.equal(model_input.b_shared_seq_len, torch.zeros(2, dtype=torch.int32))


def test_normal_decode_does_not_create_group_metadata(monkeypatch):
    _mock_diverse_mode(monkeypatch)
    model_input = _create_model_input()

    model_input._ensure_decode_group_metadata()

    assert model_input.b_mark_shared_group is None
    assert model_input.b_shared_seq_len is None


def test_prefill_does_not_create_decode_group_metadata(monkeypatch):
    _mock_diverse_mode(monkeypatch, enabled=True)
    model_input = _create_model_input(is_prefill=True)

    model_input._ensure_decode_group_metadata()

    assert model_input.b_mark_shared_group is None
    assert model_input.b_shared_seq_len is None
