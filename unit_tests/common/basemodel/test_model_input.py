from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.batch_objs as batch_objs_module
from lightllm.common.basemodel.batch_objs import ModelInput


def _create_model_input(*, is_prefill=False, b_mtp_index=None):
    batch_size = 2
    return ModelInput(
        batch_size=batch_size,
        total_token_num=batch_size,
        max_q_seq_len=1,
        max_kv_seq_len=1,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32),
        b_mtp_index=torch.zeros(batch_size, dtype=torch.int32) if b_mtp_index is None else b_mtp_index,
        b_seq_len=torch.ones(batch_size, dtype=torch.int32),
        is_prefill=is_prefill,
        multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
    )


def _mock_group_modes(monkeypatch, *, diverse=False, dynamic=False, triton=False):
    monkeypatch.setattr(
        batch_objs_module,
        "enable_diverse_mode_gqa_decode_fast_kernel",
        lambda: diverse,
    )
    monkeypatch.setattr(batch_objs_module, "enable_triton_mtp_kernel", lambda: triton)
    monkeypatch.setattr(
        batch_objs_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_dynamic_verify=dynamic),
    )


def test_diverse_decode_defaults_to_independent_groups(monkeypatch):
    _mock_group_modes(monkeypatch, diverse=True)
    model_input = _create_model_input()

    model_input._ensure_decode_group_metadata()

    assert torch.equal(model_input.b_mark_shared_group, torch.ones(2, dtype=torch.int32))
    assert torch.equal(model_input.b_shared_seq_len, torch.zeros(2, dtype=torch.int32))


@pytest.mark.parametrize("dynamic,triton", [(True, False), (False, True)])
def test_plain_spec_decode_defaults_to_single_row_groups(monkeypatch, dynamic, triton):
    _mock_group_modes(monkeypatch, dynamic=dynamic, triton=triton)
    model_input = _create_model_input()

    model_input._ensure_decode_group_metadata()

    assert torch.equal(model_input.b_mark_shared_group, torch.ones(2, dtype=torch.int32))
    assert model_input.b_shared_seq_len is None


def test_multi_row_spec_decode_defaults_to_single_row_groups(monkeypatch):
    _mock_group_modes(monkeypatch, dynamic=True)
    model_input = _create_model_input(b_mtp_index=torch.tensor([0, 1], dtype=torch.int32))

    model_input._ensure_decode_group_metadata()

    assert torch.equal(model_input.b_mark_shared_group, torch.ones(2, dtype=torch.int32))


def test_multi_row_spec_decode_preserves_explicit_group_metadata(monkeypatch):
    _mock_group_modes(monkeypatch, dynamic=True)
    model_input = _create_model_input(b_mtp_index=torch.tensor([0, 1], dtype=torch.int32))
    group_markers = torch.tensor([0, 2], dtype=torch.int32)
    model_input.b_mark_shared_group = group_markers

    model_input._ensure_decode_group_metadata()

    assert model_input.b_mark_shared_group is group_markers


def test_prefill_does_not_create_decode_group_metadata(monkeypatch):
    _mock_group_modes(monkeypatch, diverse=True, dynamic=True, triton=True)
    model_input = _create_model_input(is_prefill=True)

    model_input._ensure_decode_group_metadata()

    assert model_input.b_mark_shared_group is None
    assert model_input.b_shared_seq_len is None
