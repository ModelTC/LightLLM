import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelInput


def _create_model_input(*, is_prefill=False):
    batch_size = 2
    kwargs = dict(
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
    if not is_prefill:
        kwargs["b_shared_seq_len"] = torch.tensor([4, 4], dtype=torch.int32)
        kwargs["b_shared_radix_node_id"] = torch.tensor([10, 10], dtype=torch.int64)
    return ModelInput(**kwargs)


def test_decode_requires_shared_radix_metadata():
    with pytest.raises(AssertionError):
        ModelInput(
            batch_size=1,
            total_token_num=1,
            max_q_seq_len=1,
            max_kv_seq_len=1,
            b_req_idx=torch.zeros(1, dtype=torch.int32),
            b_mtp_index=torch.zeros(1, dtype=torch.int32),
            b_seq_len=torch.ones(1, dtype=torch.int32),
            is_prefill=False,
            multimodal_params=[{"images": [], "audios": []}],
        )


def test_decode_carries_raw_shared_radix_metadata():
    model_input = _create_model_input()

    assert torch.equal(model_input.b_shared_seq_len, torch.tensor([4, 4], dtype=torch.int32))
    assert torch.equal(model_input.b_shared_radix_node_id, torch.tensor([10, 10], dtype=torch.int64))


def test_prefill_does_not_require_shared_radix_metadata():
    model_input = _create_model_input(is_prefill=True)

    assert model_input.b_shared_seq_len is None
    assert model_input.b_shared_radix_node_id is None
