import pytest
import torch

from lightllm.common.basemodel.triton_kernel.post_process.apply_invalid_token import (
    apply_invalid_token_ids,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_apply_invalid_token_ids(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton kernels.")

    batch_size = 4
    vocab_size = 32
    logits = torch.randn((batch_size, vocab_size), device="cuda", dtype=dtype)
    expected = logits.clone()

    invalid_token_ids_per_batch = [
        [1, 3, 5],
        [],
        [0, 2, 31],
        [7],
    ]

    flat_ids = []
    cu_invalid_token_num = [0]
    invalid_token_num_start = 0
    for ids in invalid_token_ids_per_batch:
        flat_ids.extend(ids)
        invalid_token_num_start += len(ids)
        cu_invalid_token_num.append(invalid_token_num_start)

    invalid_token_ids = torch.tensor(flat_ids, device="cuda", dtype=torch.int32)
    cu_invalid_token_num = torch.tensor(cu_invalid_token_num, device="cuda", dtype=torch.int32)

    for batch_idx, ids in enumerate(invalid_token_ids_per_batch):
        if ids:
            expected[batch_idx, ids] = float("-inf")

    apply_invalid_token_ids(
        Logits=logits,
        invalid_token_ids=invalid_token_ids,
        cu_invalid_token_num=cu_invalid_token_num,
    )
    assert torch.equal(logits, expected)


if __name__ == "__main__":
    pytest.main([__file__])
