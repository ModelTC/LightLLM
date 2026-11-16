import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.chunk import chunk_gated_delta_rule
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.index import prepare_chunk_indices


@pytest.mark.parametrize("boundaries", [[0], [0, 0, 0], [0, 0, 3, 3, 132, 132]])
def test_empty_requests_keep_chunk_request_ids(boundaries):
    cu = torch.tensor(boundaries, dtype=torch.int32)
    expected = [] if boundaries[-1] == 0 else [[1, 0], [3, 0], [3, 1], [3, 2]]
    torch.testing.assert_close(
        prepare_chunk_indices(cu, 64), torch.tensor(expected, dtype=torch.int32).reshape(-1, 2), atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gdn_packed_empty_requests_match_individual_sequences():
    torch.manual_seed(53)
    lengths = [0, 3, 0, 65, 0]
    shape = (1, sum(lengths), 2, 128)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(shape[:-1], device="cuda")
    beta = torch.rand(shape[:-1], device="cuda")
    initial = torch.randn(len(lengths), 2, 128, 128, device="cuda") * 0.1
    cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    output, final = chunk_gated_delta_rule(
        q,
        k,
        v.clone(),
        g,
        beta,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
    )
    start = 0
    for row, length in enumerate(lengths):
        if length == 0:
            torch.testing.assert_close(final[row], initial[row], atol=0, rtol=0)
            continue
        end = start + length
        expected, expected_state = chunk_gated_delta_rule(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end].clone(),
            g[:, start:end],
            beta[:, start:end],
            initial_state=initial[row : row + 1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, length], device="cuda", dtype=torch.int32),
        )
        torch.testing.assert_close(output[:, start:end], expected, atol=0, rtol=0)
        torch.testing.assert_close(final[row], expected_state[0], atol=0, rtol=0)
        start = end
