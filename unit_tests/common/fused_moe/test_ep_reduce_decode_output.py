import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import ep_reduce_decode_output


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_ep_reduce_decode_output_weights_local_rows_and_compacts_metadata():
    hidden = 8
    expert_output = torch.tensor(
        [[1.0] * hidden, [2.0] * hidden, [4.0] * hidden],
        device="cuda",
        dtype=torch.bfloat16,
    )
    route_weights = torch.tensor([0.5, 0.25, 0.0], device="cuda", dtype=torch.float32)
    recv_src_metadata = torch.tensor(
        [[7, 9, 0, 1], [3, 4, -1, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    num_valid_recv_tokens = torch.tensor([1], device="cuda", dtype=torch.int32)

    dense_output, compact_metadata = ep_reduce_decode_output(
        expert_output=expert_output,
        route_weights=route_weights,
        recv_src_metadata=recv_src_metadata,
        num_valid_recv_tokens=num_valid_recv_tokens,
    )

    expected = expert_output[0].float() * 0.5 + expert_output[1].float() * 0.25
    torch.testing.assert_close(dense_output[0].float(), expected)
    assert compact_metadata[0, :2].tolist() == [7, 9]
    assert compact_metadata[0, 2].item() == 0
    assert compact_metadata[0, 3].item() == -1
