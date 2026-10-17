import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
    build_dynamic_mtp_linear_att_state_params,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("request_capacity,token_bucket", [(4, 3), (4, 16), (128, 512)])
def test_dynamic_sequences_stay_within_request_capacity_under_graph(request_capacity, token_bucket):
    hold = request_capacity
    reqs = torch.full((token_bucket,), hold, dtype=torch.int32, device="cuda")
    mtp_index = torch.zeros_like(reqs)
    accepted = torch.arange(request_capacity + 1, dtype=torch.int32, device="cuda") % 4
    sequences = min(request_capacity, token_bucket)

    def build():
        return build_dynamic_mtp_linear_att_state_params(reqs, mtp_index, accepted, hold)

    build()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cu, ids, counts = build()
    assert cu.numel() == sequences + 1
    assert ids.numel() == counts.numel() == sequences

    # Reuse one graph for full request pools, mixed verify lengths, and HOLD-only padding.
    for lengths in ([1] * sequences, [min(4, token_bucket // sequences)] * sequences, [2, 1], []):
        lengths = lengths[:request_capacity]
        if sum(lengths) > token_bucket:
            continue
        request_ids = list(reversed(range(len(lengths))))
        host_reqs = [req for req, length in zip(request_ids, lengths) for _ in range(length)]
        host_mtp = [j for length in lengths for j in range(length)]
        tokens = len(host_reqs)
        host_reqs += [hold] * (token_bucket - tokens)
        host_mtp += [0] * (token_bucket - tokens)
        reqs.copy_(torch.tensor(host_reqs, dtype=torch.int32, device="cuda"))
        mtp_index.copy_(torch.tensor(host_mtp, dtype=torch.int32, device="cuda"))
        graph.replay()

        expected_cu = [0]
        for length in lengths:
            expected_cu.append(expected_cu[-1] + length)
        expected_cu += [tokens] * (sequences + 1 - len(expected_cu))
        expected_ids = request_ids + [hold] * (sequences - len(request_ids))
        expected_counts = [req % 4 + 1 for req in request_ids] + [1] * (sequences - len(request_ids))
        assert cu.tolist() == expected_cu
        assert ids.tolist() == expected_ids
        assert counts.tolist() == expected_counts
