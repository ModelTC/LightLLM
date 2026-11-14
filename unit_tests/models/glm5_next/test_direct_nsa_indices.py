from unit_tests.common.attention.helpers import plan, build
from lightllm.common.attention.session import AttentionSession
import pytest
import torch

from lightllm.common.attention.api import AttentionBatch, AttentionPhase, ContextCapacity
from lightllm.common.attention.api.workspace import AttentionWorkspace
from lightllm.models.glm5_next.triton_kernel.kpool import expand_topk, prefill_nsa_params

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def tensor(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


@pytest.mark.parametrize("counts", [(3, 0, 1, 2), (257, 0, 1, 513), (1,) * 129])
def test_prefill_metadata_is_graph_safe(counts):
    batch, queries = len(counts), sum(counts)
    positions = torch.empty(queries, device="cuda", dtype=torch.int32)
    req = torch.empty(batch, device="cuda", dtype=torch.int32)
    cu_q = torch.empty(batch + 1, device="cuda", dtype=torch.int32)
    cu_kv = torch.empty(batch + 1, device="cuda", dtype=torch.int32)

    q_prefix, kv_prefix, visible, starts, requests = [0], [0], [], [], []
    for index, count in enumerate(counts):
        prefix = 1000000 + index * 7
        visible.extend(range(prefix + 1, prefix + count + 1))
        starts.extend([kv_prefix[-1]] * count)
        requests.extend([index] * count)
        q_prefix.append(q_prefix[-1] + count)
        kv_prefix.append(kv_prefix[-1] + prefix + count)
    positions.copy_(tensor(visible) - 1)
    req.copy_(tensor(range(batch)))
    cu_q.copy_(tensor(q_prefix))
    cu_kv.copy_(tensor(kv_prefix))

    expected = (tensor(visible), tensor(requests), tensor(starts))
    actual = prefill_nsa_params(positions, req, cu_q, cu_kv)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dense", [False, True])
def test_topk_request_lookup_preserves_strided_tables(dense):
    table = torch.arange(8 * 70, device="cuda", dtype=torch.int32).reshape(8, 70)[::2, ::2]
    req = tensor([2, 2, 0, 3, 3])
    lengths = tensor([5, 7, 9, 2, 2] if dense else [17, 18, 9, 2, 2])
    groups = tensor([[3, 1], [1, 0], [0, 1], [0, 0], [0, 0]])
    actual_mem, actual_relative = expand_topk(groups, lengths, req, table, 8, dense=dense)

    expected_mem, expected_relative = torch.full_like(actual_mem, -1), torch.full_like(actual_relative, -1)
    for row, (request, length) in enumerate(zip(req.tolist(), lengths.tolist())):
        tokens = (
            list(range(length))
            if dense
            else [group * 4 + offset for group in groups[row].tolist()[: min(length // 4, 2)] for offset in range(4)]
            + list(range(length // 4 * 4, length))
        )
        expected_relative[row, : len(tokens)] = tensor(tokens)
        expected_mem[row, : len(tokens)] = table[request, tensor(tokens).long()]
    torch.testing.assert_close(actual_mem, expected_mem, rtol=0, atol=0)
    torch.testing.assert_close(actual_relative, expected_relative, rtol=0, atol=0)


def test_model_owns_query_metadata():
    lengths = torch.tensor([5, 4], device="cuda", dtype=torch.int32)
    q_lengths = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
    batch = AttentionBatch(
        phase=AttentionPhase.PREFILL,
        batch_size=2,
        token_num=3,
        total_token_num=9,
        max_q_seq_len=2,
        max_kv_seq_len=5,
        max_cache_len=3,
        b_req_idx=tensor([0, 1]),
        b_seq_len=lengths,
        b_q_seq_len=q_lengths,
        b1_cu_q_seq_len=tensor([0, 2, 3]),
        b1_cu_kv_seq_len=tensor([0, 5, 9]),
        req_to_token_indexs=torch.arange(8 * 16, device="cuda", dtype=torch.int32).reshape(8, 16),
        hold_req_id=7,
        b_ready_cache_len=tensor([3, 3]),
        position_ids=tensor([3, 4, 3]),
    )
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.models.glm5_next.infer_struct import Glm5NextInferStateInfo
    from unittest.mock import patch

    state = Glm5NextInferStateInfo()
    for name, value in vars(batch).items():
        setattr(state, name, value)
    state.is_prefill = True
    state.input_ids = torch.empty(3, device="cuda", dtype=torch.int64)
    with patch.object(InferStateInfo, "init_some_extra_state", lambda *args: None):
        state.init_some_extra_state(None)
    torch.testing.assert_close(state.sparse_queries.q_kv_seq_len, tensor([4, 5, 4]))
    torch.testing.assert_close(state.sparse_queries.q_req_idx, tensor([0, 0, 1]))
