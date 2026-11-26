from unit_tests.common.attention.backends.step_fixture import create_att_state
from lightllm.common.attention.backends.gqa import GQAAttConfig
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from unit_tests.common.attention.backends.step_fixture import make_infer_state
from lightllm.common.attention.backends.gqa.triton_quant import (
    TritonQuantGQADecodeBackend,
)
from lightllm.common.basemodel.triton_kernel import diverse_utils


@pytest.mark.parametrize(
    "node_ids,expected_markers",
    [
        ([10, 10, 10, 10, 20, 20], [0, 0, 3, 1, 0, 2]),
        ([10, 10, 20, 10, 10], [0, 2, 1, 0, 2]),
        ([10, 10, -1, -1], [0, 2, 1, 1]),
        ([-1, -1, -1, -1], [1, 1, 1, 1]),
    ],
)
def test_build_diverse_shared_group_markers(monkeypatch, node_ids, expected_markers):
    monkeypatch.setattr(diverse_utils, "get_diverse_max_batch_shared_group_size", lambda: 3)
    b_shared_radix_node_id = torch.tensor(node_ids, dtype=torch.int64, device="cuda")

    markers = diverse_utils.build_diverse_shared_group_markers(
        b_shared_radix_node_id=b_shared_radix_node_id,
    )

    assert markers.cpu().tolist() == expected_markers


def _batch(shared_len, node_ids, *, graph_capacity=None, microbatch_index=0):
    rows = len(shared_len)
    return make_infer_state(
        req_to_token_indexs=torch.zeros(rows, 8, dtype=torch.int32, device="cuda"),
        b_req_idx=torch.arange(rows, dtype=torch.int32, device="cuda"),
        b_seq_len=torch.full((rows,), 8, dtype=torch.int32, device="cuda"),
        batch_size=rows,
        max_kv_seq_len=8,
        total_token_num=8 * rows,
        b_shared_seq_len=torch.tensor(shared_len, dtype=torch.int32, device="cuda"),
        b_shared_radix_node_id=torch.tensor(node_ids, dtype=torch.int64, device="cuda"),
        cuda_graph_max_kv_seq_len=graph_capacity,
        microbatch_index=microbatch_index,
    )


def test_int8kv_decode_step_builder_rebuilds_diverse_metadata(monkeypatch):
    monkeypatch.setattr(diverse_utils, "get_diverse_max_batch_shared_group_size", lambda: 3)
    step_builder = TritonQuantGQADecodeBackend(
        att_config=GQAAttConfig(
            num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, kv_quant="int8_group", kv_quant_group_size=8
        ),
        runtime_config=TritonQuantGQADecodeBackend.RuntimeConfig(diverse_decode=True),
    )

    att_state = create_att_state(backend=step_builder, infer_state=_batch([8, 8, 5, 0], [10, 10, 20, -1]))

    assert att_state.b_mark_shared_group.cpu().tolist() == [0, 2, 1, 1]
    assert att_state.b_shared_seq_len.cpu().tolist() == [8, 8, 0, 0]


def test_int8kv_decode_step_builder_keeps_graph_metadata_per_microbatch_and_batch_size(monkeypatch):
    monkeypatch.setattr(diverse_utils, "get_diverse_max_batch_shared_group_size", lambda: 3)
    step_builder = TritonQuantGQADecodeBackend(
        att_config=GQAAttConfig(
            num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, kv_quant="int8_group", kv_quant_group_size=8
        ),
        runtime_config=TritonQuantGQADecodeBackend.RuntimeConfig(diverse_decode=True),
    )
    captured = create_att_state(
        backend=step_builder, infer_state=_batch([8, 8, 5, 0], [10, 10, 20, -1], graph_capacity=64)
    )
    replayed = create_att_state(backend=step_builder, infer_state=_batch([4, 4, 4, 4], [7, 7, 7, 9], graph_capacity=64))
    # 重放会就地重写捕获的张量，因此图会读取新的组。
    assert replayed.b_mark_shared_group is captured.b_mark_shared_group
    assert replayed.b_shared_seq_len is captured.b_shared_seq_len
    assert captured.b_mark_shared_group.cpu().tolist() == [0, 0, 3, 1]
    assert captured.b_shared_seq_len.cpu().tolist() == [4, 4, 4, 0]
    other = create_att_state(
        backend=step_builder, infer_state=_batch([8, 8, 5, 0], [10, 10, 20, -1], graph_capacity=64, microbatch_index=1)
    )
    assert other.b_mark_shared_group is not captured.b_mark_shared_group
    eager = create_att_state(backend=step_builder, infer_state=_batch([8, 8, 5, 0], [10, 10, 20, -1]))
    assert eager.b_mark_shared_group is not captured.b_mark_shared_group
