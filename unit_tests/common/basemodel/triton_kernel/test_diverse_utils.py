from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from lightllm.common.basemodel.attention.triton import int8kv as int8kv_module
from lightllm.common.basemodel.attention.triton.int8kv import Int8kvTritonDecodeAttState
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


def test_int8kv_decode_state_rebuilds_diverse_metadata(monkeypatch):
    monkeypatch.setattr(int8kv_module, "enable_diverse_mode_gqa_decode_fast_kernel", lambda: True)
    monkeypatch.setattr(diverse_utils, "get_diverse_max_batch_shared_group_size", lambda: 3)
    infer_state = SimpleNamespace(
        b_shared_seq_len=torch.tensor([8, 8, 5, 0], dtype=torch.int32, device="cuda"),
        b_shared_radix_node_id=torch.tensor([10, 10, 20, -1], dtype=torch.int64, device="cuda"),
    )
    state = Int8kvTritonDecodeAttState(backend=SimpleNamespace(), infer_state=infer_state)

    state.init_state()

    assert state.b_mark_shared_group.cpu().tolist() == [0, 2, 1, 1]
    assert state.b_shared_seq_len.cpu().tolist() == [8, 8, 0, 0]
