import pytest
import torch

from lightllm.models.qwen4_exp.batch import PackedBatchMetadata
from lightllm.models.qwen4_exp.state import QSABatch
from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def prepare(requests, lengths, query_lengths, hold=7):
    tensors = [torch.tensor(value, device="cuda", dtype=torch.int32) for value in (requests, lengths, query_lengths)]
    return tensors, PackedBatchMetadata.prepare(*tensors, sum(query_lengths), hold, 4)


def check(metadata, requests, lengths, query_lengths, hold=7):
    counts = list(query_lengths)
    for row, request in enumerate(requests):
        if request != hold and row + 1 < len(requests) and requests[row + 1] == request:
            counts[row + 1] += counts[row]
            counts[row] = 0
    starts, active, key_lengths, first, rows, visible, mask = [0], [], [], [], [], [], []
    for request, length, count in zip(requests, lengths, counts):
        real = request != hold and count > 0
        starts.append(starts[-1] + count)
        active.append(request if real else -1)
        key_lengths.append(length if real else 0)
        first.append((length - count) // 4 if real else 0)
        rows.extend([request] * count)
        visible.extend(range(length - count + 1, length + 1) if real else [0] * count)
        mask.extend([real] * count)
    expected = [starts, active, key_lengths, first, rows, visible, mask]
    for name, values in zip(metadata.__dataclass_fields__, expected):
        actual = getattr(metadata, name)
        torch.testing.assert_close(actual, torch.tensor(values, device="cuda", dtype=actual.dtype))


@pytest.mark.parametrize(
    "requests,lengths,query_lengths",
    [
        ([5, 2, 7], [13, 21, 4], [13, 5, 4]),
        ([5, 2, 7, 7], [14, 22, 2, 2], [1, 1, 1, 1]),
        ([5, 5, 5, 2, 2, 7, 7], [14, 15, 16, 22, 23, 2, 2], [1] * 7),
        ([5, 2, 7], [13, 21, 0], [0, 0, 0]),
        ([5, 2], [1500, 1900], [1200, 1700]),
        ([], [], []),
    ],
)
def test_native_packed_batch_metadata(requests, lengths, query_lengths):
    _, metadata = prepare(requests, lengths, query_lengths)
    check(metadata, requests, lengths, query_lengths)
    tokens = sum(query_lengths)
    slots = torch.zeros(8, 2000, device="cuda", dtype=torch.int32)
    memory = torch.arange(tokens, device="cuda", dtype=torch.int32)
    rotary = torch.ones(tokens, 4, device="cuda", dtype=torch.bfloat16)
    batch = QSABatch.from_metadata(metadata, slots, memory, rotary, rotary, 4)
    assert batch.slot_table.data_ptr() == slots.data_ptr()
    assert batch.active_rows.data_ptr() == metadata.active_rows.data_ptr()
    assert batch.mem_indices.dtype == torch.int64
    assert batch.max_new_blocks == (tokens + 3) // 4


def test_metadata_visibility_matches_existing_native_speculative_layout():
    requests, lengths, query_lengths = [5, 5, 5, 2, 2, 7, 7], [14, 15, 16, 22, 23, 2, 2], [1] * 7
    tensors, metadata = prepare(requests, lengths, query_lengths)
    slots = torch.arange(8 * 32, device="cuda", dtype=torch.int32).reshape(8, 32)
    ragged = torch.empty(sum([16, 23, 2, 2]), device="cuda", dtype=torch.int32)
    _, _, native_lengths = gen_nsa_ks_ke(tensors[1], tensors[2], tensors[0], slots, 7, ragged, hold_req_idx=7)
    torch.testing.assert_close(metadata.visible_lengths[metadata.token_mask], native_lengths[metadata.token_mask])


def test_metadata_graph_replay_and_copy_preserve_buffer_addresses():
    requests, lengths, query_lengths = [5, 5, 2, 7, 7], [14, 15, 22, 2, 2], [1] * 5
    tensors, metadata = prepare(requests, lengths, query_lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = PackedBatchMetadata.prepare(*tensors, 5, 7, 4)
    pointers = {name: getattr(captured, name).data_ptr() for name in captured.__dataclass_fields__}
    for ids, sizes in [([1, 3, 5, 7, 7], [25, 41, 13, 2, 2]), ([2, 2, 2, 5, 7], [12, 13, 14, 5, 2])]:
        tensors[0].copy_(torch.tensor(ids, device="cuda", dtype=torch.int32))
        tensors[1].copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
        graph.replay()
        check(captured, ids, sizes, query_lengths)
        metadata.copy_from(captured)
        check(metadata, ids, sizes, query_lengths)
        assert pointers == {name: getattr(captured, name).data_ptr() for name in captured.__dataclass_fields__}
    _, incompatible = prepare([1], [1], [1])
    with pytest.raises(ValueError, match="Incompatible graph batch metadata"):
        metadata.copy_from(incompatible)


def test_metadata_rejects_incompatible_input_shapes():
    values = torch.ones(2, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="matching contiguous"):
        PackedBatchMetadata.prepare(values, values[:1], values, 2, 7, 4)
    with pytest.raises(ValueError, match="compression ratio"):
        PackedBatchMetadata.prepare(values, values, values, 2, 7, 0)
