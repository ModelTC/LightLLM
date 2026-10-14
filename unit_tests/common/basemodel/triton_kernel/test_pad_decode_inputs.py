import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.triton_kernel.pad_decode_inputs import pad_decode_inputs


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_inputs(batch_size, has_position_delta):
    return dict(
        input_ids=torch.arange(batch_size, dtype=torch.int64, device="cuda") + 2 ** 40,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32, device="cuda") + 10,
        b_mtp_index=torch.arange(batch_size, dtype=torch.int32, device="cuda") % 4,
        b_seq_len=torch.arange(batch_size, dtype=torch.int32, device="cuda") + 5,
        b_position_delta=(
            torch.arange(batch_size, dtype=torch.int32, device="cuda") - 9 if has_position_delta else None
        ),
        b_shared_seq_len=torch.arange(batch_size, dtype=torch.int32, device="cuda") % 3,
        b_shared_radix_node_id=torch.arange(batch_size, dtype=torch.int64, device="cuda") + 2 ** 40,
    )


def _reference(inputs, padded_batch_size, hold_request_id):
    return (
        F.pad(inputs["input_ids"], (0, padded_batch_size), value=1),
        F.pad(inputs["b_req_idx"], (0, padded_batch_size), value=hold_request_id),
        F.pad(inputs["b_mtp_index"], (0, padded_batch_size), value=0),
        F.pad(inputs["b_seq_len"], (0, padded_batch_size), value=2),
        (
            F.pad(inputs["b_position_delta"], (0, padded_batch_size), value=0)
            if inputs["b_position_delta"] is not None
            else None
        ),
        F.pad(inputs["b_shared_seq_len"], (0, padded_batch_size), value=0),
        F.pad(inputs["b_shared_radix_node_id"], (0, padded_batch_size), value=-1),
    )


@pytest.mark.parametrize("hold_request_id", [-1, 88])
@pytest.mark.parametrize("has_position_delta", [False, True])
@pytest.mark.parametrize(
    "batch_size,new_batch_size",
    [(0, 1), (0, 257), (1, 2), (3, 8), (255, 256), (256, 257), (257, 1025), (1025, 4096)],
)
def test_pad_decode_inputs_matches_torch(batch_size, new_batch_size, has_position_delta, hold_request_id):
    inputs = _make_inputs(batch_size, has_position_delta)
    originals = {name: value.clone() for name, value in inputs.items() if value is not None}
    expected = _reference(inputs, new_batch_size - batch_size, hold_request_id)

    actual = pad_decode_inputs(**inputs, new_batch_size=new_batch_size, hold_request_id=hold_request_id)

    for output, reference in zip(actual, expected):
        if reference is None:
            assert output is None
        else:
            assert output.dtype == reference.dtype
            assert output.device == reference.device
            assert output.shape == reference.shape
            assert torch.equal(output, reference)
    for name, original in originals.items():
        assert torch.equal(inputs[name], original)


@pytest.mark.parametrize("has_position_delta", [False, True])
def test_pad_decode_inputs_cuda_graph_replays_updated_inputs(has_position_delta):
    inputs = _make_inputs(batch_size=3, has_position_delta=has_position_delta)
    kwargs = dict(**inputs, new_batch_size=513, hold_request_id=88)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pad_decode_inputs(**kwargs)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = pad_decode_inputs(**kwargs)

    inputs["input_ids"].add_(7)
    inputs["b_req_idx"].add_(2)
    inputs["b_shared_radix_node_id"].add_(11)
    if has_position_delta:
        inputs["b_position_delta"].sub_(3)
    graph.replay()

    for output, reference in zip(outputs, _reference(inputs, padded_batch_size=510, hold_request_id=88)):
        if reference is None:
            assert output is None
        else:
            assert torch.equal(output, reference)
