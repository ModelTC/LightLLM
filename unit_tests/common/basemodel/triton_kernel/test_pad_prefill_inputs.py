import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.triton_kernel.pad_prefill_inputs import pad_prefill_inputs


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_inputs(batch_size, handle_token_num):
    return dict(
        input_ids=torch.arange(handle_token_num, dtype=torch.int64, device="cuda") + 2 ** 40,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32, device="cuda") + 10,
        b_mtp_index=torch.arange(batch_size, dtype=torch.int32, device="cuda") % 4,
        b_seq_len=torch.arange(batch_size, dtype=torch.int32, device="cuda") + 5,
        b_ready_cache_len=torch.arange(batch_size, dtype=torch.int32, device="cuda") % 3,
        b_is_decode_req=torch.arange(batch_size, device="cuda") % 2 == 0,
    )


def _reference(inputs, padded_token_num, hold_request_id):
    return (
        F.pad(inputs["input_ids"], (0, padded_token_num), value=1),
        F.pad(inputs["b_req_idx"], (0, 1), value=hold_request_id),
        F.pad(inputs["b_mtp_index"], (0, 1), value=0),
        F.pad(inputs["b_seq_len"], (0, 1), value=padded_token_num),
        F.pad(inputs["b_ready_cache_len"], (0, 1), value=0),
        F.pad(inputs["b_is_decode_req"], (0, 1), value=False),
    )


@pytest.mark.parametrize("hold_request_id", [-1, 88])
@pytest.mark.parametrize(
    "batch_size,handle_token_num,padded_token_num",
    [
        (0, 0, 1),
        (0, 0, 129),
        (1, 1, 3),
        (3, 17, 1),
        (257, 2051, 11),
        (1025, 4099, 257),
        (3, 65537, 7),
    ],
)
def test_pad_prefill_inputs_matches_torch(batch_size, handle_token_num, padded_token_num, hold_request_id):
    inputs = _make_inputs(batch_size, handle_token_num)
    originals = {name: value.clone() for name, value in inputs.items()}
    expected = _reference(inputs, padded_token_num, hold_request_id)

    actual = pad_prefill_inputs(
        **inputs,
        new_handle_token_num=handle_token_num + padded_token_num,
        hold_request_id=hold_request_id,
    )

    for output, reference in zip(actual, expected):
        assert output.dtype == reference.dtype
        assert output.device == reference.device
        assert output.shape == reference.shape
        assert torch.equal(output, reference)
    for name, original in originals.items():
        assert torch.equal(inputs[name], original)


def test_pad_prefill_inputs_cuda_graph_replays_updated_inputs():
    inputs = _make_inputs(batch_size=3, handle_token_num=17)
    kwargs = dict(**inputs, new_handle_token_num=24, hold_request_id=88)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pad_prefill_inputs(**kwargs)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = pad_prefill_inputs(**kwargs)

    inputs["input_ids"].add_(7)
    inputs["b_req_idx"].add_(2)
    inputs["b_is_decode_req"].logical_not_()
    graph.replay()

    for output, reference in zip(outputs, _reference(inputs, padded_token_num=7, hold_request_id=88)):
        assert torch.equal(output, reference)
