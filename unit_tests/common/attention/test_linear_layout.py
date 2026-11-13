from dataclasses import replace

import pytest
import torch

from lightllm.common.attention.backends._shared.linear.layout import LinearDecodeContext, update_linear_decode
from lightllm.common.attention.backends.base import BaseAttentionBackend
from unit_tests.common.attention.helpers import make_batch, make_context, make_layer_attention, make_spec, model_stub


class _LinearBackend(BaseAttentionBackend):
    def _allocate_decode_context(self, capacity, capture, workspace):
        return LinearDecodeContext(capacity, capture, workspace)

    def _update_decode(self, context):
        update_linear_decode(context, 2, self.dynamic_verify, replay=self.replay)

    def _forward_decode(self, layer, context, inputs, *, out=None):
        return context.accepted_tokens(layer.params)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_shared_backend_selects_each_states_cursor_through_graph_replay(dynamic, replay):
    backend = _LinearBackend(model_stub(), make_spec())
    backend.dynamic_verify, backend.replay = dynamic, replay
    requests = [0, 0, 2, 7, 7, 7] if dynamic else [0, 0, 0, 2, 2, 2]
    positions = [0, 1, 0, 0, 0, 0] if dynamic else [0, 1, 2, 0, 1, 2]
    batch = make_batch(
        (5,) * 6,
        device="cuda",
        b_req_idx=torch.tensor(requests, dtype=torch.int32, device="cuda"),
        b_mtp_index=torch.tensor(positions, dtype=torch.int32, device="cuda"),
    )
    context = make_context(backend, batch, capture=True, prepare=True)
    first = torch.tensor([0, 0, 2, 0, 0, 0, 0, 99], dtype=torch.int32, device="cuda")
    second = torch.tensor([2, 0, 1, 0, 0, 0, 0, 99], dtype=torch.int32, device="cuda")
    layer_a = make_layer_attention(backend, context, params=first)
    layer_b = replace(layer_a, logical_layer_id=1, params=second)

    def step():
        backend.update(context)
        a, b = layer_a.backend.forward(layer_a, context, None), layer_b.backend.forward(layer_b, context, None)
        assert layer_a.backend.forward(layer_a, context, None) is a
        return a, b

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        a, b = step()
    graph.replay()
    tail = [1] * 4 if dynamic else []
    assert a.tolist() == [1, 3] + tail
    assert b.tolist() == [3, 2] + tail
    first[0], first[2] = 1, 0
    second[0], second[2] = 0, 2
    graph.replay()
    assert a.tolist() == [2, 1] + tail
    assert b.tolist() == [1, 3] + tail
    # A new eager batch must not reuse the previous batch's gathered counts.
    backend.prepare(context, batch)
    a, b = step()
    assert a.tolist() == [2, 1] + tail
    assert b.tolist() == [1, 3] + tail
    expected_requests = torch.tensor([0, 2] + ([7] * 4 if dynamic else []), device="cuda")
    torch.testing.assert_close(context.b_conv_buffer_idx.long(), expected_requests)
    expected_ssm = expected_requests if replay else expected_requests[:, None] * 3 + torch.arange(3, device="cuda")
    torch.testing.assert_close(context.b_ssm_buffer_idx.long(), expected_ssm)
