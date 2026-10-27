"""MTP hidden inputs must survive capture and receive every replay's values."""
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.infer_struct import InferStateInfo


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('batch', [1, 4, 8])
def test_captured_mtp_hidden_input_is_updated_on_replay(device, batch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')

    captured = []

    def pre_forward(input_ids, state, weight):
        captured.append(state.mtp_draft_input_hiddens)
        return state.mtp_draft_input_hiddens * 2

    model = SimpleNamespace(
        pre_infer=SimpleNamespace(token_forward=pre_forward,
                                  _tpsp_sp_split=lambda input, infer_state: input),
        pre_post_weight=None,
        layers_num=0,
        post_infer=SimpleNamespace(_tpsp_allgather=lambda input, infer_state: input,
                                   token_forward=lambda input, infer_state, layer_weight: input),
        _create_model_output=lambda output, state: SimpleNamespace(
            logits=output, to_no_ref_tensor=lambda: None),
    )

    def state(value):
        result = InferStateInfo()
        result.input_ids = torch.zeros(batch, dtype=torch.int64, device=device)
        result.mtp_draft_input_hiddens = torch.full((batch, 32), value, device=device)
        result.hidden_collector = SimpleNamespace(add_final_hidden=lambda tensor: None)
        result.decode_att_state = SimpleNamespace(copy_for_decode_cuda_graph=lambda other: None)
        result.is_cuda_graph_capture = True
        return result

    graph_state = state(1.0)
    if device == 'cuda':
        warmup = state(1.0)
        TpPartBaseModel._token_forward(model, warmup)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = TpPartBaseModel._token_forward(model, graph_state)
    else:
        output = TpPartBaseModel._token_forward(model, graph_state)
        # A CPU stand-in for CUDA Graph's captured input address.
        graph = SimpleNamespace(replay=lambda: output.logits.copy_(captured[-1] * 2))

    cache = CudaGraph.__new__(CudaGraph)
    cache.graph = {batch: (graph, graph_state, output)}
    for value in (3.0, 7.0):
        actual = cache._replay(state(value))
        if device == 'cuda':
            torch.cuda.synchronize()
        torch.testing.assert_close(actual.logits, torch.full((batch, 32), value * 2, device=device))
