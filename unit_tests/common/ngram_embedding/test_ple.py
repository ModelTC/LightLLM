import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.ngram_embedding.ple import PLELayer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def make_layer(dim=65, branches=3, embed_dim=32, kernel=4, dilation=3, dtype=torch.bfloat16):
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    width = dim * branches
    return PLELayer(
        rand(width, embed_dim) / embed_dim**0.5,
        rand(dim, embed_dim) / embed_dim**0.5,
        rand(width) * 0.1,
        rand(width) * 0.1,
        rand(width) * 0.1,
        rand(width, kernel) * 0.1,
        branches,
        dilation,
    )


def norm(x, weight, branches, eps):
    grouped = x.float().unflatten(-1, (branches, -1))
    grouped = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    return (grouped.flatten(-2) * (1 + weight.float())).to(x.dtype)


def reference_values(layer, residual, embeddings, mask=None):
    key = norm(F.linear(embeddings, layer.key_weight), layer.norm_key, layer.branches, layer.eps)
    query = norm(residual, layer.norm_query, layer.branches, layer.eps)
    key, query = [x.unflatten(-1, (layer.branches, layer.hidden_size)) for x in [key, query]]
    value = F.linear(embeddings, layer.value_weight)
    gate = (key * query).sum(-1, keepdim=True) / math.sqrt(layer.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    normalized = norm(gated, layer.norm_conv, layer.branches, layer.eps)
    if mask is not None:
        gated = gated * mask[:, None]
        normalized = normalized * mask[:, None]
    return gated, normalized


def reference_conv(layer, values, gated, lengths, history):
    result, next_state = [], []
    offset = 0
    for request, count in enumerate(lengths):
        current = values[offset : offset + count]
        full = torch.cat([history[request], current])
        if count:
            conv = (
                F.conv1d(
                    full.T.unsqueeze(0),
                    layer.conv_weight.unsqueeze(1),
                    dilation=layer.dilation,
                    groups=values.shape[1],
                )
                .squeeze(0)
                .T
            )
            result.append(gated[offset : offset + count] + F.silu(conv))
        next_state.append(full[-layer.context_len :] if layer.context_len else full[:0])
        offset += count
    output = torch.cat(result) if result else values.new_empty((0, values.shape[1]))
    state = torch.stack(next_state) if next_state else history.clone()
    return output, state


def starts_for(lengths):
    return torch.tensor([0] + list(lengths), device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)


def assert_close(actual, expected, dtype):
    tolerance = max(2e-6, torch.finfo(dtype).eps * 2)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim,branches,kernel,dilation", [(65, 3, 4, 3), (128, 4, 3, 2), (32, 2, 1, 1)])
@pytest.mark.parametrize("masked", [False, True])
def test_ple_reference_equations(dtype, dim, branches, kernel, dilation, masked):
    torch.manual_seed(137)
    layer = make_layer(dim, branches, kernel=kernel, dilation=dilation, dtype=dtype)
    lengths = [0, 1, 2, 13, 0, 7]
    count = sum(lengths)
    residual = torch.randn(count, dim * branches, device="cuda", dtype=dtype)
    embeddings = torch.randn(count, 32, device="cuda", dtype=dtype)
    mask = (torch.arange(count, device="cuda") % 3 != 0) if masked else None
    history = layer.empty_history(len(lengths)).normal_()
    original = history.clone()
    actual, conv_inputs = layer.forward(residual, embeddings, starts_for(lengths), history, mask)
    gated, expected_conv_inputs = reference_values(layer, residual, embeddings, mask)
    expected, expected_state = reference_conv(layer, expected_conv_inputs, gated, lengths, history)
    assert_close(conv_inputs, expected_conv_inputs, dtype)
    assert_close(actual, expected, dtype)
    assert_close(layer.history_after(conv_inputs, starts_for(lengths), history), expected_state, dtype)
    torch.testing.assert_close(history, original, atol=0, rtol=0)


def test_production_hidden_width():
    torch.manual_seed(137)
    layer = make_layer(dim=2560, branches=4, embed_dim=2560)
    residual = torch.randn(2, 10240, device="cuda", dtype=torch.bfloat16)
    embeddings = torch.randn(2, 2560, device="cuda", dtype=torch.bfloat16)
    history = layer.empty_history(2)
    output, inputs = layer.forward(residual, embeddings, starts_for([1, 1]), history)
    gated, expected_inputs = reference_values(layer, residual, embeddings)
    expected, _ = reference_conv(layer, expected_inputs, gated, [1, 1], history)
    assert_close(inputs, expected_inputs, residual.dtype)
    assert_close(output, expected, residual.dtype)


def test_chunking_speculative_acceptance_and_request_isolation():
    torch.manual_seed(137)
    layer = make_layer(dim=32, branches=2, dtype=torch.float32)
    lengths = [8, 6, 2]
    starts = starts_for(lengths)
    residual = torch.randn(sum(lengths), 64, device="cuda")
    embeddings = torch.randn(sum(lengths), 32, device="cuda")
    history = layer.empty_history(3).normal_()
    full, full_inputs = layer.forward(residual, embeddings, starts, history)
    full_state = layer.history_after(full_inputs, starts, history)
    state = history.clone()
    consumed = [0] * 3
    per_request = [[] for _ in range(3)]
    for counts in [(1, 0, 1), (2, 3, 1), (1, 1, 0), (4, 2, 0)]:
        slices = [slice(int(starts[i]) + consumed[i], int(starts[i]) + consumed[i] + n) for i, n in enumerate(counts)]
        chunks = [torch.cat([tensor[s] for s in slices]) for tensor in [residual, embeddings]]
        chunk_starts = starts_for(counts)
        output, conv_inputs = layer.forward(*chunks, chunk_starts, state)
        state = layer.history_after(conv_inputs, chunk_starts, state)
        for i, n in enumerate(counts):
            per_request[i].append(output[int(chunk_starts[i]) : int(chunk_starts[i + 1])])
            consumed[i] += n
    assert_close(torch.cat([torch.cat(parts) for parts in per_request]), full, residual.dtype)
    assert_close(state, full_state, residual.dtype)
    accepted = torch.tensor([1, 4, 0], device="cuda", dtype=torch.int32)
    committed = layer.history_after(full_inputs, starts, history, accepted)
    expected = torch.stack(
        [
            torch.cat([history[i], full_inputs[int(starts[i]) : int(starts[i]) + n]])[-layer.context_len :]
            for i, n in enumerate([1, 4, 0])
        ]
    )
    torch.testing.assert_close(committed, expected, atol=0, rtol=0)

    order = torch.tensor([2, 0, 2], device="cuda")
    reordered = committed.index_select(0, order)
    new_residual, new_embeddings = residual[:3], embeddings[:3]
    actual, _ = layer.forward(new_residual, new_embeddings, starts_for([1, 1, 1]), reordered)
    gated, normalized = reference_values(layer, new_residual, new_embeddings)
    expected, _ = reference_conv(layer, normalized, gated, [1, 1, 1], reordered)
    assert_close(actual, expected, residual.dtype)


def test_checkpoint_binding_and_graph_replay():
    layer = make_layer(dim=32, branches=2)
    prefix = "model.language_model.layers.1.ple"
    weights = {
        f"{prefix}.{name}": tensor
        for name, tensor in [
            ("key_proj.weight", layer.key_weight),
            ("value_proj.weight", layer.value_weight),
            ("norm_key.weight", layer.norm_key),
            ("norm_query.weight", layer.norm_query),
            ("norm_conv.weight", layer.norm_conv),
            ("conv1d.weight", layer.conv_weight.unsqueeze(1)),
        ]
    }
    bound = PLELayer.from_checkpoint(weights, prefix, branches=2, dilation=3)
    assert bound.key_weight is layer.key_weight
    assert bound.conv_weight.data_ptr() == layer.conv_weight.data_ptr()
    residual = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
    embeddings = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)
    starts = starts_for([3, 2, 2])
    accepted = torch.tensor([3, 1, 2], device="cuda", dtype=torch.int32)
    mask = torch.ones(7, device="cuda", dtype=torch.bool)
    history = layer.empty_history(3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _, conv_inputs = bound.forward(residual, embeddings, starts, history, mask)
            bound.history_after(conv_inputs, starts, history, accepted)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, conv_inputs = bound.forward(residual, embeddings, starts, history, mask)
        next_state = bound.history_after(conv_inputs, starts, history, accepted)
    for lengths, counts in [([2, 0, 5], [1, 0, 4]), ([0, 3, 4], [0, 2, 0])]:
        residual.normal_()
        embeddings.normal_()
        history.normal_()
        starts.copy_(starts_for(lengths))
        accepted.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        mask[::2] = False
        graph.replay()
        gated, expected_inputs = reference_values(bound, residual, embeddings, mask)
        expected, _ = reference_conv(bound, expected_inputs, gated, lengths, history)
        assert_close(output, expected, residual.dtype)
        expected_state = []
        for i, count in enumerate(counts):
            expected_state.append(
                torch.cat([history[i], expected_inputs[int(starts[i]) : int(starts[i]) + count]])[-bound.context_len :]
            )
        assert_close(next_state, torch.stack(expected_state), residual.dtype)


def test_empty_batches_and_invalid_inputs():
    layer = make_layer(dim=32, branches=2)
    for lengths in [[], [0, 0, 0]]:
        residual = layer.key_weight.new_empty((0, 64))
        embeddings = layer.key_weight.new_empty((0, 32))
        history = layer.empty_history(len(lengths))
        output, conv_inputs = layer.forward(residual, embeddings, starts_for(lengths), history)
        assert output.shape == (0, 64)
        torch.testing.assert_close(
            layer.history_after(conv_inputs, starts_for(lengths), history), history, atol=0, rtol=0
        )
    with pytest.raises(ValueError, match="projection"):
        replace(layer, value_weight=layer.value_weight[:1])
    with pytest.raises(ValueError, match="convolution"):
        replace(layer, conv_weight=layer.conv_weight[:, :0])
    residual = layer.key_weight.new_empty((1, 64))
    embeddings = layer.key_weight.new_empty((1, 32))
    with pytest.raises(ValueError, match="history"):
        layer.forward(residual, embeddings, starts_for([1]), layer.empty_history(2))
    with pytest.raises(ValueError, match="mask"):
        layer.forward(residual, embeddings, starts_for([1]), layer.empty_history(1), torch.ones(1, device="cuda"))
    with pytest.raises(ValueError, match="accepted"):
        layer.history_after(residual, starts_for([1]), layer.empty_history(1), torch.ones(1, device="cuda"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gate_zero_near_zero_and_signed_saturation(dtype):
    from lightllm.common.ngram_embedding.ple_kernels import _gate

    dots = torch.tensor([0, 1e-8, -1e-8, 1e-5, -1e-5, 100, -100], device="cuda", dtype=dtype)
    key = torch.zeros(7, 4, device="cuda", dtype=dtype)
    query = torch.zeros_like(key)
    key[:, 0] = dots * 2
    query[:, 0] = 1
    value = torch.tensor([[-2, -1, 1, 2]], device="cuda", dtype=dtype).expand(7, -1).contiguous()
    gate = (key * query).sum(-1, keepdim=True) / 2
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    expected = torch.sigmoid(gate) * value
    actual = torch.empty_like(key)
    _gate[(7, 1)](key, query, value, None, actual, 4, 1, False, 4, enable_fp_fusion=False)
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-7)
    else:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
