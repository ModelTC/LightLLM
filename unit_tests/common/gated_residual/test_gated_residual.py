import pytest
import torch
import torch.nn.functional as F

from lightllm.common.gated_residual import GatedResidual

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def make_component(dim, branches, dtype, combine=True):
    width, rank = dim * branches, 32

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    return GatedResidual(
        rand(width) * 0.1,
        rand(rank, width) / width**0.5,
        rand(width, rank) / rank**0.5,
        rand(branches, width) / width**0.5 if combine else None,
        branches,
    )


def reference_read(component, x):
    grouped = x.float().unflatten(-1, (component.branches, -1))
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + component.eps)
    normalized = (normalized.flatten(-2) * (1 + component.norm_weight.float())).to(x.dtype)
    lowrank = F.silu(F.linear(normalized, component.down_weight) / component.branches)
    gates = torch.sigmoid(F.linear(lowrank, component.up_weight))
    mixed = (normalized * gates).unflatten(-1, (component.branches, -1)).mean(-2)
    if component.inject_weight is None:
        return mixed
    injection = 2 * torch.sigmoid(F.linear(normalized, component.inject_weight) / component.branches)
    return mixed, injection


def reference_write(residual, update, gates):
    return residual + (update.unsqueeze(-2) * gates.unsqueeze(-1)).flatten(-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,dim,branches", [(1, 65, 3), (17, 128, 4), (2, 2560, 4)])
@pytest.mark.parametrize("combine", [True, False])
def test_reference_parity(dtype, tokens, dim, branches, combine):
    torch.manual_seed(37)
    component = make_component(dim, branches, dtype, combine)
    residual = torch.randn(1, tokens, dim * branches, device="cuda", dtype=dtype)
    original = residual.clone()
    actual, expected = component.read(residual), reference_read(component, residual)

    tolerance = max(1e-6, torch.finfo(dtype).eps)
    if combine:
        for result, ref in zip(actual, expected):
            torch.testing.assert_close(result, ref, atol=tolerance, rtol=tolerance)
        update = torch.randn(1, tokens, dim, device="cuda", dtype=dtype)
        written = component.write(residual, update, actual[1])

        torch.testing.assert_close(written, reference_write(residual, update, actual[1]), atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(residual, original, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim,branches", [(65, 3), (2560, 4)])
def test_injection_gate_projection_is_independent_of_batch(dtype, dim, branches):
    from lightllm.common.gated_residual.kernels import injection_gates

    torch.manual_seed(3107)
    x = torch.randn(9, dim * branches, device="cuda", dtype=dtype)
    weight = torch.randn(branches, dim * branches, device="cuda", dtype=dtype) * 0.01
    actual = injection_gates(x, weight, branches)
    singles = torch.cat([injection_gates(row[None], weight, branches) for row in x])
    chunks = torch.cat([injection_gates(x[start : start + 3], weight, branches) for start in range(0, len(x), 3)])
    torch.testing.assert_close(actual, singles, atol=0, rtol=0)
    torch.testing.assert_close(actual, chunks, atol=0, rtol=0)
    expected = 2 * torch.sigmoid((x.double() @ weight.double().T).to(dtype) / branches)
    tolerance = max(1e-6, torch.finfo(dtype).eps)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_saturated_mix_and_write():
    from lightllm.common.gated_residual.kernels import mix_streams

    component = make_component(65, 3, torch.bfloat16)
    x = torch.randn(2, 195, device="cuda", dtype=torch.bfloat16)
    logits = torch.linspace(-100, 100, 195, device="cuda", dtype=x.dtype).expand(2, -1).contiguous()
    expected = (x * torch.sigmoid(logits)).view(2, 3, 65).mean(1)
    torch.testing.assert_close(mix_streams(x, logits, 3), expected, atol=0, rtol=0)
    gates = torch.tensor([[0, 1, 2], [2, 0, 1]], device="cuda", dtype=x.dtype)
    update = torch.randn(2, 65, device="cuda", dtype=x.dtype)
    torch.testing.assert_close(component.write(x, update, gates), reference_write(x, update, gates), atol=0, rtol=0)


def test_checkpoint_binding_and_graph_replay():
    component = make_component(128, 4, torch.bfloat16)
    prefix = "model.language_model.layers.0.attn_hyper_connection"
    weights = {
        f"{prefix}.hc_norm.weight": component.norm_weight,
        f"{prefix}.input_mix_weight_down.weight": component.down_weight,
        f"{prefix}.input_mix_weight_up.weight": component.up_weight,
        f"{prefix}.block_inject_weight.weight": component.inject_weight,
    }
    bound = GatedResidual.from_checkpoint(weights, prefix, branches=4)
    assert bound.down_weight is component.down_weight
    final = GatedResidual.from_checkpoint(weights, prefix, branches=4, use_combine=False)
    residual = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    update = torch.randn(3, 128, device="cuda", dtype=residual.dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _, gates = bound.read(residual)
            bound.write(residual, update, gates)
            final.read(residual)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mixed, gates = bound.read(residual)
        output = bound.write(residual, update, gates)
        final_output = final.read(output)
    for _ in range(2):
        residual.normal_()
        graph.replay()
        expected_mixed, expected_gates = reference_read(bound, residual)
        expected_output = reference_write(residual, update, expected_gates)
        torch.testing.assert_close(mixed, expected_mixed, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(output, expected_output, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(final_output, reference_read(final, expected_output), atol=2e-3, rtol=2e-3)


def test_empty_and_invalid_inputs():
    component = make_component(65, 3, torch.bfloat16)
    x = torch.empty(0, 195, device="cuda", dtype=torch.bfloat16)
    mixed, gates = component.read(x)
    assert component.write(x, mixed, gates).shape == x.shape
    with pytest.raises(ValueError, match="last dimension"):
        component.read(torch.empty(2, 194, device="cuda", dtype=x.dtype))
    with pytest.raises(ValueError, match="contiguous"):
        component.read(torch.empty(2, 390, device="cuda", dtype=x.dtype)[:, ::2])
    with pytest.raises(ValueError, match="matching token"):
        component.write(x, mixed, torch.empty(1, 3, device="cuda", dtype=x.dtype))
    with pytest.raises(ValueError, match="share"):
        component.read(x.float())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim,branches,rank", [(65, 3, 9), (2560, 4, 320)])
def test_mixer_is_batch_invariant_and_matches_fp64(dtype, dim, branches, rank):
    from lightllm.common.gated_residual.kernels import grouped_norm

    torch.manual_seed(701)
    width = dim * branches

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    component = GatedResidual(
        rand(width) * 0.1,
        rand(rank, width) / width**0.5,
        rand(width, rank) / rank**0.5,
        rand(branches, width) / width**0.5,
        branches,
    )
    residual = rand(129, width)
    expected = [component.read(row[None, :]) for row in residual]
    scalar = tuple(torch.cat([item[index] for item in expected]) for index in range(2))
    for count in (3, 8, 16, 24, 64, 101, 128, 129):
        outputs = [component.read(residual[start : start + count]) for start in range(0, 129, count)]
        for index in range(2):
            torch.testing.assert_close(torch.cat([item[index] for item in outputs]), scalar[index], atol=0, rtol=0)
    normalized = grouped_norm(residual, component.norm_weight, branches, component.eps)
    projected = F.linear(normalized.double(), component.down_weight.double()).to(dtype)
    lowrank = F.silu(projected / branches)
    logits = F.linear(lowrank.double(), component.up_weight.double()).to(dtype)
    oracle = (normalized * torch.sigmoid(logits)).unflatten(-1, (branches, dim)).mean(-2)
    torch.testing.assert_close(scalar[0], oracle, atol=2e-3, rtol=torch.finfo(dtype).eps)
