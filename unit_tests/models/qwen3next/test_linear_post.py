from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerInfer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

TOKENS, HEADS, DIM = 5, 4, 128


@pytest.mark.parametrize("gate_rank", [2, 3], ids=["graph_wrapper_2d", "layer_inputs_3d"])
def test_gate_reaches_the_norm_without_a_copy(gate_rank):
    torch.manual_seed(0)
    # The gate is a column slice of the packed input projection, as both callers pass it.
    projection = torch.randn(TOKENS, 3 * HEADS * DIM, device="cuda", dtype=torch.bfloat16)
    z = projection[:, HEADS * DIM : 2 * HEADS * DIM]
    if gate_rank == 3:
        z = z.view(TOKENS, HEADS, DIM)
    core = torch.randn(TOKENS, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(DIM, device="cuda", dtype=torch.bfloat16)
    gates = []

    def linear_norm(x, gate, eps):
        gates.append(gate)
        return gated_rmsnorm_forward(x=x, weight=weight, bias=None, eps=eps, z=gate)

    layer = Qwen3NextTransformerLayerInfer.__new__(Qwen3NextTransformerLayerInfer)
    layer.eps_ = 1e-6
    layer_weight = SimpleNamespace(linear_norm=linear_norm, linear_out_proj=SimpleNamespace(mm=lambda x: x))
    out = layer._linear_post(core, z, layer_weight)

    (gate,) = gates
    assert gate.data_ptr() == z.data_ptr() and not gate.is_contiguous()
    contiguous_gate = z.reshape(TOKENS * HEADS, DIM).contiguous()
    expected = gated_rmsnorm_forward(x=core.view(-1, DIM), weight=weight, bias=None, eps=1e-6, z=contiguous_gate)
    assert torch.equal(out, expected.view(TOKENS, -1))
