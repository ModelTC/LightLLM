from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.glm5_next_mtp.layer_infer import pre_layer_infer as mtp
from lightllm.models.glm5_next_mtp.triton_kernel.fused_eh_norm import fused_mtp_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(
    embeddings: torch.Tensor,
    previous_hidden: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    rmsnorm_forward(x=previous_hidden, weight=weights[0], eps=eps, out=previous_hidden)
    rmsnorm_forward(x=embeddings, weight=weights[1], eps=eps, out=embeddings)
    rmsnorm_forward(x=previous_hidden, weight=weights[2], eps=eps, out=previous_hidden)
    return torch.cat(tensors=(embeddings, previous_hidden), dim=-1)


@pytest.mark.parametrize("rows", [1, 4, 16, 64, 512, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale", [1e-4, 1.0, 100.0])
def test_norm_rounding_and_inplace_updates(rows: int, dtype: torch.dtype, scale: float) -> None:
    torch.manual_seed(seed=5514)
    hidden = 4096
    e = (torch.randn(size=(rows, hidden + 16), dtype=dtype, device="cuda") * scale)[:, :hidden]
    p = (torch.randn(size=(rows, hidden + 16), dtype=dtype, device="cuda") * scale)[:, :hidden]
    er, pr = e.clone(), p.clone()
    weights = tuple(torch.randn(size=(hidden,), dtype=dtype, device="cuda") * 0.2 + 1 for _ in range(3))
    expected = reference(embeddings=er, previous_hidden=pr, weights=weights, eps=1e-5)
    actual = fused_mtp_norm(
        embeddings=e,
        previous_hidden=p,
        main_weight=weights[0],
        embedding_weight=weights[1],
        hidden_weight=weights[2],
        eps=1e-5,
    )
    torch.testing.assert_close(actual=actual, expected=expected, atol=0, rtol=0)
    torch.testing.assert_close(actual=e, expected=er, atol=0, rtol=0)
    torch.testing.assert_close(actual=p, expected=pr, atol=0, rtol=0)


def test_graph_replay_changes_both_inputs_and_preserves_strided_storage() -> None:
    torch.manual_seed(seed=5515)
    rows, hidden = 16, 4096
    e_storage = torch.randn(size=(rows, hidden * 2), device="cuda", dtype=torch.bfloat16)
    p_storage = torch.randn(size=(rows, hidden * 2), device="cuda", dtype=torch.bfloat16)
    e, p = e_storage[:, ::2], p_storage[:, ::2]
    weights = tuple(torch.randn(size=(hidden,), device="cuda", dtype=e.dtype) * 0.2 + 1 for _ in range(3))
    fused_mtp_norm(
        embeddings=e,
        previous_hidden=p,
        main_weight=weights[0],
        embedding_weight=weights[1],
        hidden_weight=weights[2],
        eps=1e-5,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph=graph):
        actual = fused_mtp_norm(
            embeddings=e,
            previous_hidden=p,
            main_weight=weights[0],
            embedding_weight=weights[1],
            hidden_weight=weights[2],
            eps=1e-5,
        )
    for scale in (0.01, 10.0):
        e_storage.normal_().mul_(scale)
        p_storage.normal_().mul_(scale)
        untouched_e, untouched_p = e_storage[:, 1::2].clone(), p_storage[:, 1::2].clone()
        er, pr = e.clone(), p.clone()
        expected = reference(embeddings=er, previous_hidden=pr, weights=weights, eps=1e-5)
        graph.replay()
        torch.testing.assert_close(actual=actual, expected=expected, atol=0, rtol=0)
        torch.testing.assert_close(actual=e, expected=er, atol=0, rtol=0)
        torch.testing.assert_close(actual=p, expected=pr, atol=0, rtol=0)
        assert torch.equal(input=e_storage[:, 1::2], other=untouched_e)
        assert torch.equal(input=p_storage[:, 1::2], other=untouched_p)


@pytest.mark.parametrize("hidden", [4096, 1024])
def test_model_integration(hidden: int) -> None:
    torch.manual_seed(seed=5516)
    e = torch.randn(size=(7, hidden), dtype=torch.bfloat16, device="cuda")
    p = torch.randn_like(input=e)
    er, pr = e.clone(), p.clone()
    weights = tuple(torch.randn(size=(hidden,), dtype=e.dtype, device="cuda") * 0.2 + 1 for _ in range(3))

    class Norm:
        def __init__(self, weight: torch.Tensor) -> None:
            self.weight = weight

        def __call__(self, input: torch.Tensor, eps: float, out: torch.Tensor) -> torch.Tensor:
            return rmsnorm_forward(x=input, weight=self.weight, eps=eps, out=out)

    class Projection:
        def __init__(self) -> None:
            self.weight = torch.randn(size=(2 * hidden, 128), dtype=e.dtype, device=e.device) * 0.1

        def mm(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return torch.mm(input=input_tensor, mat2=self.weight)

    layer = mtp.Glm5NextMTPPreLayerInfer.__new__(mtp.Glm5NextMTPPreLayerInfer)
    layer.eps_ = 1e-5
    layer_weights = SimpleNamespace(
        main_norm_weight_=Norm(weight=weights[0]),
        enorm_weight_=Norm(weight=weights[1]),
        hnorm_weight_=Norm(weight=weights[2]),
        eh_proj_weight_=Projection(),
    )
    expected = layer_weights.eh_proj_weight_.mm(
        input_tensor=reference(embeddings=er, previous_hidden=pr, weights=weights, eps=1e-5)
    )
    actual = layer._fuse_hidden(
        input_embeddings=e, infer_state=SimpleNamespace(mtp_draft_input_hiddens=p), layer_weight=layer_weights
    )
    torch.testing.assert_close(actual=actual, expected=expected, atol=0, rtol=0)
    torch.testing.assert_close(actual=e, expected=er, atol=0, rtol=0)
    torch.testing.assert_close(actual=p, expected=pr, atol=0, rtol=0)
