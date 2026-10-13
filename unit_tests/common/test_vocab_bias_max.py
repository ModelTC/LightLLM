import pytest
import torch
from types import SimpleNamespace

from lightllm.common.basemodel.triton_kernel.local_vocab_topk import local_vocab_bias_max


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch,vocab", [(1, 31), (7, 513), (16, 129280), (64, 129280)])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_strided_markov_max_and_ties(batch, vocab, bias_dtype):
    torch.manual_seed(1013)
    storage = torch.randn(vocab, batch * 3, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(batch, vocab, device="cuda", dtype=bias_dtype)
    for step in range(3):
        base = storage[:, step::3].T
        reference = torch.max(base.float() + bias, dim=-1)
        values, indexes = local_vocab_bias_max(base, bias)
        torch.testing.assert_close(values, reference.values, rtol=0, atol=0)
        torch.testing.assert_close(indexes, reference.indices, rtol=0, atol=0)
    storage.zero_()
    bias.zero_()
    bias[:, 1] = bias[:, -1] = 8
    values, indexes = local_vocab_bias_max(storage[:, 0::3].T, bias)
    assert torch.all(values == 8)
    assert torch.all(indexes == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay_reads_new_inputs():
    storage = torch.zeros(1031, 48, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(16, 1031, device="cuda", dtype=torch.bfloat16)
    base = storage[:, 1::3].T
    local_vocab_bias_max(base, bias)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, indexes = local_vocab_bias_max(base, bias)
    for winner in [1029, 2, 511]:
        bias.zero_()
        bias[:, winner] = 4
        graph.replay()
        assert torch.all(values == 4)
        assert torch.all(indexes == winner)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sequential_markov_tokens_match_original():
    from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer

    torch.manual_seed(1013)
    batch, vocab, rank = 7, 1031, 8
    base = torch.randn(vocab, batch * 3, device="cuda", dtype=torch.bfloat16)
    embedding = torch.randn(vocab, rank, device="cuda", dtype=torch.bfloat16)
    projection = torch.randn(vocab, rank, device="cuda", dtype=torch.bfloat16)
    anchors = torch.arange(batch, device="cuda")
    infer = Qwen3DSparkPostLayerInfer.__new__(Qwen3DSparkPostLayerInfer)
    infer.block_size_ = 3
    infer.tp_world_size_ = 1
    infer.markov_head_type_ = "vanilla"
    infer.alloc_tensor = torch.empty
    weights = SimpleNamespace(
        markov_w1_weight_=SimpleNamespace(weight=embedding),
        markov_w2_weight_=SimpleNamespace(weight=projection, tp_vocab_start_id=0),
    )
    expected, previous = [], anchors
    for step in range(3):
        bias = torch.mm(torch.nn.functional.embedding(previous, embedding), projection.T)
        previous = torch.max(base[:, step::3].T.float() + bias, dim=-1).indices
        expected.append(previous)
    actual = infer._sample_markov(base, torch.empty(batch, 3, 16, device="cuda"), None, anchors, weights)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1), rtol=0, atol=0)
