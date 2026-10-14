from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import PostLayerOutput
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
from lightllm.common.basemodel.triton_kernel.local_vocab_topk import local_vocab_topk
from lightllm.common.basemodel.triton_kernel.pack_vocab_parallel_topk import (
    pack_vocab_parallel_topk,
    unpack_vocab_parallel_topk,
)
from lightllm.models.llama.layer_infer import post_layer_infer as llama_post
from lightllm.models.gemma4.layer_infer.post_layer_infer import Gemma4PostLayerInfer


def _mock_vocab_parallel_pack(monkeypatch) -> None:
    def pack(values, token_ids, vocab_start, packed):
        candidate_count = values.shape[1]
        packed[:, :candidate_count] = values.float()
        packed[:, candidate_count:] = (token_ids + vocab_start).float()

    def unpack(packed, values, token_ids, candidate_count, world_size):
        token_num = values.shape[0]
        for rank in range(world_size):
            source = packed[rank * token_num : (rank + 1) * token_num]
            candidate_slice = slice(rank * candidate_count, (rank + 1) * candidate_count)
            values[:, candidate_slice] = source[:, :candidate_count]
            token_ids[:, candidate_slice] = source[:, candidate_count:].long()

    monkeypatch.setattr(llama_post, "pack_vocab_parallel_topk", pack)
    monkeypatch.setattr(llama_post, "unpack_vocab_parallel_topk", unpack)


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_output_head_returns_candidates_without_mutating_state(monkeypatch, draft, enabled) -> None:
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    head.tp_world_size_ = 1
    head.alloc_tensor = lambda shape, dtype, **kwargs: torch.empty(shape, dtype=dtype)
    head._norm = lambda hidden, state, weight: hidden
    dense = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    class Weight:
        vocab_size = 4
        tp_vocab_start_id = 0

        def __call__(self, **kwargs) -> torch.Tensor:
            return dense

    values, ids = dense.T[:, :2], torch.tensor([[0, 1]] * 3)
    captured_top_k = []

    def candidates(*args):
        captured_top_k.append(args[-1])
        return values, ids

    head._vocab_parallel_topk = candidates
    args = SimpleNamespace(target_vocab_topk_sampling=None, draft_vocab_topk_sampling=None)
    setattr(args, "draft_vocab_topk_sampling" if draft else "target_vocab_topk_sampling", 16 if enabled else None)
    monkeypatch.setattr(llama_post, "get_env_start_args", lambda: args)
    state = SimpleNamespace(is_draft_model=draft, dist_group=None)
    weight = SimpleNamespace(lm_head_weight_=Weight())
    method = head._draft_lm_head_and_gather if draft else head._target_lm_head_and_gather
    output = method(torch.ones(3, 2), 3, weight, state)
    assert not hasattr(state, "logits_token_ids")
    if enabled and not draft:
        expected_logits = torch.full_like(dense.T, llama_post._MASKED_LOGIT_VALUE)
        expected_logits.scatter_(1, ids, values)
    else:
        expected_logits = values if enabled else dense.T
    torch.testing.assert_close(output.logits, expected_logits)
    assert output.logits_token_ids is (ids if enabled and draft else None)
    assert captured_top_k == ([16] if enabled else [])
    prompt = head._lm_head_and_gather(torch.ones(3, 2), 3, weight, state)
    torch.testing.assert_close(prompt.logits, dense.T)
    assert prompt.logits_token_ids is None


def test_vocab_parallel_topk_returns_candidates_on_single_rank(monkeypatch) -> None:
    _mock_vocab_parallel_pack(monkeypatch)
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    head.tp_world_size_ = 1
    head.alloc_tensor = lambda shape, dtype, **kwargs: torch.empty(shape, dtype=dtype)
    local_logits = torch.tensor(
        [
            [1.0, 8.0],
            [7.0, 2.0],
            [3.0, 6.0],
            [5.0, 4.0],
        ]
    )
    head._project_local_logits = lambda *args: local_logits
    weight = SimpleNamespace(lm_head_weight_=SimpleNamespace(vocab_size=4, tp_vocab_start_id=0))

    logits, token_ids = head._vocab_parallel_topk(None, 2, weight, SimpleNamespace(dist_group=None), 2)

    expected_logits, expected_token_ids = torch.topk(local_logits, k=2, dim=0, sorted=False)
    assert logits.dtype == torch.float32 and token_ids.dtype == torch.int64
    assert token_ids.is_contiguous()
    torch.testing.assert_close(logits, expected_logits.permute(1, 0))
    torch.testing.assert_close(token_ids, expected_token_ids.permute(1, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("vocab_size, token_num, top_k", [(257, 15, 512), (257, 65, 64), (248320, 64, 64)])
def test_single_rank_vocab_topk_graph_preserves_output_contract(vocab_size, token_num, top_k) -> None:
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    head.tp_world_size_ = 1
    local_logits = torch.randn((vocab_size, token_num), dtype=torch.bfloat16, device="cuda")
    head._project_local_logits = lambda *args: local_logits
    weight = SimpleNamespace(lm_head_weight_=SimpleNamespace(vocab_size=vocab_size, tp_vocab_start_id=0))
    state = SimpleNamespace(dist_group=None)

    for _ in range(3):
        head._vocab_parallel_topk(None, token_num, weight, state, top_k)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, token_ids = head._vocab_parallel_topk(None, token_num, weight, state, top_k)

    local_logits.normal_()
    graph.replay()
    torch.cuda.synchronize()
    candidate_count = min(top_k, vocab_size)
    expected_values = torch.topk(local_logits.T, k=candidate_count, dim=1, sorted=True).values.float()
    assert values.shape == token_ids.shape == (token_num, candidate_count)
    assert values.dtype == torch.float32 and token_ids.dtype == torch.int64
    assert token_ids.is_contiguous()
    assert torch.all((token_ids >= 0) & (token_ids < vocab_size))
    assert torch.all(token_ids.sort(dim=1).values.diff(dim=1) > 0)
    torch.testing.assert_close(values, local_logits.T.gather(1, token_ids).float(), rtol=0, atol=0)
    torch.testing.assert_close(values.sort(dim=1, descending=True).values, expected_values, rtol=0, atol=0)


def test_vocab_parallel_topk_requires_enough_local_vocabulary() -> None:
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    head.tp_world_size_ = 2
    head._project_local_logits = lambda *args: torch.ones(2, 1)
    weight = SimpleNamespace(lm_head_weight_=SimpleNamespace(vocab_size=6, tp_vocab_start_id=0))

    with pytest.raises(AssertionError, match="local vocabulary size 2 must be at least the candidate count 3"):
        head._vocab_parallel_topk(None, 1, weight, SimpleNamespace(dist_group=None), 3)


def test_vocab_parallel_topk_merges_rank_candidates(monkeypatch) -> None:
    _mock_vocab_parallel_pack(monkeypatch)
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    head.tp_world_size_ = 2
    head.alloc_tensor = lambda shape, dtype, **kwargs: torch.empty(shape, dtype=dtype)
    local_logits = torch.tensor(
        [
            [1.0, 6.0],
            [5.0, 2.0],
            [3.0, 4.0],
        ]
    )
    head._project_local_logits = lambda *args: local_logits
    weight = SimpleNamespace(lm_head_weight_=SimpleNamespace(vocab_size=6, tp_vocab_start_id=0))
    remote_values = torch.tensor([[10.0, 9.0], [8.0, 7.0]])
    remote_token_ids = torch.tensor([[3, 4], [5, 4]])
    gather_count = 0

    def all_gather(output, local, **kwargs):
        nonlocal gather_count
        gather_count += 1
        output[:2] = local
        output[2:, :2] = remote_values
        output[2:, 2:] = remote_token_ids.float()

    monkeypatch.setattr(llama_post, "all_gather_into_tensor", all_gather)

    logits, token_ids = head._vocab_parallel_topk(None, 2, weight, SimpleNamespace(dist_group=None), 2)

    expected_local_values, expected_local_token_ids = torch.topk(local_logits, k=2, dim=0, sorted=False)
    expected_logits = torch.cat((expected_local_values.T, remote_values), dim=1)
    expected_token_ids = torch.cat((expected_local_token_ids.T, remote_token_ids), dim=1)
    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(token_ids, expected_token_ids)
    assert gather_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("transposed", [False, True])
def test_vocab_parallel_topk_pack_round_trip_preserves_token_id_bits(transposed) -> None:
    device = torch.device("cuda")
    values0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float16, device=device)
    values1 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float16, device=device)
    token_ids0 = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64, device=device)
    token_ids1 = torch.tensor([[6, 7], [8, 9], [10, 11]], dtype=torch.int64, device=device)
    if transposed:
        # 回退路径返回 [B, K] 转置视图；用 B != K 检查维度和 stride 是否正确。
        values0, values1 = values0.T.contiguous().T, values1.T.contiguous().T
        token_ids0, token_ids1 = token_ids0.T.contiguous().T, token_ids1.T.contiguous().T
    vocab_starts = (20_000_001, 30_000_001)
    packed0 = torch.empty((3, 4), dtype=torch.float32, device=device)
    packed1 = torch.empty_like(packed0)

    pack_vocab_parallel_topk(values0, token_ids0, vocab_starts[0], packed0)
    pack_vocab_parallel_topk(values1, token_ids1, vocab_starts[1], packed1)
    gathered = torch.cat((packed0, packed1), dim=0)
    values = torch.empty((3, 4), dtype=torch.float32, device=device)
    token_ids = torch.empty((3, 4), dtype=torch.int64, device=device)
    unpack_vocab_parallel_topk(gathered, values, token_ids, candidate_count=2, world_size=2)

    expected_values = torch.cat((values0.float(), values1.float()), dim=1)
    expected_token_ids = torch.cat((token_ids0 + vocab_starts[0], token_ids1 + vocab_starts[1]), dim=1)
    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(token_ids, expected_token_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(129, 15), (129, 16), (129, 17), (62080, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_local_vocab_topk_preserves_values_and_token_ids(shape, dtype) -> None:
    logits = torch.randn(shape, dtype=dtype, device="cuda")
    values, token_ids = local_vocab_topk(logits, top_k=64)
    expected_values = torch.topk(logits, k=64, dim=0, sorted=True).values.T

    assert values.shape == token_ids.shape == (shape[1], 64)
    assert values.dtype == dtype and token_ids.dtype == torch.int64
    assert torch.all((token_ids >= 0) & (token_ids < shape[0]))
    assert torch.all(token_ids.sort(dim=1).values.diff(dim=1) > 0)
    torch.testing.assert_close(values, logits.T.gather(1, token_ids), rtol=0, atol=0)
    # Equal-valued boundary candidates can have different valid IDs/order.
    torch.testing.assert_close(values.sort(dim=1, descending=True).values, expected_values, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_local_vocab_topk_accepts_noncontiguous_logits() -> None:
    logits = torch.randn((65, 129), dtype=torch.bfloat16, device="cuda").T
    values, token_ids = local_vocab_topk(logits, top_k=8)
    expected_values = torch.topk(logits, k=8, dim=0, sorted=True).values.T
    torch.testing.assert_close(values, logits.T.gather(1, token_ids), rtol=0, atol=0)
    torch.testing.assert_close(values.sort(dim=1, descending=True).values, expected_values, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("top_k", [8, 64, 512])
def test_local_vocab_topk_graph_replay_reads_updated_logits(top_k) -> None:
    logits = torch.randn((1025, 65), dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        local_vocab_topk(logits, top_k)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, token_ids = local_vocab_topk(logits, top_k)

    logits.normal_()
    logits[:, 0] = -float("inf")
    logits[:, 1] = float("inf")
    logits[:, 2] = float("nan")
    logits[:, 3] = 0
    logits[::2, 3] = -0.0
    logits[:20, 4] = float("nan")
    logits[:, 5] = torch.finfo(torch.bfloat16).max
    graph.replay()
    torch.cuda.synchronize()

    expected_values = torch.topk(logits, k=top_k, dim=0, sorted=True).values.T
    assert torch.all((token_ids >= 0) & (token_ids < logits.shape[0]))
    assert torch.all(token_ids.sort(dim=1).values.diff(dim=1) > 0)
    torch.testing.assert_close(values, logits.T.gather(1, token_ids), rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(
        values.sort(dim=1, descending=True).values, expected_values, rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("draft", [False, True])
def test_prefill_prompt_logits_do_not_replace_candidate_output(draft) -> None:
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    state = SimpleNamespace(is_draft_model=draft, prompt_logics=torch.ones(5, 2))
    head._slice_get_last_input = lambda *args: (torch.ones(1, 2), 1)
    candidates = PostLayerOutput(
        torch.ones(1, 2),
        torch.tensor([[10, 20]]) if draft else None,
    )
    prompt = torch.ones(5, 32)
    if draft:
        head._draft_lm_head_and_gather = lambda *args: candidates
    else:
        head._target_lm_head_and_gather = lambda *args: candidates
    head._lm_head_and_gather = lambda *args: PostLayerOutput(prompt)
    assert head.token_forward(None, state, None) is candidates
    assert state.prompt_logics is prompt


def test_model_output_and_unpadding_preserve_candidate_mapping() -> None:
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    logits = torch.randn(3, 2)
    ids = torch.tensor([[10, 20], [30, 40], [50, 60]])
    state = SimpleNamespace(hidden_collector=NoopHiddenCollector(), prompt_logics=None)
    output = model._create_model_output(PostLayerOutput(logits, ids), state)
    unpadded = model._create_unpad_decode_model_output(output, 2)
    torch.testing.assert_close(unpadded.logits, logits[:2])
    torch.testing.assert_close(unpadded.logits_token_ids, ids[:2])


def test_gemma_softcap_preserves_candidate_ids(monkeypatch) -> None:
    ids = torch.tensor([[10, 20]])
    logits = torch.tensor([[1.0, 4.0]])
    monkeypatch.setattr(llama_post.LlamaPostLayerInfer, "token_forward", lambda *args: PostLayerOutput(logits, ids))
    head = Gemma4PostLayerInfer.__new__(Gemma4PostLayerInfer)
    head.final_logit_softcapping = 2.0
    state = SimpleNamespace(prompt_logics=None)
    output = head.token_forward(None, state, None)
    torch.testing.assert_close(output.logits, torch.tanh(logits / 2) * 2)
    assert output.logits_token_ids is ids


def test_overlap_keeps_each_microbatch_mapping() -> None:
    head = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    outputs = [PostLayerOutput(torch.ones(1, 2), torch.tensor([[10, 20]])), PostLayerOutput(torch.ones(1, 3))]
    head.token_forward = lambda hidden, state, layer_weight: outputs[state]
    output0, output1 = head.overlap_tpsp_token_forward(None, None, 0, 1, None)
    assert output0 is outputs[0]
    assert output1 is outputs[1]
