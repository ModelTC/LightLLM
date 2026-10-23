from types import SimpleNamespace

import pytest
import torch

from lightllm.models.glm5_next.indexer import Glm5NextNsaInfer
from lightllm.models.glm5_next.triton_kernel.index_bf16 import bf16_mqa_logits
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("heads", [4, 32])
@pytest.mark.parametrize("max_pools", [640, 8192, 262144])
def test_bf16_scores_match_fp32_and_replay_changing_bounds(heads, max_pools):
    torch.manual_seed(53)
    q = torch.randn(6, heads, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(2 * max_pools, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(6, heads, device="cuda") * heads**-0.5
    starts = torch.tensor([0, max_pools, 0, max_pools, 0, max_pools], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 1, 511, 512, 639, max_pools], device="cuda", dtype=torch.int32)
    ends = starts + lengths
    bf16_mqa_logits(q, keys, weights, starts, ends, max_pools)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = bf16_mqa_logits(q, keys, weights, starts, ends, max_pools)
    # Explicit FP64 oracle avoids any global TF32 setting affecting the check.
    for shift in range(2):
        lengths.copy_(lengths.roll(shift))
        ends.copy_(starts + lengths)
        graph.replay()
        for row, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
            reference = (q[row].double() @ keys[start : start + length].double().T * 128**-0.5).relu()
            reference = (reference * weights[row, :, None].double()).sum(0).float()
            torch.testing.assert_close(actual[row, :length], reference, atol=3e-6, rtol=2e-5)
            assert torch.isneginf(actual[row, length:]).all()


@pytest.mark.parametrize("is_prefill", [True, False])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_bf16_indexer_forward_causal_pools_mtp_and_hold(monkeypatch, is_prefill, cuda_graph):
    monkeypatch.setenv("LIGHTLLM_GLM5_INDEXER_BF16", "1")
    monkeypatch.setattr("lightllm.models.glm5_next.indexer.HAS_VLLM", False)

    def unexpected_fp8(*args, **kwargs):
        raise AssertionError("BF16 indexer must not quantize Q or call the FP8 scorer")

    monkeypatch.setattr("lightllm.models.glm5_next.indexer.hadamard_transform_quant_fp8", unexpected_fp8)
    torch.manual_seed(57)
    heads, dim = 4, 128
    prefixes, counts, requests = [2300, 2049, 7], [5, 3, 3], [2, 0, 1]
    lengths = [prefix + count for prefix, count in zip(prefixes, counts)]
    table = torch.randperm(4 * 2500, device="cuda", dtype=torch.int32).view(4, 2500)
    storage = torch.zeros(4 * 2500, 1, 640, device="cuda", dtype=torch.bfloat16)
    packed = storage[:, :, 512:]
    tail = torch.zeros(4, 8, 256, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(4, dim, device="cuda")
    tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    sources = [torch.randn(length, 256, device="cuda", dtype=torch.bfloat16) for length in lengths]
    for req, prefix, source in zip(requests, prefixes, sources):
        compress_pools(
            source[:prefix],
            tail,
            packed,
            ape,
            tensor(range(1, prefix + 1)),
            tensor([0] * prefix),
            table[req],
            tensor([req]),
            tensor([0, prefix]),
            tensor([prefix]),
            prefix,
        )
    hidden = torch.cat([source[prefix:] for source, prefix in zip(sources, prefixes)])
    q = torch.randn(hidden.shape[0], heads, dim, device="cuda", dtype=torch.bfloat16)
    visible, req_rows, mtp_rows, ragged, starts, cu = [], [], [], [], [], [0]
    for req, prefix, count, length in zip(requests, prefixes, counts, lengths):
        visible.extend(range(prefix + 1, length + 1))
        req_rows.extend([req] * count)
        mtp_rows.extend(range(count))
        starts.extend([len(ragged)] * count)
        ragged.extend(table[req, :length].tolist())
        cu.append(cu[-1] + count)
    if not is_prefill:
        # Each verify position is an independent row with its own visible pool
        # length and ragged start; graph padding uses the HOLD request slot.
        hidden = torch.cat((hidden, torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)))
        q = torch.cat((q, torch.zeros(1, heads, dim, device="cuda", dtype=torch.bfloat16)))
        req_rows.append(3)
        mtp_rows.append(0)
        visible.append(0)
        starts.append(len(ragged))
    infer = SimpleNamespace(
        is_prefill=is_prefill,
        max_kv_seq_len=max(lengths),
        max_q_seq_len=max(counts) if is_prefill else 1,
        b_req_idx=tensor(requests if is_prefill else req_rows),
        b_seq_len=tensor(lengths if is_prefill else visible),
        b1_cu_q_seq_len=tensor(cu if is_prefill else range(hidden.shape[0] + 1)),
        b_mtp_index=tensor(mtp_rows),
        mem_manager=SimpleNamespace(get_indexer_k_buffer=lambda _: packed),
        req_manager=SimpleNamespace(req_to_token_indexs=table, get_indexer_tail_buffer=lambda _: tail),
    )
    att = SimpleNamespace(lengths=tensor(visible), ks=tensor(starts), ragged_mem_index=tensor(ragged))
    weight = SimpleNamespace(
        wk_proj_=SimpleNamespace(mm=lambda x: x[:, :dim].contiguous()),
        k_norm_=lambda x, **kwargs: x,
        index_kpool_compress_gate=SimpleNamespace(mm=lambda x: x[:, dim:].contiguous()),
        index_kpool_compress_ape=SimpleNamespace(weight=ape),
        wq_b_proj_=SimpleNamespace(mm=lambda x: x),
        weights_proj_=SimpleNamespace(mm=lambda x: x[:, :heads].contiguous() * 0.1),
    )
    indexer = Glm5NextNsaInfer(3, {"index_topk": 2048, "index_n_heads": heads, "index_head_dim": dim}, 8)
    run = lambda: indexer._get_indices(hidden, q.flatten(1), infer, att, weight)
    before = tail.clone()
    run()
    tail.copy_(before)
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, relative = run()
        tail.copy_(before)
        graph.replay()
    else:
        actual, relative = run()
    row = 0
    for req, prefix, count, source in zip(requests, prefixes, counts, sources):
        for step in range(count):
            length = prefix + step + 1
            pools = length // 4
            values = source[: pools * 4].view(pools, 4, 256)
            pooled = (values[:, :, :dim].float() * (values[:, :, dim:].float() + ape).softmax(1)).sum(1).bfloat16()
            scores = (q[row].double() @ pooled.double().T * dim**-0.5).relu()
            weights = hidden[row, :heads].float() * 0.1 * heads**-0.5
            scores = (scores * weights[:, None].double()).sum(0)
            selected = relative[row][relative[row] >= 0]
            # Dense requests include all tokens; sparse requests include full
            # selected pools followed by the uncompressed partial tail.
            if length <= 2048:
                assert sorted(selected.tolist()) == list(range(length))
            else:
                groups = selected[:2048:4] // 4
                assert groups.unique().numel() == 512
                expected = scores.topk(512).values.sort().values
                got = scores[groups.long()].sort().values
                torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)
                assert selected[2048:].tolist() == list(range(pools * 4, length))
            assert torch.equal(actual[row, : selected.numel()], table[req, selected.long()])
            row += 1
    if not is_prefill:
        assert (actual[-1] == -1).all()
        assert not tail[3].any()
