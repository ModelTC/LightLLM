import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.nsa.glm5_next import (
    Glm5NextSparseAttBackend,
    Glm5NextSparseDecodeState,
)
from lightllm.common.basemodel.attention.nsa.fp8_glm5_next import (
    Fp8Glm5NextSparseAttBackend,
    Fp8Glm5NextSparseDecodeState,
)
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_write import write_fp8_e8m0_packed_kv
from lightllm.common.basemodel.attention.nsa import fp8_glm5_next
from lightllm.models.glm5_next.triton_kernel.kpool import expand_topk, prefill_nsa_params


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def tensor(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


@pytest.mark.parametrize("counts", [(3, 0, 1, 2), (257, 0, 1, 513), (1,) * 129])
def test_prefill_metadata_reuses_existing_prefixes_during_graph_replay(counts):
    batch, queries = len(counts), sum(counts)
    positions = torch.empty(queries, device="cuda", dtype=torch.int32)
    req = torch.empty(batch, device="cuda", dtype=torch.int32)
    cu_q, cu_kv = torch.empty(batch + 1, device="cuda", dtype=torch.int32), torch.empty(
        batch + 1, device="cuda", dtype=torch.int32
    )

    def update(counts, phase):
        q_prefix, kv_prefix, visible, starts, requests = [0], [0], [], [], []
        req_values = [(batch - i + phase) % (batch + 1) for i in range(batch)]
        for i, count in enumerate(counts):
            prefix = 1000000 + i * 7 + phase
            visible.extend(range(prefix + 1, prefix + count + 1))
            starts.extend([kv_prefix[-1]] * count)
            requests.extend([req_values[i]] * count)
            q_prefix.append(q_prefix[-1] + count)
            kv_prefix.append(kv_prefix[-1] + prefix + count)
        positions.copy_(tensor(visible) - 1)
        req.copy_(tensor(req_values))
        cu_q.copy_(tensor(q_prefix))
        cu_kv.copy_(tensor(kv_prefix))
        return tensor(visible), tensor(requests), tensor(starts)

    expected = update(counts, 0)
    actual = prefill_nsa_params(positions, req, cu_q, cu_kv)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = prefill_nsa_params(positions, req, cu_q, cu_kv)
    expected = update(counts[::-1], 11)
    graph.replay()
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dense", [False, True])
def test_topk_direct_request_lookup_preserves_mtp_hold_and_strided_tables(dense):
    table = torch.arange(8 * 70, device="cuda", dtype=torch.int32).reshape(8, 70)[::2, ::2]
    req = tensor([2, 2, 0, 3, 3])
    lengths = tensor([5, 7, 9, 2, 2] if dense else [17, 18, 9, 2, 2])
    groups = tensor([[3, 1], [1, 0], [0, 1], [0, 0], [0, 0]])

    def check(actual):
        mem, relative = actual
        expected_mem, expected_relative = torch.full_like(mem, -1), torch.full_like(relative, -1)
        for i, (request, length) in enumerate(zip(req.tolist(), lengths.tolist())):
            tokens = (
                list(range(length))
                if dense
                else [group * 4 + j for group in groups[i].tolist()[: min(length // 4, 2)] for j in range(4)]
                + list(range(length // 4 * 4, length))
            )
            expected_relative[i, : len(tokens)] = tensor(tokens)
            expected_mem[i, : len(tokens)] = table[request, tensor(tokens).long()]
        torch.testing.assert_close(mem, expected_mem, rtol=0, atol=0)
        torch.testing.assert_close(relative, expected_relative, rtol=0, atol=0)

    check(expand_topk(groups, lengths, req, table, 8, dense=dense))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = expand_topk(groups, lengths, req, table, 8, dense=dense)
    req.copy_(tensor([0, 0, 2, 3, 3]))
    table.add_(10000)
    lengths[:2].add_(1)
    graph.replay()
    check(actual)


@pytest.mark.parametrize("fp8", [False, True])
def test_decode_aliases_infer_state_without_ragged_allocation(monkeypatch, fp8):
    import lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("GLM must not generate or copy a full ragged index")

    monkeypatch.setattr(legacy, "gen_nsa_ks_ke", forbidden)
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))
    model = type(
        "Model",
        (),
        {
            "config": {"index_kpool": 4, "index_topk": 2048},
            "batch_max_tokens": 4,
            "max_req_num": 1,
            "max_seq_length": 32,
            "att_state_role": "decode_test",
            "args": SimpleNamespace(enable_prefill_microbatch_overlap=False, run_mode="decode"),
        },
    )()
    backend = (Fp8Glm5NextSparseAttBackend if fp8 else Glm5NextSparseAttBackend)(model)
    assert not hasattr(backend, "ragged_mem_buffers")
    infer = InferStateInfo()
    infer.b_req_idx = tensor([2, 2, 0, 3, 3])
    infer.b_seq_len = tensor([999999, 1000000, 65536, 2, 2])
    cls = Fp8Glm5NextSparseDecodeState if fp8 else Glm5NextSparseDecodeState
    state = cls(backend=backend, _infer_state_weak_ref=weakref.ref(infer))
    state.init_state()
    assert state.q_kv_seq_len is infer.b_seq_len
    assert state.q_req_idx is infer.b_req_idx
    assert not hasattr(state, "ragged_mem_index") and not hasattr(state, "ks")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.init_state()
    infer.b_seq_len[:3].sub_(100)
    infer.b_req_idx[:3].copy_(tensor([0, 0, 1]))
    graph.replay()
    assert state.q_kv_seq_len is infer.b_seq_len and state.q_req_idx is infer.b_req_idx


@pytest.mark.parametrize(
    "queries,heads,total_kv,chunk_size",
    [
        (1, 64, 65536, 2097152),
        (2, 64, 65536, 2097152),
        (3, 8, 65536, 2097152),
        (9, 64, 512, 2097152),
        (9, 64, 131073, 2097152),
        (4, 16, 40, 3),
        (4, 64, 40, 7),
    ],
)
def test_fp8_prefill_reads_request_table_without_ragged_buffer(monkeypatch, queries, heads, total_kv, chunk_size):
    monkeypatch.setattr(fp8_glm5_next, "KV_GATHER_CHUNK_SIZE", chunk_size)

    torch.manual_seed(1685)
    batch = min(queries, 2)
    counts = [queries] if batch == 1 else [queries // 2, queries - queries // 2]
    seqs = [total_kv] if batch == 1 else [total_kv // 2, total_kv - total_kv // 2]
    req = tensor([2, 0][:batch])
    table = torch.full((4, 2 * (max(seqs) + 7)), -1, device="cuda", dtype=torch.int32)[:, ::2]
    physical = torch.randperm(total_kv + 32, device="cuda", dtype=torch.int32)[:total_kv]
    offset = 0
    for request, length in zip(req.tolist(), seqs):
        table[request, :length] = physical[offset : offset + length]
        offset += length
    q = torch.randn(queries, heads, 512, device="cuda", dtype=torch.bfloat16)
    source = torch.randn(total_kv + 32, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed = torch.zeros(total_kv + 32, 1, 720, device="cuda", dtype=torch.uint8)
    write_fp8_e8m0_packed_kv(source, torch.arange(total_kv + 32, device="cuda", dtype=torch.int32), packed)
    prefill_kv = torch.randn(queries, 1, 512, device="cuda", dtype=torch.bfloat16)
    visible, query_req, current_positions, q_prefix, kv_prefix = [], [], [], [0], [0]
    for request, length, count in zip(req.tolist(), seqs, counts):
        visible.extend(range(length - count + 1, length + 1))
        query_req.extend([request] * count)
        current_positions.extend(range(length - count, length))
        q_prefix.append(q_prefix[-1] + count)
        kv_prefix.append(kv_prefix[-1] + length)
    query_req, current_positions = tensor(query_req), tensor(current_positions)
    mem_index = table[query_req.long(), current_positions.long()]
    topk = torch.full((queries, 2176), -1, device="cuda", dtype=torch.int32)
    for i, length in enumerate(visible):
        topk[i, :7] = tensor([0, length - 1, 0, 1, length // 2, length - 1, -1])
    topk_mem = torch.empty_like(topk)

    def refresh_physical():
        topk_mem.copy_(table[query_req.long()[:, None], topk.clamp_min(0).long()])
        topk_mem.masked_fill_(topk < 0, -1)
        mem_index.copy_(table[query_req.long(), current_positions.long()])

    refresh_physical()
    infer = InferStateInfo()
    infer.mem_index, infer.b_req_idx = mem_index, req
    infer.position_ids = tensor(visible) - 1
    infer.b1_cu_q_seq_len, infer.b1_cu_kv_seq_len = tensor(q_prefix), tensor(kv_prefix)
    infer.total_token_num = total_kv
    infer.req_manager = SimpleNamespace(req_to_token_indexs=table)
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))
    model = type(
        "Model",
        (),
        {
            "batch_max_tokens": queries,
            "prefill_graph": None,
            "max_req_num": batch,
            "max_seq_length": max(seqs),
            "att_state_role": "indices_test",
            "config": {"index_topk": 2048},
            "args": SimpleNamespace(enable_prefill_microbatch_overlap=False, run_mode="normal"),
        },
    )()
    backend = Fp8Glm5NextSparseAttBackend(model)
    state = backend.create_att_prefill_state(infer)
    state.init_state()
    assert not hasattr(state, "ragged_mem_index") and not hasattr(backend, "ragged_mem_buffers")
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict={
            "topk_indices": topk,
            "topk_mem_indices": topk_mem,
            "prefill_cache_kv": prefill_kv,
            "softmax_scale": 512 ** -0.5,
            "kv_lora_rank": 512,
        },
    )

    def run():
        return state.prefill_att(q, packed[..., :584], None, control)

    def check(output):
        scales = packed[..., 576:583].view(torch.float8_e8m0fnu).float().reshape(-1, 7, 1)
        values = packed[..., :448].view(torch.float8_e4m3fn).float().reshape(-1, 7, 64) * scales
        kv = torch.cat((values.reshape(-1, 1, 448).bfloat16(), packed[..., 448:576].view(torch.bfloat16)), -1)
        kv[mem_index.long()] = prefill_kv
        expected = []
        for i in range(queries):
            keys = kv[topk_mem[i][topk_mem[i] >= 0].long(), 0].float()
            expected.append((q[i].float() @ keys.T * 512 ** -0.5).softmax(-1) @ keys)
        torch.testing.assert_close(output.float(), torch.stack(expected), atol=0.012, rtol=0.015)

    check(run())
    workspace = backend._prefill_workspaces[0]
    if workspace is not None:
        assert set(workspace) == {"gathered_kv", "chunk_topk_indices", "chunk_cache_seqlens", "cu_seqlens_q"}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    graph.replay()
    check(output)
    q.normal_()
    prefill_kv.normal_()
    for request, length in zip(req.tolist(), seqs):
        table[request, :length].copy_(table[request, :length].roll(1))
    topk[0].fill_(-1)
    refresh_physical()
    graph.replay()
    check(output)
