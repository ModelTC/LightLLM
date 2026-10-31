import weakref
from types import SimpleNamespace

import pytest
import torch
import triton

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.attention.nsa.glm5_next import (
    Glm5NextSparsePrefillState,
    Glm5NextSparseDecodeState,
)
from lightllm.common.basemodel.attention.nsa.fp8_glm5_next import (
    Fp8Glm5NextSparseAttBackend,
    Fp8Glm5NextSparsePrefillState,
    Fp8Glm5NextSparseDecodeState,
)
from lightllm.models.glm5_next.triton_kernel.fp8_sparse_decode import (
    can_use_fp8_sparse_direct_decode,
    fp8_sparse_direct_decode,
)
from lightllm.models.glm5_next.triton_kernel.pad_query_heads import _fused_strided_qpad_kernel
from lightllm.common.attention_state.full_att.fp8_e8m0_packed_mla_write import write_fp8_e8m0_packed_kv
from lightllm.models.glm5_next.triton_kernel import context_flashattention_nopad_fp8 as fp8_prefill
from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
    _build_prefill_row_table_kernel,
    _gather_prefill_kv_kernel,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_infer_state(**attributes):
    state = InferStateInfo()
    vars(state).update(attributes)
    return state


def _make_workspace_backend(batch_max_tokens):
    backend = object.__new__(Fp8Glm5NextSparseAttBackend)
    backend.model = SimpleNamespace(batch_max_tokens=batch_max_tokens, prefill_graph=None)
    backend._prefill_workspaces = [{"smallq": None, "generic": None} for _ in range(2)]
    return backend


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("kv_type", ["None", "fp8kv_dsa"])
def test_main_and_native_mtp_backend_selection(monkeypatch, draft, kv_type):
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel
    from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel

    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    model = object.__new__(Glm5NextMTPModel if draft else Glm5NextTpPartModel)
    model.args = SimpleNamespace(llm_kv_type=kv_type)
    model.config = {"index_topk": 2048, "index_kpool": 4}
    model.graph_max_batch_size = 4
    model.max_seq_length = 128
    model._init_att_backend()
    infer_state = InferStateInfo()
    prefill = model.prefill_att_backend.create_att_prefill_state(infer_state)
    decode = model.decode_att_backend.create_att_decode_state(infer_state)
    assert type(prefill) is (Fp8Glm5NextSparsePrefillState if kv_type == "fp8kv_dsa" else Glm5NextSparsePrefillState)
    assert type(decode) is (Fp8Glm5NextSparseDecodeState if kv_type == "fp8kv_dsa" else Glm5NextSparseDecodeState)
    infer_state.prefill_att_state = prefill
    infer_state.decode_att_state = decode
    infer_state.check_att_states()
    assert prefill.infer_state is infer_state
    assert decode.infer_state is infer_state
    owner_ref = weakref.ref(infer_state)
    del infer_state
    assert owner_ref() is None
    with pytest.raises(ReferenceError):
        prefill.infer_state
    with pytest.raises(ReferenceError):
        decode.infer_state


def _dequantize_kv_cache(packed_kv):
    values = packed_kv[..., :448].view(torch.float8_e4m3fn).float().reshape(-1, 7, 64)
    scales = packed_kv[..., 576:583].view(torch.float8_e8m0fnu).float().reshape(-1, 7, 1)
    tail = packed_kv[..., 448:576].view(torch.bfloat16)
    return torch.cat(((values * scales).reshape(-1, 1, 448).bfloat16(), tail), dim=-1)


def _torch_sparse_attention(q, kv, indices, softmax_scale=0.0625):
    outputs = []
    for row in range(q.shape[0]):
        keys = kv[indices[row][indices[row] >= 0].long(), 0].float()
        outputs.append((q[row].float() @ keys.T * softmax_scale).softmax(-1) @ keys)
    return torch.stack(outputs)


def _capture_cuda_graph(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    return graph, output


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("source_d_stride", [1, 2])
def test_fp8_scatter_quantization_and_gather_prefill(index_dtype, source_d_stride):
    torch.manual_seed(53)
    # Noncontiguous KV from a fused projection, and a strided packed cache.
    raw_source = torch.randn(11, 1, 512 * source_d_stride, device="cuda", dtype=torch.bfloat16)
    source = raw_source if source_d_stride == 1 else raw_source[..., ::2]
    source[0].zero_()
    source[1].mul_(1e-5)
    mem_index = torch.randperm(31, device="cuda")[:11].to(index_dtype)
    packed_kv = torch.full((31, 1, 720), 71, device="cuda", dtype=torch.uint8)
    write_fp8_e8m0_packed_kv(source, mem_index, packed_kv)
    groups = source[..., :448].float().reshape(11, 1, 7, 64)
    scales = torch.exp2(torch.ceil(torch.log2((groups.abs().amax(-1) / 448).clamp_min(1e-4))))
    values = (groups / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    actual_scales = packed_kv[..., 576:583].view(torch.float8_e8m0fnu).float()
    torch.testing.assert_close(actual_scales[mem_index], scales, rtol=0, atol=0)
    assert torch.equal(packed_kv[mem_index, :, :448], values.view(torch.uint8).reshape(11, 1, 448))
    assert torch.equal(packed_kv[..., 448:576].view(torch.bfloat16)[mem_index], source[..., 448:])
    assert not packed_kv[mem_index, :, 583].any()
    assert (packed_kv[..., 584:] == 71).all()
    unused = torch.ones(31, device="cuda", dtype=torch.bool)
    unused[mem_index] = False
    assert (packed_kv[unused] == 71).all()

    # The packer is graph-safe for both source layouts and mutable destination mapping.
    graph, _ = _capture_cuda_graph(lambda: write_fp8_e8m0_packed_kv(source, mem_index, packed_kv))
    graph.replay()
    assert torch.equal(packed_kv[mem_index, :, :448], values.view(torch.uint8).reshape(11, 1, 448))
    source.add_(0.125)
    mem_index.copy_(mem_index.roll(1))
    graph.replay()
    changed_groups = source[..., :448].float().reshape(11, 1, 7, 64)
    changed_scales = torch.exp2(torch.ceil(torch.log2((changed_groups.abs().amax(-1) / 448).clamp_min(1e-4))))
    changed_values = (changed_groups / changed_scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    assert torch.equal(packed_kv[mem_index, :, :448], changed_values.view(torch.uint8).reshape(11, 1, 448))

    indices = mem_index[[0, 2, 8, 2, 5, 1, 9, 3]]
    prefill_kv = source[8:].clone().add_(0.25)
    row_table = torch.full((31,), -1, device="cuda", dtype=torch.int32)
    _build_prefill_row_table_kernel[(1,)](mem_index[8:], row_table, 3, BLOCK=256)
    gathered_kv = torch.empty((indices.numel(), 1, 512), device="cuda", dtype=torch.bfloat16)
    _gather_prefill_kv_kernel[(triton.cdiv(indices.numel(), 4),)](
        packed_kv[..., :584],
        indices,
        row_table,
        prefill_kv,
        gathered_kv,
        indices.numel(),
        packed_kv.stride(0),
        prefill_kv.stride(0),
        prefill_kv.stride(2),
        BLOCK_ROWS=4,
    )
    expected = _dequantize_kv_cache(packed_kv)
    expected[mem_index[8:]] = prefill_kv
    assert torch.equal(gathered_kv, expected[indices.long()])


@pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
def test_fp8_nope_decode_and_cuda_graph(num_heads):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53)
    packed_kv = torch.zeros(4096, 1, 720, device="cuda", dtype=torch.uint8)
    source = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
    mem_index = torch.arange(4096, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem_index, packed_kv)
    q = torch.randn(num_heads, 9, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    # Includes a HOLD row, short tails, and three consecutive MTP positions.
    lengths = torch.tensor([0, 1, 3, 127, 128, 511, 2048, 2049, 2051], device="cuda", dtype=torch.int32)
    indices = torch.randint(0, 4096, (9, 2176), device="cuda", dtype=torch.int32)
    lanes = torch.arange(2176, device="cuda")
    indices.masked_fill_(lanes[None, :] >= lengths[:, None], -1)
    control = AttControl(
        nsa_decode=True, nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512}
    )
    infer_state = _make_infer_state()
    state = Fp8Glm5NextSparseDecodeState(
        _infer_state_weak_ref=weakref.ref(infer_state), flashmla_sched_meta=flash_mla.get_mla_metadata()[0]
    )

    def run():
        return state.decode_att((q, q[..., :0]), packed_kv[..., :584], None, control)

    def check(actual):
        expected = _torch_sparse_attention(q, _dequantize_kv_cache(packed_kv), indices)
        torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.015)

    check(run())
    graph, out = _capture_cuda_graph(run)
    graph.replay()
    check(out)
    q.normal_()
    source.normal_()
    write_fp8_e8m0_packed_kv(source, mem_index, packed_kv)
    lengths.copy_(lengths.roll(1))
    indices.random_(0, 4096)
    indices.masked_fill_(lanes[None, :] >= lengths[:, None], -1)
    graph.replay()
    check(out)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
@pytest.mark.parametrize("chunk_size", [3, 7, 32768])
def test_fp8_nope_batched_prefill_and_graph(monkeypatch, cached, num_heads, chunk_size):
    monkeypatch.setattr(fp8_prefill, "KV_GATHER_CHUNK_SIZE", chunk_size)
    torch.manual_seed(53)
    # Two requests whose physical cache locations are unrelated to batch order.
    prefix_len = 4 if cached else 0
    seq_len = prefix_len + 3
    ragged_mem_index = torch.randperm(64, device="cuda")[: 2 * seq_len].int()
    prefill_rows = torch.cat(
        (torch.arange(prefix_len, seq_len), torch.arange(seq_len + prefix_len, 2 * seq_len))
    ).cuda()
    prefill_mem_index = ragged_mem_index[prefill_rows]
    source = torch.randn(2 * seq_len, 1, 512, device="cuda", dtype=torch.bfloat16)
    prefill_kv = source[prefill_rows].clone().add_(0.25)
    packed_kv = torch.zeros(64, 1, 720, device="cuda", dtype=torch.uint8)
    write_fp8_e8m0_packed_kv(source, ragged_mem_index, packed_kv)
    q = torch.randn(num_heads, 6, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    kv_start_offsets = torch.tensor([0] * 3 + [seq_len] * 3, device="cuda", dtype=torch.int32)
    topk_indices = torch.full((6, 128), -1, device="cuda", dtype=torch.int32)
    for row in range(6):
        length = prefix_len + row % 3 + 1
        topk_indices[row, :length] = torch.arange(length, device="cuda")
    topk_indices[0].fill_(-1)
    topk_indices[1, :2] = prefix_len
    topk_indices[3].fill_(-1)
    topk_indices[3, 0] = prefix_len
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict={
            "topk_indices": topk_indices,
            "prefill_cache_kv": prefill_kv,
            "softmax_scale": 512 ** -0.5,
            "kv_lora_rank": 512,
        },
    )
    infer_state = _make_infer_state(max_cache_len=prefix_len, mem_index=prefill_mem_index)
    state = Fp8Glm5NextSparsePrefillState(
        backend=_make_workspace_backend(q.shape[0]),
        _infer_state_weak_ref=weakref.ref(infer_state),
        ks=kv_start_offsets,
        ragged_mem_index=ragged_mem_index,
    )

    def run():
        return state.prefill_att(q, packed_kv[..., :584], None, control)

    def check(out):
        expected_kv = _dequantize_kv_cache(packed_kv)[ragged_mem_index.long()]
        expected_kv[prefill_rows] = prefill_kv
        indices = torch.where(topk_indices >= 0, topk_indices + kv_start_offsets[:, None], -1)
        expected = _torch_sparse_attention(q, expected_kv, indices, control.nsa_prefill_dict["softmax_scale"])
        torch.testing.assert_close(out.float(), expected, atol=0.012, rtol=0.015)

    check(run())
    graph, out = _capture_cuda_graph(run)
    graph.replay()
    check(out)
    q.normal_()
    source.normal_()
    prefill_kv.copy_(source[prefill_rows] + 0.25)
    ragged_mem_index.copy_(ragged_mem_index.roll(1))
    prefill_mem_index.copy_(ragged_mem_index[prefill_rows])
    topk_indices.copy_(topk_indices.roll(1, dims=1))
    write_fp8_e8m0_packed_kv(source, ragged_mem_index, packed_kv)
    graph.replay()
    check(out)


@pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
@pytest.mark.parametrize("softmax_scale", [0.0625, 512 ** -0.5])
def test_fp8_chunked_prefill_sparse_rows_and_graph(monkeypatch, num_heads, softmax_scale):
    monkeypatch.setattr(fp8_prefill, "KV_GATHER_CHUNK_SIZE", 129)
    torch.manual_seed(53)
    num_tokens = 1025
    source = torch.randn(num_tokens, 1, 1024, device="cuda", dtype=torch.bfloat16)[..., :512]
    ragged_mem_index = torch.randperm(2048, device="cuda")[:num_tokens].int()
    prefill_rows = torch.tensor([1, 15, 128, 129, 255, 257, 511, 512, 1024], device="cuda")
    prefill_mem_index = ragged_mem_index[prefill_rows].long()
    prefill_kv = torch.randn(9, 1, 1024, device="cuda", dtype=torch.bfloat16)[..., :512]
    packed_kv = torch.zeros(2048, 1, 720, device="cuda", dtype=torch.uint8)
    write_fp8_e8m0_packed_kv(source, ragged_mem_index, packed_kv)
    q = torch.randn(num_heads, 9, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    lengths = torch.tensor([0, 1, 3, 127, 128, 511, 2048, 2049, 2051], device="cuda")
    topk_indices = torch.randint(num_tokens, (9, 2176), device="cuda", dtype=torch.int32)
    lanes = torch.arange(2176, device="cuda")
    topk_indices.masked_fill_(lanes[None, :] >= lengths[:, None], -1)
    topk_indices[1, 0] = 15
    topk_indices[2, :3] = torch.tensor([2, 3, 4], device="cuda")
    topk_indices[3, :127] = prefill_rows[torch.randint(9, (127,), device="cuda")].int()
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict={
            "topk_indices": topk_indices,
            "prefill_cache_kv": prefill_kv,
            "softmax_scale": softmax_scale,
            "kv_lora_rank": 512,
        },
    )
    infer_state = _make_infer_state(mem_index=prefill_mem_index)
    state = Fp8Glm5NextSparsePrefillState(
        backend=_make_workspace_backend(q.shape[0]),
        _infer_state_weak_ref=weakref.ref(infer_state),
        ks=torch.zeros(9, device="cuda", dtype=torch.int32),
        ragged_mem_index=ragged_mem_index,
    )

    def run():
        return state.prefill_att(q, packed_kv[..., :584], None, control)

    def check(actual):
        expected_kv = _dequantize_kv_cache(packed_kv)[ragged_mem_index.long()]
        expected_kv[prefill_rows] = prefill_kv
        expected = _torch_sparse_attention(q, expected_kv, topk_indices, softmax_scale)
        torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.015)

    check(run())
    graph, output = _capture_cuda_graph(run)
    graph.replay()
    check(output)
    q.normal_()
    source.normal_()
    prefill_kv.normal_()
    ragged_mem_index.copy_(ragged_mem_index.roll(1))
    prefill_mem_index.copy_(ragged_mem_index[prefill_rows])
    write_fp8_e8m0_packed_kv(source, ragged_mem_index, packed_kv)
    topk_indices.copy_(topk_indices.roll(1, dims=1))
    graph.replay()
    check(output)


def _sampled_h64_sparse_reference(q, packed, ragged, prefill_mem, prefill_kv, starts, topk, samples):
    """Reference only a few H64 rows so the large-Q regression stays bounded."""
    expected = []
    for row, heads in samples:
        local = topk[row]
        local = local[local >= 0].long()
        physical = ragged[(starts[row].long() + local)].long()
        values = _dequantize_kv_cache(packed[physical])[:, 0].float()
        current = physical[:, None] == prefill_mem[None, :].long()
        has_current = current.any(dim=1)
        current_row = current.int().argmax(dim=1)
        values[has_current] = prefill_kv[current_row[has_current], 0].float()
        expected.append(((q[row, heads].float() @ values.T * 0.0625).softmax(-1) @ values))
    return expected


@pytest.mark.parametrize("num_q", [16, 33])
def test_fp8_h64_largeq_native_prefill_permuted_ragged_graph(monkeypatch, num_q):
    """The native sparse prefill path maps top-k ragged positions, never physical IDs."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse prefill target is Hopper")
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    torch.manual_seed(53100 + num_q)
    capacity = 131072
    ragged_n = 65536
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    write_fp8_e8m0_packed_kv(source, torch.arange(capacity, device="cuda", dtype=torch.int32), packed)
    ragged = torch.randperm(capacity, device="cuda", dtype=torch.int32)[:ragged_n]
    q = torch.randn(num_q, 64, 512, device="cuda", dtype=torch.bfloat16)
    starts = torch.where(
        torch.arange(num_q, device="cuda", dtype=torch.int32) < num_q // 2,
        torch.full((num_q,), 1024, device="cuda", dtype=torch.int32),
        torch.full((num_q,), 32768, device="cuda", dtype=torch.int32),
    )
    current_local = (torch.arange(num_q, device="cuda", dtype=torch.int32) * 37) % 2048 + 64
    current_positions = starts + current_local
    prefill_mem = ragged[current_positions.long()].clone()
    prefill_kv = torch.randn(num_q, 1, 512, device="cuda", dtype=torch.bfloat16)
    prefill_kv.add_(torch.arange(num_q, device="cuda", dtype=torch.bfloat16)[:, None, None])
    topk = torch.randint(0, 2176, (num_q, 2176), device="cuda", dtype=torch.int32)
    topk[:, ::19] = -1  # Holes must survive the native position-preserving map.
    topk[:, 0] = current_local
    topk[:, 1] = current_local  # Duplicates are valid selected keys.
    topk[:, 2] = topk[:, 3]
    assert q.is_contiguous() and topk.is_contiguous()

    observed = []

    def observed_sparse(*args, **kwargs):
        observed.append((kwargs["q"].shape, kwargs["kv"].shape, kwargs["indices"].shape))
        return flash_mla_sparse_fwd(*args, **kwargs)

    monkeypatch.setattr("sgl_kernel.flash_mla.flash_mla_sparse_fwd", observed_sparse)

    def run():
        return fp8_prefill.context_attention_fwd_fp8(
            q, packed[..., :584], ragged, prefill_mem, prefill_kv, starts, topk, 0.0625
        )

    samples = [(0, torch.tensor([0, 9, 63], device="cuda")), (num_q - 1, torch.tensor([3, 31, 62], device="cuda"))]

    def check(actual):
        expected = _sampled_h64_sparse_reference(q, packed, ragged, prefill_mem, prefill_kv, starts, topk, samples)
        for (row, heads), value in zip(samples, expected):
            torch.testing.assert_close(actual[row, heads].float(), value, atol=0.012, rtol=0.015)

    eager = run()
    check(eager)
    assert observed == [((num_q, 64, 512), (ragged_n, 1, 512), (num_q, 1, 2176))]
    graph, output = _capture_cuda_graph(run)
    q.normal_()
    prefill_kv.normal_()
    ragged.copy_(ragged.roll(1))
    prefill_mem.copy_(ragged[current_positions.long()])
    topk.copy_(topk.roll(1, dims=1))
    topk[:, 1] = current_local
    graph.replay()
    check(output)
    topk.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(output) == 0
    assert len(observed) == 3  # eager, graph warmup, and capture; replay stays inside the CUDA graph.


def test_fp8_h8_largeq_prefill_keeps_flashattention_path(monkeypatch):
    """H8 large-Q prefill must not enter the H64 native sparse fast path."""
    import sgl_kernel.flash_attn as flash_attn
    import sgl_kernel.flash_mla as flash_mla

    torch.manual_seed(53180)
    capacity = 65536
    num_q = 16
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    write_fp8_e8m0_packed_kv(source, torch.arange(capacity, device="cuda", dtype=torch.int32), packed)
    q = torch.randn(num_q, 8, 512, device="cuda", dtype=torch.bfloat16)
    ragged = torch.randperm(capacity, device="cuda", dtype=torch.int32)
    starts = torch.full((num_q,), 1024, device="cuda", dtype=torch.int32)
    current_local = torch.arange(num_q, device="cuda", dtype=torch.int32) + 64
    prefill_mem = ragged[(starts + current_local).long()].clone()
    prefill_kv = torch.randn(num_q, 1, 512, device="cuda", dtype=torch.bfloat16)
    topk = torch.full((num_q, 2176), -1, device="cuda", dtype=torch.int32)
    topk[:, :64] = torch.arange(64, device="cuda", dtype=torch.int32)
    topk[:, 0] = current_local
    seen_flashattention = []
    original_flashattention = flash_attn.flash_attn_with_kvcache

    def observed_flashattention(*args, **kwargs):
        seen_flashattention.append(True)
        return original_flashattention(*args, **kwargs)

    def unexpected_native(*args, **kwargs):
        raise AssertionError("H8 large-Q prefill entered the native H64 sparse path")

    monkeypatch.setattr(flash_attn, "flash_attn_with_kvcache", observed_flashattention)
    monkeypatch.setattr(flash_mla, "flash_mla_sparse_fwd", unexpected_native)
    output = fp8_prefill.context_attention_fwd_fp8(
        q, packed[..., :584], ragged, prefill_mem, prefill_kv, starts, topk, 0.0625
    )
    assert seen_flashattention
    assert output.shape == q.shape


@pytest.mark.parametrize(
    "num_q,num_heads",
    [(1, 8), (2, 8), (8, 8), (1, 64), (2, 64), (3, 64), (4, 64), (5, 64), (6, 64), (7, 64), (8, 64)],
)
def test_fp8_direct_smallq_prefill_dispatch_strided_holes_and_current(num_q, num_heads):
    """Exercise eager and graph replay for no-rowtable, small-Q packed prefill."""
    torch.manual_seed(53016 + num_q)
    capacity = 65536
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem, packed)
    ragged = torch.arange(capacity, device="cuda", dtype=torch.int32)
    q = torch.randn(num_heads, num_q, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    # Requests deliberately have different spans; starts are not fixed-stride.
    seq_lens = torch.tensor([127, 521, 1031, 2053, 4099, 8191, 12289, 16381], device="cuda", dtype=torch.int32)[:num_q]
    starts = torch.nn.functional.pad(seq_lens.cumsum(0), (1, 0))[:-1]
    current_rows = starts + seq_lens - 1
    prefill_mem = ragged[current_rows]
    prefill_kv = torch.randn(num_q, 1, 1024, device="cuda", dtype=torch.bfloat16)[..., :512]
    topk_backing = torch.full((num_q, 2176 * 2), -1, device="cuda", dtype=torch.int32)
    topk = topk_backing[:, ::2]  # Exercise the topk stride passed to the direct kernel.
    topk[:, :3] = torch.tensor([3, 11, 11], device="cuda", dtype=torch.int32)
    topk[:, 3] = seq_lens - 1

    def run():
        return fp8_prefill.context_attention_fwd_fp8(
            q, packed[..., :584], ragged, prefill_mem, prefill_kv, starts, topk, 0.0625
        )

    def check(actual):
        assert actual.shape == q.shape and actual.is_contiguous()
        expected_kv = _dequantize_kv_cache(packed)[ragged.long()]
        expected_kv[current_rows.long()] = prefill_kv
        physical = torch.where(topk >= 0, topk + starts[:, None], -1)
        expected = _torch_sparse_attention(q, expected_kv, physical)
        torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.015)

    actual = run()
    check(actual)
    assert torch.count_nonzero(actual) > 0  # Includes BF16 current KV in every row.
    graph, output = _capture_cuda_graph(run)
    graph.replay()
    check(output)
    q.normal_()
    source.normal_()
    prefill_kv.normal_()
    ragged.copy_(ragged.roll(1))
    prefill_mem.copy_(ragged[current_rows])
    write_fp8_e8m0_packed_kv(source, mem, packed)
    topk.copy_(topk.roll(1, dims=1))
    graph.replay()
    check(output)
    topk.fill_(-1)  # Empty graph replay must not read invalid packed pointers.
    graph.replay()
    assert torch.count_nonzero(output) == 0


def test_fp8_h64_direct_smallq_state_workspace_reuse_and_fresh_state():
    """Q1 H64 direct workspace is owned by the backend and reused across infer states."""
    torch.manual_seed(53150)
    capacity = 65536
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem, packed)
    ragged = torch.arange(capacity, device="cuda", dtype=torch.int32)
    q = torch.randn(64, 1, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1).contiguous()
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    current_rows = torch.tensor([2048], device="cuda", dtype=torch.int32)
    prefill_mem = ragged[current_rows]
    prefill_kv = torch.randn(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    topk = torch.full((1, 2176), -1, device="cuda", dtype=torch.int32)
    topk[:, :2048] = torch.arange(2048, device="cuda", dtype=torch.int32)
    topk[:, 2048] = current_rows
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict={
            "topk_indices": topk,
            "prefill_cache_kv": prefill_kv,
            "softmax_scale": 0.0625,
            "kv_lora_rank": 512,
        },
    )
    infer_state = _make_infer_state(mem_index=prefill_mem)
    backend = _make_workspace_backend(q.shape[0])
    state = Fp8Glm5NextSparsePrefillState(
        backend=backend, _infer_state_weak_ref=weakref.ref(infer_state), ks=starts, ragged_mem_index=ragged
    )

    def expected():
        values = _dequantize_kv_cache(packed)[ragged.long()]
        values[current_rows.long()] = prefill_kv
        return _torch_sparse_attention(q, values, topk)

    def run(active_state=state):
        return active_state.prefill_att(q, packed[..., :584], None, control)

    torch.testing.assert_close(run().float(), expected(), atol=0.012, rtol=0.015)
    cache = backend._prefill_workspaces[0]["smallq"]
    assert not hasattr(infer_state, "_glm5_fp8_smallq_workspace_cache")
    workspace = cache["direct"]
    pointers = {name: tensor.data_ptr() for name, tensor in workspace.items()}
    for _ in range(10):
        layer_state = Fp8Glm5NextSparsePrefillState(
            backend=backend, _infer_state_weak_ref=weakref.ref(infer_state), ks=starts, ragged_mem_index=ragged
        )
        torch.testing.assert_close(run(layer_state).float(), expected(), atol=0.012, rtol=0.015)
    assert {name: tensor.data_ptr() for name, tensor in cache["direct"].items()} == pointers

    fresh_infer_state = _make_infer_state(mem_index=prefill_mem.clone())
    fresh_state = Fp8Glm5NextSparsePrefillState(
        backend=backend, _infer_state_weak_ref=weakref.ref(fresh_infer_state), ks=starts, ragged_mem_index=ragged
    )
    torch.testing.assert_close(run(fresh_state).float(), expected(), atol=0.012, rtol=0.015)
    fresh_workspace = backend._prefill_workspaces[0]["smallq"]["direct"]
    assert fresh_workspace["partial_o"].data_ptr() == workspace["partial_o"].data_ptr()

    q.normal_()
    source.normal_()
    prefill_kv.normal_()
    write_fp8_e8m0_packed_kv(source, mem, packed)
    torch.testing.assert_close(run().float(), expected(), atol=0.012, rtol=0.015)
    assert {name: tensor.data_ptr() for name, tensor in cache["direct"].items()} == pointers


@pytest.mark.parametrize("num_q", [2, 3, 5, 8])
def test_fp8_h64_selected_prefill_state_workspace_reuse(num_q):
    """H64 selected prefill rewrites backend workspace across DSA layers."""
    torch.manual_seed(53200 + num_q)
    capacity = 65536
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem, packed)
    ragged = torch.arange(capacity, device="cuda", dtype=torch.int32)
    q = torch.randn(64, num_q, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    starts = torch.arange(num_q, device="cuda", dtype=torch.int32) * 4096
    current_rows = starts + 2048
    prefill_mem = ragged[current_rows]
    prefill_kv = torch.randn(num_q, 1, 512, device="cuda", dtype=torch.bfloat16)
    topk = torch.full((num_q, 2176), -1, device="cuda", dtype=torch.int32)
    topk[:, :2048] = torch.arange(2048, device="cuda", dtype=torch.int32)
    topk[:, 2048] = 2048
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict={
            "topk_indices": topk,
            "prefill_cache_kv": prefill_kv,
            "softmax_scale": 0.0625,
            "kv_lora_rank": 512,
        },
    )
    infer_state = _make_infer_state(mem_index=prefill_mem)
    backend = _make_workspace_backend(q.shape[0])
    state = Fp8Glm5NextSparsePrefillState(
        backend=backend, _infer_state_weak_ref=weakref.ref(infer_state), ks=starts, ragged_mem_index=ragged
    )

    def expected():
        values = _dequantize_kv_cache(packed)[ragged.long()]
        values[current_rows.long()] = prefill_kv
        physical = torch.where(topk >= 0, topk + starts[:, None], -1)
        return _torch_sparse_attention(q, values, physical)

    def run(active_state=state):
        return active_state.prefill_att(q, packed[..., :584], None, control)

    eager = run()
    torch.testing.assert_close(eager.float(), expected(), atol=0.012, rtol=0.015)
    cache = backend._prefill_workspaces[0]["smallq"]
    assert not hasattr(infer_state, "_glm5_fp8_smallq_workspace_cache")
    workspace = cache["selected"]
    pointers = {name: tensor.data_ptr() for name, tensor in workspace.items()}
    # All 11 DSA layers share the backend allocation and rewrite its contents.
    for _ in range(10):
        layer_state = Fp8Glm5NextSparsePrefillState(
            backend=backend, _infer_state_weak_ref=weakref.ref(infer_state), ks=starts, ragged_mem_index=ragged
        )
        torch.testing.assert_close(run(layer_state).float(), expected(), atol=0.012, rtol=0.015)
    assert {name: tensor.data_ptr() for name, tensor in cache["selected"].items()} == pointers

    fresh_infer_state = _make_infer_state(mem_index=prefill_mem.clone())
    fresh_state = Fp8Glm5NextSparsePrefillState(
        backend=backend, _infer_state_weak_ref=weakref.ref(fresh_infer_state), ks=starts, ragged_mem_index=ragged
    )
    torch.testing.assert_close(run(fresh_state).float(), expected(), atol=0.012, rtol=0.015)
    fresh_workspace = backend._prefill_workspaces[0]["smallq"]["selected"]
    assert fresh_workspace["compact"].data_ptr() == workspace["compact"].data_ptr()

    source.normal_()
    write_fp8_e8m0_packed_kv(source, mem, packed)
    prefill_kv.normal_()
    q.normal_()
    eager = run()
    assert {name: tensor.data_ptr() for name, tensor in cache["selected"].items()} == pointers
    torch.testing.assert_close(eager.float(), expected(), atol=0.012, rtol=0.015)
    graph, output = _capture_cuda_graph(run)
    q.normal_()
    topk.copy_(topk.roll(1, dims=1))
    graph.replay()
    torch.testing.assert_close(output.float(), expected(), atol=0.012, rtol=0.015)
    topk.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(output) == 0


def test_fp8_h64_b1_decode_uses_native_eager_and_direct_graph():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53300)
    capacity = 4096
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem, packed)
    q = torch.randn(64, 1, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    indices = torch.full((1, 2176), -1, device="cuda", dtype=torch.int32)
    indices[:, :2049] = torch.randint(0, capacity, (1, 2049), device="cuda", dtype=torch.int32)
    control = AttControl(
        nsa_decode=True, nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512}
    )
    infer = _make_infer_state(is_cuda_graph_capture=False)
    state = Fp8Glm5NextSparseDecodeState(
        _infer_state_weak_ref=weakref.ref(infer),
        decode_topk_bound=2051,
        flashmla_sched_meta=flash_mla.get_mla_metadata()[0],
    )

    def run():
        return state.decode_att((q, q[..., :0]), packed[..., :584], None, control)

    expected = _torch_sparse_attention(q, _dequantize_kv_cache(packed), indices)
    eager = run()
    torch.testing.assert_close(eager.float(), expected, atol=0.012, rtol=0.015)
    infer.is_cuda_graph_capture = True
    graph, output = _capture_cuda_graph(run)
    infer.is_cuda_graph_capture = False
    graph.replay()
    torch.testing.assert_close(output.float(), expected, atol=0.012, rtol=0.015)


@pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
def test_fp8_decode_qpad_kernel_strides_and_zero_padding(num_heads):
    """The capture padding kernel reads a true D-stride and writes every padded head as zero."""
    torch.manual_seed(53016 + num_heads)
    batch = 2
    # This has BMM-style B/H strides and a non-unit final dimension stride.
    q = torch.randn(num_heads, batch, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2].transpose(0, 1)
    padded_heads = ((num_heads + 63) // 64) * 64
    head_dim = q.shape[-1]
    block_size = triton.next_power_of_2(2 * head_dim)
    padded = torch.empty((batch, 1, padded_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    _fused_strided_qpad_kernel[(batch, triton.cdiv(padded_heads * head_dim, block_size))](
        q,
        padded,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        num_heads,
        padded_heads,
        head_dim=head_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    expected = torch.zeros_like(padded)
    expected[:, 0, :num_heads] = q
    assert torch.equal(padded, expected)
    assert torch.count_nonzero(padded[:, 0, num_heads:]) == 0


@pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
def test_fp8_decode_infer_state_graph_qpad_and_mutation(num_heads):
    """Exercise the actual backend's eager and graph flags with strided BMM Q."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53100 + num_heads)
    batch, capacity = 2, 4096
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem, packed)
    q = torch.randn(num_heads, batch, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2].transpose(0, 1)
    indices = torch.full((batch, 2176), -1, device="cuda", dtype=torch.int32)
    indices[:, :2049] = torch.randint(0, capacity, (batch, 2049), device="cuda", dtype=torch.int32)
    control = AttControl(
        nsa_decode=True,
        nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512},
    )
    infer_state = _make_infer_state(is_cuda_graph_capture=False)
    state = Fp8Glm5NextSparseDecodeState(
        _infer_state_weak_ref=weakref.ref(infer_state), flashmla_sched_meta=flash_mla.get_mla_metadata()[0]
    )

    def run():
        return state.decode_att((q, q[..., :0]), packed[..., :584], None, control)

    def reference():
        return _torch_sparse_attention(q, _dequantize_kv_cache(packed), indices)

    eager = run()
    torch.testing.assert_close(eager.float(), reference(), atol=0.012, rtol=0.015)
    assert eager.shape == q.shape
    infer_state.is_cuda_graph_capture = True
    graph, output = _capture_cuda_graph(run)
    infer_state.is_cuda_graph_capture = False
    graph.replay()
    torch.testing.assert_close(output.float(), reference(), atol=0.012, rtol=0.015)
    q.copy_(q.roll(1, dims=-1))
    saved = indices.clone()
    indices[:, :2049].copy_(saved[:, :2049].roll(1, dims=1))
    assert torch.equal(indices[:, 2049:], saved[:, 2049:])
    graph.replay()
    torch.testing.assert_close(output.float(), reference(), atol=0.012, rtol=0.015)


def test_fp8_sparse_direct_decode_factory_bound_guard():
    backend = object.__new__(Fp8Glm5NextSparseAttBackend)
    backend.model = SimpleNamespace(config={"index_topk": 2048, "index_kpool": 4})
    infer_state = InferStateInfo()
    assert backend.create_att_decode_state(infer_state).decode_topk_bound == 2051
    backend.model = SimpleNamespace(config={"index_topk": 2048, "index_kpool": 8})
    assert backend.create_att_decode_state(infer_state).decode_topk_bound is None
    backend.model = SimpleNamespace(config={"index_topk": 2047, "index_kpool": 4})
    assert backend.create_att_decode_state(infer_state).decode_topk_bound is None


@pytest.mark.parametrize("batch", [1, 8])
def test_fp8_sparse_direct_decode_actual_entry_edges_and_graph(monkeypatch, batch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53200 + batch)
    capacity = 4096
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem, packed)
    q = torch.randn(8, batch, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    indices = torch.full((batch, 2176), -1, device="cuda", dtype=torch.int32)
    counts = [2051] if batch == 1 else [0, 2048, 2049, 2050, 2051, 1985, 2051, 2051]
    for row in range(batch):
        count = counts[row]
        if count:
            indices[row, :count] = torch.randint(0, capacity, (count,), device="cuda", dtype=torch.int32)
    if batch == 8:
        indices[5, :64] = -1  # holes
        indices[6, 64:128] = indices[6, :64]  # duplicates
    assert torch.equal(indices[:, 2051:], torch.full_like(indices[:, 2051:], -1))
    control = AttControl(
        nsa_decode=True, nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512}
    )
    state_owner = _make_infer_state(is_cuda_graph_capture=False)
    state = Fp8Glm5NextSparseDecodeState(
        _infer_state_weak_ref=weakref.ref(state_owner),
        flashmla_sched_meta=flash_mla.get_mla_metadata()[0],
        decode_topk_bound=2051,
    )

    def native_must_not_run(*args, **kwargs):
        raise AssertionError("supported GLM-5.3 decode took native FlashMLA fallback")

    def run():
        return state.decode_att((q, q[..., :0]), packed[..., :584], None, control)

    assert not can_use_fp8_sparse_direct_decode(q, packed, indices.unsqueeze(1), 512, 2051)
    expected = _torch_sparse_attention(q, _dequantize_kv_cache(packed), indices)
    eager = run()
    torch.testing.assert_close(eager.float(), expected, atol=0.02, rtol=0.02)
    if batch == 1:
        assert torch.count_nonzero(eager) > 0
    monkeypatch.setattr(flash_mla, "flash_mla_with_kvcache", native_must_not_run)
    state.infer_state.is_cuda_graph_capture = True
    graph, output = _capture_cuda_graph(run)
    state.infer_state.is_cuda_graph_capture = False
    graph.replay()
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
    q.copy_(q.roll(1, dims=-1))
    saved = indices.clone()
    nonempty = (indices >= 0).sum(dim=1)
    for row, count in enumerate(nonempty.tolist()):
        if count:
            indices[row, :count].copy_(saved[row, :count].roll(1))
    assert torch.equal(indices[:, 2051:], saved[:, 2051:])
    graph.replay()
    expected = _torch_sparse_attention(q, _dequantize_kv_cache(packed), indices)
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
    indices.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize(
    "num_heads,batch",
    [
        (1, 1),
        (4, 3),
        (7, 5),
        (8, 2),
        (8, 3),
        (8, 4),
        (8, 5),
        (16, 1),
        (16, 3),
        (16, 4),
        (17, 3),
        (32, 1),
        (32, 2),
        (32, 3),
        (64, 1),
        (64, 2),
        (128, 1),
    ],
)
def test_fp8_sparse_direct_decode_general_graph_factory_entry(monkeypatch, num_heads, batch):
    """General head/token shapes use the direct path selected by the central predicate."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53300 + batch)
    capacity = 4096
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem, packed)
    # BMM-style B/H stride: (512, batch * 512, 1).
    q = torch.randn(num_heads, batch, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    assert q.stride() == (512, batch * 512, 1)
    indices = torch.full((batch, 2176), -1, device="cuda", dtype=torch.int32)
    counts = [([2051, 0, 2048, 2050, 1, 3, 127, 128][row % 8]) for row in range(batch)]
    for row, count in enumerate(counts):
        if count:
            indices[row, :count] = torch.randint(0, capacity, (count,), device="cuda", dtype=torch.int32)
    if batch >= 3:
        indices[2, :32] = -1  # holes
    if batch >= 4:
        indices[3, 64:128] = indices[3, :64]  # duplicates
    assert torch.equal(indices[:, 2051:], torch.full_like(indices[:, 2051:], -1))
    control = AttControl(
        nsa_decode=True,
        nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512},
    )
    backend = object.__new__(Fp8Glm5NextSparseAttBackend)
    backend.model = SimpleNamespace(config={"index_topk": 2048, "index_kpool": 4})
    infer_state = _make_infer_state(is_cuda_graph_capture=False)
    state = backend.create_att_decode_state(infer_state)
    state.flashmla_sched_meta = flash_mla.get_mla_metadata()[0]
    assert state.decode_topk_bound == 2051
    native_state_owner = _make_infer_state(is_cuda_graph_capture=False)
    native_state = Fp8Glm5NextSparseDecodeState(
        _infer_state_weak_ref=weakref.ref(native_state_owner),
        flashmla_sched_meta=flash_mla.get_mla_metadata()[0],
        decode_topk_bound=None,
    )

    def run(current_state):
        return current_state.decode_att((q, q[..., :0]), packed[..., :584], None, control)

    native_calls = []
    original_native = flash_mla.flash_mla_with_kvcache

    def observe_native(*args, **kwargs):
        native_calls.append(True)
        return original_native(*args, **kwargs)

    monkeypatch.setattr(flash_mla, "flash_mla_with_kvcache", observe_native)
    eager = run(state)
    native_output = run(native_state)
    assert native_calls
    assert torch.equal(eager, native_output)
    expected = _torch_sparse_attention(q, _dequantize_kv_cache(packed), indices)
    torch.testing.assert_close(eager.float(), expected, atol=0.02, rtol=0.02)

    def native_must_not_run(*args, **kwargs):
        raise AssertionError("supported graph shape did not use the direct packed-KV path")

    monkeypatch.setattr(flash_mla, "flash_mla_with_kvcache", native_must_not_run)
    state.infer_state.is_cuda_graph_capture = True
    graph, output = _capture_cuda_graph(lambda: run(state))
    state.infer_state.is_cuda_graph_capture = False
    graph.replay()
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
    q.copy_(q.roll(1, dims=-1))
    saved = indices.clone()
    for row, count in enumerate((indices >= 0).sum(dim=1).tolist()):
        if count:
            indices[row, :count].copy_(saved[row, :count].roll(1))
    assert torch.equal(indices[:, 2051:], saved[:, 2051:])
    graph.replay()
    torch.testing.assert_close(
        output.float(), _torch_sparse_attention(q, _dequantize_kv_cache(packed), indices), atol=0.02, rtol=0.02
    )
    indices.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(output) == 0


def test_fp8_sparse_direct_decode_unknown_bound_fallback_and_strided_q(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53250)
    capacity = 4096
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem, packed)
    q = torch.randn(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    late = torch.full((1, 2176), -1, device="cuda", dtype=torch.int32)
    late[:, 2175] = 17
    control = AttControl(
        nsa_decode=True, nsa_decode_dict={"topk_mem_indices": late, "softmax_scale": 0.0625, "kv_lora_rank": 512}
    )
    calls = []
    original = flash_mla.flash_mla_with_kvcache

    def native(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(flash_mla, "flash_mla_with_kvcache", native)
    unknown_owner = _make_infer_state(is_cuda_graph_capture=False)
    unknown = Fp8Glm5NextSparseDecodeState(
        _infer_state_weak_ref=weakref.ref(unknown_owner), flashmla_sched_meta=flash_mla.get_mla_metadata()[0]
    )
    actual = unknown.decode_att((q, q[..., :0]), packed[..., :584], None, control)
    torch.testing.assert_close(
        actual.float(), _torch_sparse_attention(q, _dequantize_kv_cache(packed), late), atol=0.02, rtol=0.02
    )
    assert calls and not can_use_fp8_sparse_direct_decode(
        q, packed[..., :584], late.unsqueeze(1), 512, None, is_cuda_graph_capture=True
    )
    q_d2 = torch.randn(1, 8, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2]
    good = torch.full((1, 2176), -1, device="cuda", dtype=torch.int32)
    good[:, :2051] = torch.randint(0, capacity, (1, 2051), device="cuda", dtype=torch.int32)
    assert can_use_fp8_sparse_direct_decode(
        q_d2, packed[..., :584], good.unsqueeze(1), 512, 2051, is_cuda_graph_capture=True
    )
    assert not can_use_fp8_sparse_direct_decode(
        q, packed[..., :584], good.unsqueeze(1), 256, 2051, is_cuda_graph_capture=True
    )


@pytest.mark.parametrize("num_heads,batch", [(1, 3), (7, 5), (16, 16), (32, 7), (65, 3), (128, 8)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fp8_sparse_direct_decode_strided_inputs_and_packed_row_padding(monkeypatch, num_heads, batch, index_dtype):
    """The kernel handles partial head tiles, larger batches and actual Q/index strides."""
    torch.manual_seed(53600 + num_heads + batch)
    capacity = 4096
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    write_fp8_e8m0_packed_kv(source, torch.arange(capacity, device="cuda", dtype=torch.int32), packed)
    q = torch.randn(num_heads, batch, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2].transpose(0, 1)
    indices = torch.full((batch, 1, 4352), -1, device="cuda", dtype=index_dtype)[..., ::2]
    indices[..., :2051] = torch.randint(0, capacity, (batch, 1, 2051), device="cuda", dtype=index_dtype)
    indices[0].fill_(-1)

    def run():
        return fp8_sparse_direct_decode(q, packed, indices, 0.0625)

    def expected():
        return _torch_sparse_attention(q, _dequantize_kv_cache(packed), indices[:, 0])

    if (num_heads, batch) in ((1, 3), (7, 5)):
        infer = _make_infer_state(is_cuda_graph_capture=True)
        state = Fp8Glm5NextSparseDecodeState(_infer_state_weak_ref=weakref.ref(infer), decode_topk_bound=2051)

        def native_must_not_run(*args, **kwargs):
            raise AssertionError("supported strided decode took native fallback")

        monkeypatch.setattr("flash_mla.flash_mla_with_kvcache", native_must_not_run)
        torch.testing.assert_close(
            state.decode_att(
                (q, q[..., :0]),
                packed,
                None,
                AttControl(
                    nsa_decode=True,
                    nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512},
                ),
            ).float(),
            expected(),
            atol=0.02,
            rtol=0.02,
        )

    torch.testing.assert_close(run().float(), expected(), atol=0.02, rtol=0.02)
    graph, output = _capture_cuda_graph(run)
    graph.replay()
    torch.testing.assert_close(output.float(), expected(), atol=0.02, rtol=0.02)
    q.normal_()
    indices[..., :2051].copy_(indices[..., :2051].roll(1, dims=0))
    graph.replay()
    torch.testing.assert_close(output.float(), expected(), atol=0.02, rtol=0.02)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fp8_sparse_direct_decode_central_predicate_accepts_strides_and_rejects_unaligned_kv(index_dtype):
    q = torch.empty(1, 8, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2]
    packed = torch.empty(8, 1, 720, device="cuda", dtype=torch.uint8)
    topk = torch.empty(1, 1, 4352, device="cuda", dtype=index_dtype)[..., ::2]
    assert can_use_fp8_sparse_direct_decode(q, packed, topk, 512, 2051, is_cuda_graph_capture=True)
    odd_rows = torch.empty(8, 1, 721, device="cuda", dtype=torch.uint8)[..., :584]
    odd_base = torch.empty(8, 1, 722, device="cuda", dtype=torch.uint8)[..., 1:585]
    strided_columns = torch.empty(8, 1, 1440, device="cuda", dtype=torch.uint8)[..., ::2]
    for invalid in (odd_rows, odd_base, strided_columns):
        assert not can_use_fp8_sparse_direct_decode(q, invalid, topk, 512, 2051, is_cuda_graph_capture=True)


@pytest.mark.parametrize("strided_input", ["q", "topk"])
def test_fp8_sparse_direct_decode_offsets_beyond_int32(strided_input):
    batch, heads, capacity = 4, 1, 64
    stride = 800_000_000
    shape = (batch, heads, 512) if strided_input == "q" else (batch, 1, 2176)
    dtype = torch.bfloat16 if strided_input == "q" else torch.int32
    storage_size = (batch - 1) * stride + shape[-1]
    required_bytes = storage_size * torch.empty((), dtype=dtype).element_size()
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < required_bytes + 512 * 1024 ** 2:
        pytest.skip("Large-stride regression requires sufficient free GPU memory")
    # Allocate address space without initializing the gaps between logical rows.
    storage = torch.empty(storage_size, device="cuda", dtype=dtype)
    strided = storage.as_strided(shape, (stride, shape[-1], 1))
    q = strided if strided_input == "q" else torch.empty(batch, heads, 512, device="cuda", dtype=torch.bfloat16)
    topk = strided if strided_input == "topk" else torch.empty(batch, 1, 2176, device="cuda", dtype=torch.int32)
    q.copy_(torch.randn(q.shape, device="cuda", dtype=q.dtype))
    topk.fill_(-1)
    topk[..., :7] = torch.randint(0, capacity, (batch, 1, 7), device="cuda", dtype=topk.dtype)
    packed = torch.empty(capacity, 1, 720, device="cuda", dtype=torch.uint8)
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    write_fp8_e8m0_packed_kv(source, torch.arange(capacity, device="cuda", dtype=torch.int32), packed)
    assert can_use_fp8_sparse_direct_decode(q, packed, topk, 512, 2051, is_cuda_graph_capture=True)
    actual = fp8_sparse_direct_decode(q, packed, topk, 0.0625)
    expected = _torch_sparse_attention(q, _dequantize_kv_cache(packed), topk[:, 0])
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("num_heads,batch", [(16, 16), (32, 8), (64, 4), (128, 2)])
def test_fp8_sparse_direct_decode_large_workload_retains_native_graph(monkeypatch, num_heads, batch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53700 + num_heads + batch)
    packed = torch.empty(4096, 1, 720, device="cuda", dtype=torch.uint8)
    source = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
    write_fp8_e8m0_packed_kv(source, torch.arange(4096, device="cuda", dtype=torch.int32), packed)
    q = torch.randn(batch, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    topk = torch.full((batch, 1, 2176), -1, device="cuda", dtype=torch.int32)
    topk[..., :2051] = torch.randint(0, 4096, (batch, 1, 2051), device="cuda", dtype=torch.int32)
    assert not can_use_fp8_sparse_direct_decode(q, packed, topk, 512, 2051, is_cuda_graph_capture=True)

    def direct_must_not_run(*args, **kwargs):
        raise AssertionError("larger workload took the slower direct path")

    monkeypatch.setattr(
        "lightllm.models.glm5_next.triton_kernel.fp8_sparse_decode.fp8_sparse_direct_decode", direct_must_not_run
    )
    infer = _make_infer_state(is_cuda_graph_capture=True)
    state = Fp8Glm5NextSparseDecodeState(
        _infer_state_weak_ref=weakref.ref(infer),
        decode_topk_bound=2051,
        flashmla_sched_meta=flash_mla.get_mla_metadata()[0],
    )
    control = AttControl(
        nsa_decode=True, nsa_decode_dict={"topk_mem_indices": topk, "softmax_scale": 0.0625, "kv_lora_rank": 512}
    )
    run = lambda: state.decode_att((q, q[..., :0]), packed[..., :584], None, control)
    expected = _torch_sparse_attention(q, _dequantize_kv_cache(packed), topk[:, 0])
    torch.testing.assert_close(run().float(), expected, atol=0.02, rtol=0.02)
    graph, output = _capture_cuda_graph(run)
    graph.replay()
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)


def test_fp8_h64_generic_prefill_workspace_reuse_mutable_inputs_and_graph_fallback():
    """Generic H64 backend buffers are reused, batch metadata resets, and graph falls back."""
    torch.manual_seed(53450)
    capacity, num_q = 65536, 13
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    write_fp8_e8m0_packed_kv(source, mem, packed)
    ragged = torch.randperm(capacity, device="cuda", dtype=torch.int32)
    starts = torch.arange(num_q, device="cuda", dtype=torch.int32) * 2048
    current_local = torch.full((num_q,), 2048, device="cuda", dtype=torch.int32)
    prefill_mem = ragged[(starts + current_local).long()].clone()
    prefill_kv = torch.randn(num_q, 1, 512, device="cuda", dtype=torch.bfloat16).add_(2)
    q = torch.randn(64, num_q, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1).contiguous()
    topk = torch.full((num_q, 2176), -1, device="cuda", dtype=torch.int32)
    topk[:, :2048] = torch.arange(2048, device="cuda", dtype=torch.int32)
    topk[:, 2048] = current_local
    topk[:, 2049] = current_local
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict={
            "topk_indices": topk,
            "prefill_cache_kv": prefill_kv,
            "softmax_scale": 0.0625,
            "kv_lora_rank": 512,
        },
    )
    infer = _make_infer_state(mem_index=prefill_mem)
    backend = _make_workspace_backend(2 * num_q)

    def expected(active_ragged=ragged):
        values = _dequantize_kv_cache(packed)[active_ragged.long()]
        values[(starts + current_local).long()] = prefill_kv
        physical = torch.where(topk >= 0, topk + starts[:, None], -1)
        return _torch_sparse_attention(q, values, physical)

    def run(active):
        return active.prefill_att(q, packed[..., :584], None, control)

    state = Fp8Glm5NextSparsePrefillState(
        backend=backend, _infer_state_weak_ref=weakref.ref(infer), ks=starts, ragged_mem_index=ragged
    )
    torch.testing.assert_close(run(state).float(), expected(), atol=0.012, rtol=0.015)
    cache = backend._prefill_workspaces[0]
    assert not hasattr(infer, "_fp8_sparse_generic_prefill_workspace_cache")
    workspace = cache["generic"]["workspace"]
    pointers = {k: v.data_ptr() for k, v in workspace.items() if isinstance(v, torch.Tensor)}
    for _ in range(10):
        layer = Fp8Glm5NextSparsePrefillState(
            backend=backend, _infer_state_weak_ref=weakref.ref(infer), ks=starts, ragged_mem_index=ragged
        )
        torch.testing.assert_close(run(layer).float(), expected(), atol=0.012, rtol=0.015)
    assert workspace["row_table_built"]
    assert {
        k: v.data_ptr() for k, v in cache["generic"]["workspace"].items() if isinstance(v, torch.Tensor)
    } == pointers

    # Change raw KV/top-k in place: reusable metadata must not cache either.
    q.normal_()
    prefill_kv.normal_()
    topk.copy_(topk.roll(1, dims=1))
    topk[:, 2049] = current_local
    torch.testing.assert_close(run(state).float(), expected(), atol=0.012, rtol=0.015)

    fresh_ragged = ragged.roll(1)
    fresh_mem = fresh_ragged[(starts + current_local).long()].clone()
    fresh = _make_infer_state(mem_index=fresh_mem)
    fresh_state = Fp8Glm5NextSparsePrefillState(
        backend=backend, _infer_state_weak_ref=weakref.ref(fresh), ks=starts, ragged_mem_index=fresh_ragged
    )
    torch.testing.assert_close(run(fresh_state).float(), expected(fresh_ragged), atol=0.012, rtol=0.015)
    assert cache["generic"]["workspace"]["prefill_row_table"].data_ptr() == workspace["prefill_row_table"].data_ptr()

    graph_infer = _make_infer_state(mem_index=prefill_mem, prefill_cuda_graph_exe_list=[])
    graph_state = Fp8Glm5NextSparsePrefillState(
        backend=backend, _infer_state_weak_ref=weakref.ref(graph_infer), ks=starts, ragged_mem_index=ragged
    )
    torch.testing.assert_close(run(graph_state).float(), expected(), atol=0.012, rtol=0.015)
    assert not hasattr(graph_infer, "_fp8_sparse_generic_prefill_workspace_cache")

    explicit_graph_infer = _make_infer_state(mem_index=prefill_mem, is_cuda_graph_capture=True)
    explicit_graph_state = Fp8Glm5NextSparsePrefillState(
        backend=backend, _infer_state_weak_ref=weakref.ref(explicit_graph_infer), ks=starts, ragged_mem_index=ragged
    )
    torch.testing.assert_close(run(explicit_graph_state).float(), expected(), atol=0.012, rtol=0.015)
    assert not hasattr(explicit_graph_infer, "_fp8_sparse_generic_prefill_workspace_cache")


def test_fp8_generic_prefill_workspace_graph_query_is_generic_h64_only(monkeypatch):
    """H8/direct/selected prefill never queries CUDA graph-capture state."""
    import lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 as fp8_prefill

    queries = []

    def query_spy():
        queries.append(True)
        raise AssertionError("non-generic path queried CUDA graph capture state")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", query_spy)
    kv = torch.empty((1, 1, 584), dtype=torch.uint8)
    params = {
        "prefill_cache_kv": torch.empty((1, 1, 512), dtype=torch.bfloat16),
        "topk_indices": torch.empty((1, 2176), dtype=torch.int32),
        "softmax_scale": 0.0625,
    }
    control = AttControl(nsa_prefill=True, nsa_prefill_dict=params)

    def fake_context(*args):
        return args[0]

    monkeypatch.setattr(fp8_prefill, "context_attention_fwd_fp8", fake_context)
    for heads, direct, selected in ((8, False, False), (64, True, False), (64, False, True)):
        monkeypatch.setattr(fp8_prefill, "can_use_direct_smallq", lambda *args, d=direct: d)
        monkeypatch.setattr(fp8_prefill, "can_use_selected_prefill", lambda *args, s=selected: s)
        infer = _make_infer_state(mem_index=torch.empty(1, dtype=torch.int32))
        backend = _make_workspace_backend(1)
        backend.get_smallq_workspace = lambda active_infer, active_q, kind="direct" if direct else "selected": {
            "kind": kind
        }
        state = Fp8Glm5NextSparsePrefillState(
            backend=backend,
            _infer_state_weak_ref=weakref.ref(infer),
            ks=torch.zeros(1, dtype=torch.int32),
            ragged_mem_index=torch.zeros(1, dtype=torch.int32),
        )
        assert state._nsa_prefill_att(torch.empty((1, heads, 512), dtype=torch.bfloat16), kv, control).shape[1] == heads
    assert not queries


def test_fp8_state_preserves_kv_and_indexer_across_cpu_and_pd(monkeypatch):
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.attention_state.full_att import FullAttentionConfig, create_mla_state

    args = SimpleNamespace(cpu_cache_token_page_size=4, pd_kv_page_size=4)
    monkeypatch.setattr("lightllm.common.attention_state.manager.get_env_start_args", lambda: args)
    monkeypatch.setattr("lightllm.common.attention_state.full_att.fp8_e8m0_packed_mla_state.get_env_start_args", lambda: args)
    config = FullAttentionConfig(logical_layer_groups=((3,),), global_head_num=1, head_dim=512,
                                dtype=torch.bfloat16, replicated=True)
    state = create_mla_state(
        config=config, kv_type="fp8kv_dsa", packed=True, fp8_format="group64_e8m0", page_size=4,
    )
    state.allocate_storage(token_capacity=16)
    state.storage.zero_()
    manager = AttentionStateManager(request_capacity=1)
    manager.register_state("main", state)
    manager.finish_state_registration()
    indices = torch.arange(4, dtype=torch.int64, device="cuda")
    kv = torch.randn((4, 1, 512), dtype=torch.bfloat16, device="cuda")
    state.write_full_att_kv(logical_layer_id=3, token_indices=indices, kv=kv)
    expected = torch.zeros((4, 1, 584), dtype=torch.uint8, device="cuda")
    write_fp8_e8m0_packed_kv(kv, indices, expected)
    torch.testing.assert_close(state.get_layer_view(logical_layer_id=3).kv[:4], expected, rtol=0, atol=0)
    state.get_layer_view(logical_layer_id=3).indexer_k[:4].random_(0, 256)
    cpu_layout = manager.get_cpu_page_layouts()[state.state_name]
    cpu_pages = torch.empty((1, manager.get_cpu_page_bytes()), dtype=torch.uint8, pin_memory=True)
    state.write_to_cpu(token_indices=indices, cpu_page_buffers=cpu_pages, cpu_page_indices=[0], layout=cpu_layout)
    state.read_from_cpu(cpu_page_buffers=cpu_pages, cpu_page_indices=[0], token_indices=indices + 4, layout=cpu_layout)
    pd_layout = manager.get_pd_token_page_layouts()[state.state_name]
    pd_pages = torch.empty((1, manager.get_pd_page_bytes()), dtype=torch.uint8, device="cuda")
    state.write_to_pd_page(token_indices=indices, pd_page_buffers=pd_pages, pd_page_index=0, layout=pd_layout)
    state.read_from_pd_page(pd_page_buffers=pd_pages, token_indices=indices + 8, pd_page_index=0, layout=pd_layout)
    torch.cuda.synchronize()
    for begin in (4, 8):
        actual = state.get_layer_view(logical_layer_id=3)
        torch.testing.assert_close(actual.kv[begin:begin + 4], actual.kv[:4], rtol=0, atol=0)
        torch.testing.assert_close(actual.indexer_k[begin:begin + 4], actual.indexer_k[:4], rtol=0, atol=0)
