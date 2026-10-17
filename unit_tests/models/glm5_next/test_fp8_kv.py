from types import SimpleNamespace

import pytest
import torch
import triton

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.nsa.glm5_next import (
    Glm5NextSparsePrefillState,
    Glm5NextSparseDecodeState,
)
from lightllm.common.basemodel.attention.nsa.fp8_glm5_next import (
    Fp8Glm5NextSparseAttBackend,
    Fp8Glm5NextSparsePrefillState,
    Fp8Glm5NextSparseDecodeState,
    _can_use_glm53_direct_decode,
    _fused_strided_qpad_kernel,
)
from lightllm.models.glm5_next.triton_kernel.destindex_copy_kv_flashmla_fp8 import destindex_copy_kv_flashmla_fp8
from lightllm.models.glm5_next.triton_kernel import context_flashattention_nopad_fp8 as fp8_prefill
from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
    _build_prefill_row_table_kernel,
    _gather_prefill_kv_kernel,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("kv_type", ["None", "fp8kv_dsa"])
def test_main_and_native_mtp_backend_selection(monkeypatch, draft, kv_type):
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel
    from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel

    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    model = object.__new__(Glm5NextMTPModel if draft else Glm5NextTpPartModel)
    model.args = SimpleNamespace(llm_kv_type=kv_type)
    model.graph_max_batch_size = 4
    model.max_seq_length = 128
    model._init_att_backend()
    prefill = model.prefill_att_backend.create_att_prefill_state(SimpleNamespace())
    decode = model.decode_att_backend.create_att_decode_state(SimpleNamespace())
    assert type(prefill) is (Fp8Glm5NextSparsePrefillState if kv_type == "fp8kv_dsa" else Glm5NextSparsePrefillState)
    assert type(decode) is (Fp8Glm5NextSparseDecodeState if kv_type == "fp8kv_dsa" else Glm5NextSparseDecodeState)


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
    destindex_copy_kv_flashmla_fp8(source, mem_index, packed_kv)
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
    graph, _ = _capture_cuda_graph(lambda: destindex_copy_kv_flashmla_fp8(source, mem_index, packed_kv))
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
    destindex_copy_kv_flashmla_fp8(source, mem_index, packed_kv)
    q = torch.randn(num_heads, 9, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    # Includes a HOLD row, short tails, and three consecutive MTP positions.
    lengths = torch.tensor([0, 1, 3, 127, 128, 511, 2048, 2049, 2051], device="cuda", dtype=torch.int32)
    indices = torch.randint(0, 4096, (9, 2176), device="cuda", dtype=torch.int32)
    lanes = torch.arange(2176, device="cuda")
    indices.masked_fill_(lanes[None, :] >= lengths[:, None], -1)
    control = AttControl(
        nsa_decode=True, nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512}
    )
    state = Fp8Glm5NextSparseDecodeState(flashmla_sched_meta=flash_mla.get_mla_metadata()[0])

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
    destindex_copy_kv_flashmla_fp8(source, mem_index, packed_kv)
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
    destindex_copy_kv_flashmla_fp8(source, ragged_mem_index, packed_kv)
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
            "softmax_scale": 512**-0.5,
            "kv_lora_rank": 512,
        },
    )
    state = Fp8Glm5NextSparsePrefillState(
        infer_state=SimpleNamespace(max_cache_len=prefix_len, mem_index=prefill_mem_index),
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
    destindex_copy_kv_flashmla_fp8(source, ragged_mem_index, packed_kv)
    graph.replay()
    check(out)


@pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
@pytest.mark.parametrize("softmax_scale", [0.0625, 512**-0.5])
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
    destindex_copy_kv_flashmla_fp8(source, ragged_mem_index, packed_kv)
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
    state = Fp8Glm5NextSparsePrefillState(
        infer_state=SimpleNamespace(mem_index=prefill_mem_index),
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
    destindex_copy_kv_flashmla_fp8(source, ragged_mem_index, packed_kv)
    topk_indices.copy_(topk_indices.roll(1, dims=1))
    graph.replay()
    check(output)


@pytest.mark.parametrize("num_q", [1, 2, 8])
def test_fp8_direct_smallq_prefill_dispatch_strided_holes_and_current(num_q):
    """Exercise eager and graph replay for no-rowtable, small-Q packed prefill."""
    torch.manual_seed(53016 + num_q)
    capacity = 65536
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    destindex_copy_kv_flashmla_fp8(source, mem, packed)
    ragged = torch.arange(capacity, device="cuda", dtype=torch.int32)
    q = torch.randn(8, num_q, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
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
    destindex_copy_kv_flashmla_fp8(source, mem, packed)
    topk.copy_(topk.roll(1, dims=1))
    graph.replay()
    check(output)
    topk.fill_(-1)  # Empty graph replay must not read invalid packed pointers.
    graph.replay()
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
def test_fp8_decode_qpad_kernel_strides_and_zero_padding(num_heads):
    """The capture padding kernel reads a true D-stride and writes every padded head as zero."""
    torch.manual_seed(53016 + num_heads)
    batch = 2
    # This has BMM-style B/H strides and a non-unit final dimension stride.
    q = torch.randn(num_heads, batch, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2].transpose(0, 1)
    padded_heads = ((num_heads + 63) // 64) * 64
    padded = torch.empty((batch, 1, padded_heads, 512), device="cuda", dtype=torch.bfloat16)
    _fused_strided_qpad_kernel[(batch, triton.cdiv(padded_heads * 512, 1024))](
        q,
        padded,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        num_heads,
        padded_heads,
        BLOCK1024=1024,
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
    destindex_copy_kv_flashmla_fp8(source, mem, packed)
    q = torch.randn(num_heads, batch, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2].transpose(0, 1)
    indices = torch.full((batch, 2176), -1, device="cuda", dtype=torch.int32)
    indices[:, :2049] = torch.randint(0, capacity, (batch, 2049), device="cuda", dtype=torch.int32)
    control = AttControl(
        nsa_decode=True,
        nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512},
    )
    infer_state = SimpleNamespace(is_cuda_graph=False)
    state = Fp8Glm5NextSparseDecodeState(infer_state=infer_state, flashmla_sched_meta=flash_mla.get_mla_metadata()[0])

    def run():
        return state.decode_att((q, q[..., :0]), packed[..., :584], None, control)

    def reference():
        return _torch_sparse_attention(q, _dequantize_kv_cache(packed), indices)

    eager = run()
    torch.testing.assert_close(eager.float(), reference(), atol=0.012, rtol=0.015)
    assert eager.shape == q.shape
    infer_state.is_cuda_graph = True
    graph, output = _capture_cuda_graph(run)
    infer_state.is_cuda_graph = False
    graph.replay()
    torch.testing.assert_close(output.float(), reference(), atol=0.012, rtol=0.015)
    q.copy_(q.roll(1, dims=-1))
    saved = indices.clone()
    indices[:, :2049].copy_(saved[:, :2049].roll(1, dims=1))
    assert torch.equal(indices[:, 2049:], saved[:, 2049:])
    graph.replay()
    torch.testing.assert_close(output.float(), reference(), atol=0.012, rtol=0.015)


def test_glm53_direct_decode_factory_bound_guard():
    backend = object.__new__(Fp8Glm5NextSparseAttBackend)
    backend.model = SimpleNamespace(config={"index_topk": 2048, "index_kpool": 4})
    assert backend.create_att_decode_state(SimpleNamespace()).decode_topk_bound == 2051
    backend.model = SimpleNamespace(config={"index_topk": 2048, "index_kpool": 8})
    assert backend.create_att_decode_state(SimpleNamespace()).decode_topk_bound is None
    backend.model = SimpleNamespace(config={"index_topk": 2047, "index_kpool": 4})
    assert backend.create_att_decode_state(SimpleNamespace()).decode_topk_bound is None


@pytest.mark.parametrize("batch", [1, 8])
def test_glm53_direct_decode_actual_entry_edges_and_graph(monkeypatch, batch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53200 + batch)
    capacity = 4096
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    destindex_copy_kv_flashmla_fp8(source, mem, packed)
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
    state = Fp8Glm5NextSparseDecodeState(
        infer_state=SimpleNamespace(is_cuda_graph=False),
        flashmla_sched_meta=flash_mla.get_mla_metadata()[0],
        decode_topk_bound=2051,
    )

    def native_must_not_run(*args, **kwargs):
        raise AssertionError("supported GLM-5.3 decode took native FlashMLA fallback")

    monkeypatch.setattr(flash_mla, "flash_mla_with_kvcache", native_must_not_run)

    def run():
        return state.decode_att((q, q[..., :0]), packed[..., :584], None, control)

    expected = _torch_sparse_attention(q, _dequantize_kv_cache(packed), indices)
    eager = run()
    torch.testing.assert_close(eager.float(), expected, atol=0.02, rtol=0.02)
    if batch == 1:
        assert torch.count_nonzero(eager) > 0
    state.infer_state.is_cuda_graph = True
    graph, output = _capture_cuda_graph(run)
    state.infer_state.is_cuda_graph = False
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


@pytest.mark.parametrize("batch", [2, 4])
def test_glm53_direct_decode_small_batch_graph_only_factory_entry(monkeypatch, batch):
    """B2/B4 stay native in eager mode and take the direct route only during graph capture."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53300 + batch)
    capacity = 4096
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    destindex_copy_kv_flashmla_fp8(source, mem, packed)
    # BMM-style B/H stride: (512, batch * 512, 1).
    q = torch.randn(8, batch, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    assert q.stride() == (512, batch * 512, 1)
    indices = torch.full((batch, 2176), -1, device="cuda", dtype=torch.int32)
    counts = [2051, 2048] if batch == 2 else [0, 2048, 2050, 2051]
    for row, count in enumerate(counts):
        if count:
            indices[row, :count] = torch.randint(0, capacity, (count,), device="cuda", dtype=torch.int32)
    if batch == 4:
        indices[2, :32] = -1  # holes
        indices[3, 64:128] = indices[3, :64]  # duplicates
    assert torch.equal(indices[:, 2051:], torch.full_like(indices[:, 2051:], -1))
    control = AttControl(
        nsa_decode=True,
        nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512},
    )
    backend = object.__new__(Fp8Glm5NextSparseAttBackend)
    backend.model = SimpleNamespace(config={"index_topk": 2048, "index_kpool": 4})
    state = backend.create_att_decode_state(SimpleNamespace(is_cuda_graph=False))
    state.flashmla_sched_meta = flash_mla.get_mla_metadata()[0]
    assert state.decode_topk_bound == 2051
    native_state = Fp8Glm5NextSparseDecodeState(
        infer_state=SimpleNamespace(is_cuda_graph=False),
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
        raise AssertionError("B2/B4 graph capture did not use the direct packed-KV graph path")

    monkeypatch.setattr(flash_mla, "flash_mla_with_kvcache", native_must_not_run)
    state.infer_state.is_cuda_graph = True
    graph, output = _capture_cuda_graph(lambda: run(state))
    state.infer_state.is_cuda_graph = False
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


def test_glm53_direct_decode_unknown_and_dstride_fallback(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53250)
    capacity = 4096
    source = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((capacity, 1, 720), device="cuda", dtype=torch.uint8)
    mem = torch.arange(capacity, device="cuda", dtype=torch.int32)
    destindex_copy_kv_flashmla_fp8(source, mem, packed)
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
    unknown = Fp8Glm5NextSparseDecodeState(
        infer_state=SimpleNamespace(is_cuda_graph=False), flashmla_sched_meta=flash_mla.get_mla_metadata()[0]
    )
    actual = unknown.decode_att((q, q[..., :0]), packed[..., :584], None, control)
    torch.testing.assert_close(
        actual.float(), _torch_sparse_attention(q, _dequantize_kv_cache(packed), late), atol=0.02, rtol=0.02
    )
    assert calls and not _can_use_glm53_direct_decode(q, packed[..., :584], late.unsqueeze(1), 0.0625, 512, None)
    q_d2 = torch.randn(1, 8, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2]
    good = torch.full((1, 2176), -1, device="cuda", dtype=torch.int32)
    good[:, :2051] = torch.randint(0, capacity, (1, 2051), device="cuda", dtype=torch.int32)
    assert not _can_use_glm53_direct_decode(q_d2, packed[..., :584], good.unsqueeze(1), 0.0625, 512, 2051)
    assert not _can_use_glm53_direct_decode(q, packed[..., :584], good.unsqueeze(1), 0.0625, 256, 2051)
