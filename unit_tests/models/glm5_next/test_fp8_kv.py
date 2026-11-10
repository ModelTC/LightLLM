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
from lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_write import write_fp8_e8m0_packed_kv
from lightllm.models.glm5_next.triton_kernel import context_flashattention_nopad_fp8 as fp8_prefill
from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
    _gather_prefill_kv_kernel,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_infer_state(**attributes):
    state = InferStateInfo()
    vars(state).update(attributes)
    return state


def _make_workspace_backend(batch_max_tokens):
    backend = object.__new__(Fp8Glm5NextSparseAttBackend)
    backend.model = SimpleNamespace(
        batch_max_tokens=batch_max_tokens, prefill_graph=None, config={"index_topk": 2048, "index_kpool": 4}
    )
    backend._prefill_workspaces = [{"direct": None, "generic": None} for _ in range(2)]
    return backend


class _PrefillCase:
    def __init__(self, seq_lens, q_lens, heads, width=2176, capacity=None, backend=None, softmax_scale=0.0625):
        device = "cuda"
        tensor = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
        self.seq_lens, self.q_lens = seq_lens, q_lens
        self.requests = list(range(2 * len(seq_lens) - 1, 0, -2))
        total, queries = sum(seq_lens), sum(q_lens)
        capacity = capacity or total + 17
        self.req_to_token_indexs = torch.full(
            (2 * len(seq_lens) + 1, max(seq_lens) + 7), -1, device=device, dtype=torch.int32
        )
        slots = torch.randperm(capacity, device=device, dtype=torch.int32)[:total]
        cu_q, cu_kv, q_req_idx, positions = [0], [0], [], []
        for req, length, count in zip(self.requests, seq_lens, q_lens):
            self.req_to_token_indexs[req, :length] = slots[cu_kv[-1] : cu_kv[-1] + length]
            q_req_idx.extend([req] * count)
            positions.extend(range(length - count, length))
            cu_q.append(cu_q[-1] + count)
            cu_kv.append(cu_kv[-1] + length)
        self.q_req_idx, self.position_ids = tensor(q_req_idx).long(), tensor(positions).long()
        self.q = torch.randn(queries, heads, 512, device=device, dtype=torch.bfloat16)
        self.source_kv = torch.randn(capacity, 1, 512, device=device, dtype=torch.bfloat16)
        self.packed_kv = torch.zeros(capacity, 1, 720, device=device, dtype=torch.uint8)
        self.all_slots = torch.arange(capacity, device=device, dtype=torch.int32)
        write_fp8_e8m0_packed_kv(self.source_kv, self.all_slots, self.packed_kv)
        self.prefill_cache_kv = torch.randn(queries, 1, 1024, device=device, dtype=torch.bfloat16)[..., ::2]
        visible = tensor(positions) + 1
        self.topk_indices = torch.randint(max(seq_lens), (queries, width), device=device, dtype=torch.int32)
        self.topk_indices.remainder_(visible[:, None])
        self.topk_indices.masked_fill_(
            torch.arange(width, device=device)[None, :] >= visible.clamp_max(2048)[:, None], -1
        )
        self.topk_indices[:, ::19] = -1
        self.topk_indices[:, 0] = visible - 1
        self.topk_indices[:, 1] = visible - 1
        self.topk_mem_indices = torch.empty(queries, width * 2, device=device, dtype=torch.int32)[:, ::2]
        self.infer_state = _make_infer_state(
            is_prefill=True,
            req_manager=SimpleNamespace(req_to_token_indexs=self.req_to_token_indexs),
            b_req_idx=tensor(self.requests),
            b_seq_len=tensor(seq_lens),
            b1_cu_q_seq_len=tensor(cu_q),
            b1_cu_kv_seq_len=tensor(cu_kv),
            position_ids=tensor(positions),
            total_token_num=total,
            mem_index=torch.empty(queries, device=device, dtype=torch.int32),
        )
        self.backend = backend or _make_workspace_backend(queries)
        _, self.att_state = self.new_state()
        self.control = AttControl(
            nsa_prefill=True,
            nsa_prefill_dict={
                "topk_indices": self.topk_indices,
                "topk_mem_indices": self.topk_mem_indices,
                "prefill_cache_kv": self.prefill_cache_kv,
                "softmax_scale": softmax_scale,
                "kv_lora_rank": 512,
            },
        )

    def new_state(self, **overrides):
        if overrides:
            infer = _make_infer_state(**vars(self.infer_state))
            vars(infer).update(overrides)
        else:
            infer = self.infer_state
        state = self.backend.create_att_prefill_state(infer)
        state.init_state()
        return infer, state

    def refresh(self):
        self.infer_state.mem_index.copy_(self.req_to_token_indexs[self.q_req_idx, self.position_ids])
        self.topk_mem_indices.copy_(
            self.req_to_token_indexs[self.q_req_idx[:, None], self.topk_indices.clamp_min(0).long()]
        )
        self.topk_mem_indices.masked_fill_(self.topk_indices < 0, -1)
        self.control.nsa_prefill_dict["topk_indices"] = self.topk_indices

    def run(self, state=None):
        self.refresh()
        active = self.att_state if state is None else state
        return active.prefill_att(self.q, self.packed_kv[..., :584], None, self.control)

    def check(self, actual):
        values = _dequantize_kv_cache(self.packed_kv)
        values[self.infer_state.mem_index.long()] = self.prefill_cache_kv
        physical = self.req_to_token_indexs[self.q_req_idx[:, None], self.topk_indices.clamp_min(0).long()].masked_fill(
            self.topk_indices < 0, -1
        )
        expected = _torch_sparse_attention(self.q, values, physical, self.control.nsa_prefill_dict["softmax_scale"])
        torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.015)

    def mutate(self):
        self.q.normal_()
        self.prefill_cache_kv.normal_()
        self.source_kv.normal_()
        write_fp8_e8m0_packed_kv(self.source_kv, self.all_slots, self.packed_kv)
        for req, length in zip(self.requests, self.seq_lens):
            self.req_to_token_indexs[req, :length].copy_(self.req_to_token_indexs[req, :length].roll(1))
        self.topk_indices.copy_(self.topk_indices.roll(1, dims=1))


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

    prefill_kv = source[8:].clone().add_(0.25)
    gathered_kv = torch.empty((8, 1, 512), device="cuda", dtype=torch.bfloat16)
    _gather_prefill_kv_kernel[(3,)](
        packed_kv[..., :584],
        prefill_kv,
        gathered_kv,
        mem_index[None],
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.tensor([0, 11], device="cuda", dtype=torch.int32),
        torch.tensor([0, 3], device="cuda", dtype=torch.int32),
        2,
        8,
        packed_kv.stride(0),
        prefill_kv.stride(0),
        prefill_kv.stride(2),
        req_to_token_indexs_stride_b=11,
        req_to_token_indexs_stride_s=1,
        batch_size=1,
        BLOCK_ROWS=4,
    )
    expected = _dequantize_kv_cache(packed_kv)
    expected[mem_index[8:].long()] = prefill_kv
    assert torch.equal(gathered_kv, expected[mem_index[2:10].long()])


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
    prefix = 4 if cached else 0
    case = _PrefillCase([prefix + 3] * 2, [3, 3], num_heads, width=128, capacity=64, softmax_scale=512 ** -0.5)
    case.q = torch.randn(num_heads, 6, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    case.topk_indices[0].fill_(-1)
    case.topk_indices[1, :2] = prefix
    case.topk_indices[3].fill_(-1)
    case.topk_indices[3, 0] = prefix
    case.check(case.run())
    graph, output = _capture_cuda_graph(case.run)
    graph.replay()
    case.check(output)
    case.mutate()
    graph.replay()
    case.check(output)


@pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
@pytest.mark.parametrize("softmax_scale", [0.0625, 512 ** -0.5])
def test_fp8_chunked_prefill_sparse_rows_and_graph(monkeypatch, num_heads, softmax_scale):
    monkeypatch.setattr(fp8_prefill, "KV_GATHER_CHUNK_SIZE", 129)
    torch.manual_seed(53)
    case = _PrefillCase([1025], [9], num_heads, capacity=2048, softmax_scale=softmax_scale)
    lengths = torch.tensor([0, 1, 3, 127, 128, 511, 2048, 2049, 2051], device="cuda")
    case.topk_indices.masked_fill_(torch.arange(2176, device="cuda")[None, :] >= lengths[:, None], -1)
    case.topk_indices[1, 0] = 1016
    case.topk_indices[2, :3] = torch.tensor([2, 3, 4], device="cuda")
    case.topk_indices[3, :127] = torch.randint(1016, 1025, (127,), device="cuda", dtype=torch.int32)
    case.check(case.run())
    graph, output = _capture_cuda_graph(case.run)
    graph.replay()
    case.check(output)
    case.mutate()
    graph.replay()
    case.check(output)


@pytest.mark.parametrize("num_q", [64, 129])
def test_fp8_h64_largeq_native_prefill_request_table_graph(monkeypatch, num_q):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse prefill target is Hopper")
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    torch.manual_seed(53100 + num_q)
    case = _PrefillCase([32768, 32768], [num_q // 2, num_q - num_q // 2], 64, capacity=131072)
    observed = []

    def observed_sparse(*args, **kwargs):
        observed.append((kwargs["q"].shape, kwargs["kv"].shape, kwargs["indices"].shape))
        return flash_mla_sparse_fwd(*args, **kwargs)

    monkeypatch.setattr("sgl_kernel.flash_mla.flash_mla_sparse_fwd", observed_sparse)
    case.check(case.run())
    assert observed == [((num_q, 64, 512), (65536, 1, 512), (num_q, 1, 2176))]
    graph, output = _capture_cuda_graph(case.run)
    case.mutate()
    graph.replay()
    case.check(output)
    case.topk_indices.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(output) == 0
    assert len(observed) == 3


def test_fp8_h8_largeq_prefill_keeps_flashattention_path(monkeypatch):
    import sgl_kernel.flash_attn as flash_attn

    torch.manual_seed(53180)
    case = _PrefillCase([65536], [16], 8, capacity=65536)
    observed = []
    original = flash_attn.flash_attn_with_kvcache

    def observed_flashattention(*args, **kwargs):
        observed.append(True)
        return original(*args, **kwargs)

    def unexpected_native(*args, **kwargs):
        raise AssertionError("H8 large-Q prefill entered the native H64 sparse path")

    monkeypatch.setattr(flash_attn, "flash_attn_with_kvcache", observed_flashattention)
    monkeypatch.setattr("sgl_kernel.flash_mla.flash_mla_sparse_fwd", unexpected_native)
    case.check(case.run())
    assert observed


@pytest.mark.parametrize(
    "num_q,num_heads",
    [(1, 8), (2, 8), (8, 8), (1, 64), (2, 64), (3, 64), (4, 64), (5, 64), (6, 64), (7, 64), (8, 64)],
)
def test_fp8_direct_prefill_dispatch_strided_holes_and_current(num_q, num_heads):
    torch.manual_seed(53016 + num_q)
    seqs = [65536 + value for value in [127, 521, 1031, 2053, 4099, 8191, 12289, 16381][:num_q]]
    case = _PrefillCase(seqs, [1] * num_q, num_heads)
    case.q = torch.randn(num_heads, num_q, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    backing = torch.full((num_q, 4352), -1, device="cuda", dtype=torch.int32)
    backing[:, ::2] = case.topk_indices
    case.topk_indices = backing[:, ::2]
    output = case.run()
    case.check(output)
    assert output.is_contiguous() and torch.count_nonzero(output) > 0
    graph, output = _capture_cuda_graph(case.run)
    graph.replay()
    case.check(output)
    case.mutate()
    graph.replay()
    case.check(output)


def test_fp8_h64_direct_prefill_state_workspace_reuse_and_fresh_state():
    torch.manual_seed(53150)
    case = _PrefillCase([65536], [1], 64)
    case.check(case.run())
    workspace = case.backend._prefill_workspaces[0]["direct"]
    pointers = {name: value.data_ptr() for name, value in workspace.items()}
    for _ in range(10):
        infer, state = case.new_state()
        case.check(case.run(state))
    assert {name: value.data_ptr() for name, value in workspace.items()} == pointers
    fresh_infer, fresh = case.new_state(mem_index=case.infer_state.mem_index.clone())
    case.check(case.run(fresh))
    assert {name: value.data_ptr() for name, value in workspace.items()} == pointers
    case.mutate()
    case.check(case.run())


@pytest.mark.parametrize("num_q", [2, 3, 5, 8, 16, 32, 128])
def test_fp8_h64_selected_prefill_cached_chunks_and_graph(num_q):
    torch.manual_seed(53200 + num_q)
    case = _PrefillCase([131072, 131072], [num_q // 2, num_q - num_q // 2], 64)
    case.check(case.run())
    for _ in range(2):
        infer, state = case.new_state()
        case.check(case.run(state))
    fresh_infer, fresh = case.new_state(mem_index=case.infer_state.mem_index.clone())
    case.check(case.run(fresh))
    graph, output = _capture_cuda_graph(case.run)
    case.mutate()
    graph.replay()
    case.check(output)
    case.topk_indices.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(output) == 0


def test_fp8_selected_prefill_graph_with_empty_requests_and_changing_boundaries():
    case = _PrefillCase([65536] * 4, [0, 4, 0, 3], 64)
    case.check(case.run())
    graph, output = _capture_cuda_graph(case.run)
    replacement = _PrefillCase([65536] * 4, [2, 0, 5, 0], 64)
    for name in (
        "q",
        "packed_kv",
        "prefill_cache_kv",
        "req_to_token_indexs",
        "q_req_idx",
        "position_ids",
        "topk_indices",
        "topk_mem_indices",
    ):
        getattr(case, name).copy_(getattr(replacement, name))
    case.infer_state.b1_cu_q_seq_len.copy_(replacement.infer_state.b1_cu_q_seq_len)
    case.infer_state.position_ids.copy_(replacement.infer_state.position_ids)
    for name in ("q_kv_seq_len", "q_req_idx", "q_kv_start_loc"):
        getattr(case.att_state, name).copy_(getattr(replacement.att_state, name))
    graph.replay()
    case.check(output)


def test_fp8_one_query_prefill_and_decode_read_their_own_kv_sources():
    case = _PrefillCase([65536], [1], 64)
    case.q.zero_()
    case.prefill_cache_kv.fill_(1)
    case.source_kv.fill_(2)
    write_fp8_e8m0_packed_kv(case.source_kv, case.all_slots, case.packed_kv)
    case.topk_indices.fill_(-1)
    case.topk_indices[:, 0] = 65535
    case.refresh()
    decode_infer = _make_infer_state(**vars(case.infer_state))
    decode_infer.is_prefill = False
    decode_infer.is_cuda_graph_capture = True
    decode_state = case.backend.create_att_decode_state(decode_infer)
    decode_state.init_state()
    control = AttControl(
        nsa_decode=True,
        nsa_decode_dict={
            "topk_mem_indices": case.topk_mem_indices,
            "softmax_scale": 0.0625,
            "kv_lora_rank": 512,
        },
    )

    def run():
        prefill = case.run()
        decode = decode_state.decode_att((case.q, case.q[..., :0]), case.packed_kv[..., :584], None, control)
        return prefill, decode

    prefill, decode = run()
    torch.testing.assert_close(prefill, torch.ones_like(prefill), rtol=0, atol=0)
    torch.testing.assert_close(decode, torch.full_like(decode, 2), rtol=0, atol=0)
    graph, (prefill, decode) = _capture_cuda_graph(run)
    case.prefill_cache_kv.fill_(3)
    case.source_kv.fill_(4)
    write_fp8_e8m0_packed_kv(case.source_kv, case.all_slots, case.packed_kv)
    graph.replay()
    torch.testing.assert_close(prefill, torch.full_like(prefill, 3), rtol=0, atol=0)
    torch.testing.assert_close(decode, torch.full_like(decode, 4), rtol=0, atol=0)


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
    torch.manual_seed(53450)
    backend = _make_workspace_backend(66)
    case = _PrefillCase([32768, 16384, 16384], [11, 11, 11], 64, backend=backend)
    case.check(case.run())
    workspace = backend._prefill_workspaces[0]["generic"]
    assert set(workspace) == {"chunk_topk_indices", "chunk_cache_seqlens", "cu_seqlens_q"}
    pointers = {name: value.data_ptr() for name, value in workspace.items()}
    for _ in range(10):
        infer, state = case.new_state()
        case.check(case.run(state))
    case.mutate()
    case.check(case.run())
    fresh = _PrefillCase([16384, 32768, 16384], [9, 11, 13], 64, capacity=131072, backend=backend)
    fresh.check(fresh.run())
    assert {name: value.data_ptr() for name, value in backend._prefill_workspaces[0]["generic"].items()} == pointers
    for overrides in ({"prefill_cuda_graph_exe_list": []}, {"is_cuda_graph_capture": True}):
        infer, state = case.new_state(**overrides)
        case.check(case.run(state))
    graph, output = _capture_cuda_graph(case.run)
    case.mutate()
    graph.replay()
    case.check(output)


def test_fp8_generic_prefill_workspace_graph_query_is_generic_h64_only(monkeypatch):
    def forbidden():
        raise AssertionError("Non-generic prefill queried CUDA graph capture state")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden)
    monkeypatch.setattr(fp8_prefill, "context_attention_fwd_fp8", lambda **kwargs: kwargs["q"])
    params = {
        "prefill_cache_kv": torch.empty((1, 1, 512), dtype=torch.bfloat16),
        "topk_indices": torch.empty((1, 2176), dtype=torch.int32),
        "topk_mem_indices": torch.empty((1, 2176), dtype=torch.int32),
        "softmax_scale": 0.0625,
    }
    control = AttControl(nsa_prefill=True, nsa_prefill_dict=params)
    for heads, direct, selected in ((8, False, False), (64, True, False), (64, False, True)):
        monkeypatch.setattr(fp8_prefill, "can_use_direct_prefill", lambda *args, d=direct: d)
        monkeypatch.setattr(fp8_prefill, "can_use_selected_prefill", lambda *args, s=selected: s)
        infer = _make_infer_state(
            mem_index=torch.empty(1, dtype=torch.int32),
            req_manager=SimpleNamespace(req_to_token_indexs=torch.empty((1, 1), dtype=torch.int32)),
        )
        backend = _make_workspace_backend(1)
        backend.get_direct_prefill_workspace = lambda *args: {}
        state = Fp8Glm5NextSparsePrefillState(backend=backend, _infer_state_weak_ref=weakref.ref(infer))
        assert (
            state._nsa_prefill_att(torch.empty((1, heads, 512), dtype=torch.bfloat16), None, control).shape[1] == heads
        )


def test_fp8_state_preserves_kv_and_indexer_across_cpu_and_pd(monkeypatch):
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.attention_state.full_att import PackedMLALogicalConfig, create_packed_mla_state

    args = SimpleNamespace(cpu_cache_token_page_size=4, pd_kv_page_size=4)
    monkeypatch.setattr("lightllm.common.attention_state.manager.get_env_start_args", lambda: args)
    monkeypatch.setattr("lightllm.common.attention_state.cpu_cache_operator.get_env_start_args", lambda: args)
    monkeypatch.setattr("lightllm.common.attention_state.pd_operator.get_env_start_args", lambda: args)
    monkeypatch.setattr(
        "lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_state.get_env_start_args", lambda: args
    )
    config = PackedMLALogicalConfig(
        logical_layer_groups=((3,),),
        latent_dim=512,
        format="packed_group64_e8m0",
    )
    state = create_packed_mla_state(config=config, page_size=4)
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
    cpu_layout = manager.get_cpu_cache_operator().get_page_layouts()[state.state_name]
    cpu_pages = torch.empty((1, manager.get_cpu_cache_operator().get_page_bytes()), dtype=torch.uint8, pin_memory=True)
    state.write_to_cpu(token_indices=indices, cpu_page_buffers=cpu_pages, cpu_page_indices=[0], layout=cpu_layout)
    state.read_from_cpu(cpu_page_buffers=cpu_pages, cpu_page_indices=[0], token_indices=indices + 4, layout=cpu_layout)
    pd_layout = manager.get_pd_operator().get_pd_full_att_page_layouts()[state.state_name]
    pd_pages = torch.empty((1, manager.get_pd_operator().get_page_bytes()), dtype=torch.uint8, device="cuda")
    state.write_to_pd_page(token_indices=indices, pd_page_buffers=pd_pages, pd_page_index=0, layout=pd_layout)
    state.read_from_pd_page(pd_page_buffers=pd_pages, token_indices=indices + 8, pd_page_index=0, layout=pd_layout)
    torch.cuda.synchronize()
    for begin in (4, 8):
        actual = state.get_layer_view(logical_layer_id=3)
        torch.testing.assert_close(actual.kv[begin : begin + 4], actual.kv[:4], rtol=0, atol=0)
        torch.testing.assert_close(actual.indexer_k[begin : begin + 4], actual.indexer_k[:4], rtol=0, atol=0)
