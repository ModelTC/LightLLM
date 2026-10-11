"""Production `_c4_indices` CUDA integration regression; run after services stop."""
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(
    not __import__("torch").cuda.is_available() or __import__("torch").cuda.get_device_capability()[0] < 9,
    reason="requires SM90+ CUDA FP8",
)


def _pack_pagewise_kv(torch, value):
    pages, block, heads, dim = value.shape
    assert heads == 1
    amax = value.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    scale = amax / 448.0
    quantized = (value / scale).to(torch.float8_e4m3fn)
    packed = torch.empty((pages, block * (dim + 4)), device=value.device, dtype=torch.uint8)
    packed[:, : block * dim] = quantized.reshape(pages, block * dim).view(torch.uint8)
    packed[:, block * dim :] = scale.reshape(pages, block).view(torch.uint8)
    return packed.view(pages, block, heads, dim + 4)


class _Workspace:
    def __init__(self, torch):
        self.torch = torch
        self._slots = None
        self._lengths = None

    def c4(self, _microbatch, rows, topk):
        # Deliberately reuse backing storage, matching production workspace behavior.
        if self._slots is None or self._slots.shape != (rows, topk):
            self._slots = self.torch.empty((rows, topk), device="cuda", dtype=self.torch.int32)
            self._lengths = self.torch.empty((rows,), device="cuda", dtype=self.torch.int32)
        return self._slots, self._lengths


class _C4Pool:
    def __init__(self, page_size, kv):
        self.page_size = page_size
        self.num_pages = kv.shape[0]
        self._kv = kv

    def get_layer_buffer(self, _layer):
        return self._kv


def _make_state(torch, kv, seq, ready, plan):
    q_lengths = [s - r for s, r in zip(seq, ready)]
    rows = sum(q_lengths)
    positions = torch.cat([r + torch.arange(q, device="cuda", dtype=torch.int64) for r, q in zip(ready, q_lengths)])
    token_to_batch = torch.repeat_interleave(
        torch.arange(len(seq), device="cuda", dtype=torch.long), torch.tensor(q_lengths, device="cuda")
    )
    pool = _C4Pool(64, kv)
    manager = SimpleNamespace(c4_indexer_pool=pool, layer_to_c4_idx={0: 0})
    return SimpleNamespace(
        mem_manager=manager,
        dsv4_workspace=_Workspace(torch),
        microbatch_index=0,
        max_kv_seq_len=3072,
        is_prefill=True,
        _dsv4_token_to_batch_idx=token_to_batch,
        b_req_idx=torch.arange(len(seq), device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor(seq, device="cuda", dtype=torch.int32),
        req_manager=SimpleNamespace(req_to_token_indexs=None, HOLD_REQUEST_ID=-1),
        _c4_paged_meta=None,
        _c4_paged_n1_meta=None,
        _c4_paged_pair_meta=None,
        dsv4_c4_ragged_pair_plan=plan,
        _c4_ragged_pair_gpu=None,
        _c4_ragged_pair_workspace=None,
        _c4_ragged_pair_metadata=None,
        positions=positions,
        rows=rows,
    )


def _make_indexer(torch, cls):
    indexer = object.__new__(cls)
    indexer.layer_idx_ = 0
    indexer.index_topk = 512
    indexer.index_head_dim = 128
    indexer.index_n_heads = 64
    indexer._dsv4_prefill_role = False
    indexer.alloc_tensor = lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device)
    return indexer


def _clone_output(torch, result):
    slots, lengths = result
    return slots.clone(), lengths.clone()


def _assert_sorted_slots(torch, lhs, rhs):
    assert torch.equal(torch.sort(lhs, dim=-1).values, torch.sort(rhs, dim=-1).values)


def test_production_c4_indices_pair_roles_capture_error_and_chunking(monkeypatch):
    import torch
    import deep_gemm
    import lightllm.common.basemodel  # initialize shared kernel imports before the DS4 layer module
    import lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer as infer_mod
    from lightllm.models.deepseek_v4.c4_ragged_pair import build_c4_ragged_pair_plan
    from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4IndexInfer
    import lightllm.models.deepseek_v4.triton_kernel.gather_c4_indexer_k_dsv4 as gather_mod

    free, _ = torch.cuda.mem_get_info()
    if free < 256 * 1024**2:
        pytest.skip("need ≥256 MiB free GPU memory")
    torch.manual_seed(20261011)
    h, d, page, cap = 64, 128, 64, 768
    seq, ready = [2689, 2947], [2560, 2816]  # q=[129,131], T=260
    plan = build_c4_ragged_pair_plan(seq, ready, 260)
    assert plan is not None and plan.kind == "ragged"
    all_even = build_c4_ragged_pair_plan([2690, 2946], ready, 260)
    assert all_even is not None and all_even.kind == "all_even"
    kv = _pack_pagewise_kv(torch, torch.randn((24, page, 1, d), device="cuda", dtype=torch.bfloat16))
    q = torch.randn((260, h, d), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    weights = torch.rand((260, h), device="cuda", dtype=torch.float32).add_(0.25)
    page_table = torch.stack(
        (torch.arange(12, device="cuda", dtype=torch.int32), torch.arange(12, 24, device="cuda", dtype=torch.int32))
    )

    def fake_pages(_manager, _reqs, _lengths, _cap, _token_indexes, _hold, out=None):
        if out is not None:
            out.copy_(page_table)
            return out
        return page_table

    metadata_calls = []
    logits_calls = []
    real_metadata = deep_gemm.get_paged_mqa_logits_metadata
    real_logits = deep_gemm.fp8_paged_mqa_logits

    def metadata_spy(ctx, *args, **kwargs):
        metadata_calls.append(tuple(ctx.shape))
        return real_metadata(ctx, *args, **kwargs)

    def logits_spy(normalized_q, *args, **kwargs):
        result = real_logits(normalized_q, *args, **kwargs)
        logits_calls.append(
            {
                "n": normalized_q.shape[1],
                "logits": result.clone(),
                "returned_ptr": result.data_ptr(),
                "out": kwargs.get("out"),
            }
        )
        return result

    monkeypatch.setattr(gather_mod, "build_c4_indexer_page_table", fake_pages)
    monkeypatch.setattr(deep_gemm, "get_paged_mqa_logits_metadata", metadata_spy)
    monkeypatch.setattr(deep_gemm, "fp8_paged_mqa_logits", logits_spy)
    indexer = _make_indexer(torch, DeepseekV4IndexInfer)

    # Clone immediately: both calls receive workspace-backed production tensors.
    n1_state = _make_state(torch, kv, seq, ready, None)
    n1_slots, n1_lengths = _clone_output(torch, indexer._c4_indices(n1_state, q, weights, n1_state.positions))
    n1_logits = [call["logits"] for call in logits_calls if call["n"] == 1][-1].squeeze(1)
    before_pair_n1 = sum(shape[1] == 1 for shape in metadata_calls)
    pair_state = _make_state(torch, kv, seq, ready, plan)
    indexer._dsv4_prefill_role = True
    pair_slots, pair_lengths = _clone_output(torch, indexer._c4_indices(pair_state, q, weights, pair_state.positions))
    assert sum(shape[1] == 1 for shape in metadata_calls) == before_pair_n1
    pair_logits = [call["logits"] for call in logits_calls if call["n"] == 2][-1].reshape(262, cap)
    original_to_packed = torch.tensor(plan.original_to_packed, device="cuda", dtype=torch.long)
    pair_logits = pair_logits.index_select(0, original_to_packed)
    valid = ((n1_state.positions + 1) // 4).to(torch.int32)
    for row, length in enumerate(valid.cpu().tolist()):
        torch.testing.assert_close(n1_logits[row, :length], pair_logits[row, :length], atol=1e-3, rtol=1e-3)
    _assert_sorted_slots(torch, n1_slots, pair_slots)
    assert torch.equal(n1_lengths, pair_lengths)

    # P capture must error; D-prefill is a legitimate N=1 mode using the same cached state.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    try:
        indexer._c4_indices(pair_state, q, weights, pair_state.positions)
    except RuntimeError:
        pass
    else:
        raise AssertionError("P capture must not silently select N=1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    indexer._dsv4_prefill_role = False
    d_slots, d_lengths = _clone_output(torch, indexer._c4_indices(pair_state, q, weights, pair_state.positions))
    _assert_sorted_slots(torch, n1_slots, d_slots)
    assert torch.equal(n1_lengths, d_lengths)
    indexer._dsv4_prefill_role = True

    # Force a 128-row pair budget. Chunks cross the q=129 request boundary; pairs remain request-local.
    monkeypatch.setattr(infer_mod, "C4_PREFILL_LOGITS_BUDGET_BYTES", ((cap + 63) // 64 * 64) * 4 * 128)
    chunked_state = _make_state(torch, kv, seq, ready, plan)
    chunked_slots, chunked_lengths = _clone_output(
        torch, indexer._c4_indices(chunked_state, q, weights, chunked_state.positions)
    )
    _assert_sorted_slots(torch, n1_slots, chunked_slots)
    assert torch.equal(n1_lengths, chunked_lengths)
    owners = [0] * 129 + [1] * 131
    for left, right in zip(plan.packed_to_original[::2], plan.packed_to_original[1::2]):
        assert owners[left] == owners[right]

    # Direct helper: supplied output buffer must be passed to DeepGEMM and all-even also matches N=1.
    out = torch.full((260, cap), float("nan"), device="cuda", dtype=torch.float32)
    ctx = torch.clamp(valid, min=1).reshape(-1, 1)
    meta = real_metadata(ctx, page, deep_gemm.get_num_sms())
    slots = torch.empty((260, 512), device="cuda", dtype=torch.int32)
    per_row_pages = page_table.repeat_interleave(torch.tensor([129, 131], device="cuda"), dim=0)
    indexer._c4_score_topk(
        q.unsqueeze(1), kv, weights, ctx, per_row_pages, meta, cap, valid, per_row_pages, slots, page, logits_out=out
    )
    helper_call = logits_calls[-1]
    assert helper_call["out"] is out and helper_call["returned_ptr"] == out.data_ptr()
    for row, length in enumerate(valid.cpu().tolist()):
        assert torch.isfinite(out[row, :length]).all()
    _assert_sorted_slots(torch, n1_slots.squeeze(1), slots)

    # All-even production plan and the same forced budget must agree with its N=1 fallback.
    even_seq = [2690, 2946]
    even_q = q.clone()
    even_w = weights.clone()
    even_n1 = _make_state(torch, kv, even_seq, ready, None)
    even_pair = _make_state(torch, kv, even_seq, ready, all_even)
    indexer._dsv4_prefill_role = False
    expected_even = _clone_output(torch, indexer._c4_indices(even_n1, even_q, even_w, even_n1.positions))
    indexer._dsv4_prefill_role = True
    observed_even = _clone_output(torch, indexer._c4_indices(even_pair, even_q, even_w, even_pair.positions))
    _assert_sorted_slots(torch, expected_even[0], observed_even[0])
    assert torch.equal(expected_even[1], observed_even[1])

    # Removed min-token/high-padding gates: actual P N=2 also handles T=2 and T=3.
    for short_seq, short_ready, token_count in (([2561, 2817], [2560, 2816], 2), ([2561, 2818], [2560, 2816], 3)):
        short_plan = build_c4_ragged_pair_plan(short_seq, short_ready, token_count)
        assert short_plan.kind == "ragged"
        short_q, short_w = q[:token_count].contiguous(), weights[:token_count].contiguous()
        indexer._dsv4_prefill_role = False
        short_n1 = _make_state(torch, kv, short_seq, short_ready, None)
        expected_short = _clone_output(torch, indexer._c4_indices(short_n1, short_q, short_w, short_n1.positions))
        before_n2 = sum(call["n"] == 2 for call in logits_calls)
        indexer._dsv4_prefill_role = True
        short_pair = _make_state(torch, kv, short_seq, short_ready, short_plan)
        observed_short = _clone_output(torch, indexer._c4_indices(short_pair, short_q, short_w, short_pair.positions))
        assert sum(call["n"] == 2 for call in logits_calls) > before_n2
        _assert_sorted_slots(torch, expected_short[0], observed_short[0])
        assert torch.equal(expected_short[1], observed_short[1])

    # Actual D decode: two last-token rows use page-table rows directly and match D-prefill N=1.
    decode_seq, decode_ready = [2689, 2947], [2688, 2946]
    decode_q = q.index_select(0, torch.tensor([128, 259], device="cuda"))
    decode_w = weights.index_select(0, torch.tensor([128, 259], device="cuda"))
    indexer._dsv4_prefill_role = False
    d_prefill = _make_state(torch, kv, decode_seq, decode_ready, None)
    reference_decode = _clone_output(torch, indexer._c4_indices(d_prefill, decode_q, decode_w, d_prefill.positions))
    d_decode = _make_state(torch, kv, decode_seq, decode_ready, None)
    d_decode.is_prefill = False
    observed_decode = _clone_output(torch, indexer._c4_indices(d_decode, decode_q, decode_w, d_decode.positions))
    _assert_sorted_slots(torch, reference_decode[0], observed_decode[0])
    assert torch.equal(reference_decode[1], observed_decode[1])
