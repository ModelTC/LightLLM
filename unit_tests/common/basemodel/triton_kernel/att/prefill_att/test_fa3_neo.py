"""Unit test for the FA3-based prefill path with image-token support.

This test pre-wires a call to ``flash_attn_with_kvcache`` with an
``image_token_tag`` keyword argument. The expectation is that ``fa3-neo``'s
``flash_attn_with_kvcache`` will be extended with an optional
``image_token_tag`` parameter that, for queries flagged as image tokens,
relaxes the causal mask so they can attend bidirectionally to every real key
in the request.

Torch reference expresses the *semantics* of the attention, not FA3's internal
tiling — it has no notion of BLOCK_N / BLOCK_M. For each batch element we
gather K/V for the whole request (prompt + new tokens) and apply::

    allow[m, k] = (k <= q_pos[m]) OR image_tag[m]          for k in [0, total)

i.e. normal queries are causal, image-token queries can see every real key in
the request. If FA3 disagrees with this reference, the kernel is wrong.

Run directly for quick debugging:

    python unit_tests/common/basemodel/triton_kernel/att/prefill_att/\
        test_fa3_neo.py

or via pytest:

    pytest unit_tests/common/basemodel/triton_kernel/att/prefill_att/\
        test_fa3_neo.py -x -s
"""

import math
import pytest
import torch

from flash_attn_interface import flash_attn_with_kvcache

try:
    import triton
    import triton.testing as triton_testing
except ImportError:
    triton = None
    triton_testing = None

try:
    from lightllm.models.neo_chat_moe.triton_kernel.context_attention_fwd_neo import (
        context_attention_fwd_neo,
    )
except ImportError:
    context_attention_fwd_neo = None


def torch_reference_context_attention_neo(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_q_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_prompt_cache_len: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    b_image_token_tag: torch.Tensor,
) -> torch.Tensor:
    device = q.device
    dtype = q.dtype
    _, Hq, D = q.shape
    Hk = k.shape[1]
    kv_group = Hq // Hk
    scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)

    for b in range(b_seq_len.shape[0]):
        req_idx = int(b_req_idx[b].item())
        seq_len = int(b_seq_len[b].item())
        prompt_cache_len = int(b_prompt_cache_len[b].item())
        q_seq_len = seq_len - prompt_cache_len
        if q_seq_len <= 0:
            continue

        q_start = int(b_q_start_loc[b].item())
        q_blk = q[q_start : q_start + q_seq_len]  # [M, Hq, D]
        image_tag = b_image_token_tag[q_start : q_start + q_seq_len].to(torch.bool)

        token_locs = req_to_token_indexs[req_idx, :seq_len].to(torch.int64)
        k_blk = k[token_locs]  # [seq_len, Hk, D]
        v_blk = v[token_locs]

        q_pos = torch.arange(prompt_cache_len, seq_len, device=device, dtype=torch.int64)  # [M]
        k_pos = torch.arange(0, seq_len, device=device, dtype=torch.int64)  # [seq_len]
        causal = k_pos[None, :] <= q_pos[:, None]
        allow = causal | image_tag[:, None]

        out_blk = torch.empty_like(q_blk)
        for h in range(Hq):
            h_k = h // kv_group
            q_h = q_blk[:, h, :].to(torch.float32)
            k_h = k_blk[:, h_k, :].to(torch.float32)
            v_h = v_blk[:, h_k, :].to(torch.float32)

            scores = (q_h @ k_h.transpose(0, 1)) * scale
            scores = torch.where(allow, scores, torch.full_like(scores, -1.0e8))
            probs = torch.softmax(scores, dim=-1)
            out_h = (probs @ v_h).to(dtype)
            out_blk[:, h, :] = out_h

        out[q_start : q_start + q_seq_len] = out_blk

    return out


def _build_inputs(
    batch: int,
    Hq: int,
    Hk: int,
    D: int,
    dtype: torch.dtype,
    device: str,
    max_q_seq_len: int = 256,
    max_prompt_cache_len: int = 512,
    image_prob: float = 0.7,
    num_image_spans_max: int = 3,
    image_span_len_max: int = 24,
    kv_pool_slack: int = 4096,
    seed: int = 0,
):
    """Build one realistic prefill batch.

    Naming matches lightllm's infer_state:
      - ``q_seq_len`` = number of new Q tokens in this prefill call
      - ``prompt_cache_len`` = length of the already-cached prefix for this req
      - ``seq_len`` = prompt_cache_len + q_seq_len (total KV length)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    q_seq_lens = torch.randint(low=1, high=max_q_seq_len + 1, size=(batch,), generator=g)
    prompt_cache_lens = torch.randint(low=0, high=max_prompt_cache_len + 1, size=(batch,), generator=g)
    seq_lens = q_seq_lens + prompt_cache_lens

    sum_q = int(q_seq_lens.sum().item())
    sum_total = int(seq_lens.sum().item())
    max_seq_len_in_batch = int(seq_lens.max().item())
    max_q_seq_len_in_batch = int(q_seq_lens.max().item())

    b_q_start_loc = torch.zeros(batch, dtype=torch.int32)
    cur = 0
    for i in range(batch):
        b_q_start_loc[i] = cur
        cur += int(q_seq_lens[i].item())

    # Permute so batch idx != request idx: exercises the page_table indexing.
    b_req_idx = torch.randperm(batch, generator=g).to(torch.int32)

    # Global KV pool with scattered, non-contiguous slot assignment per request.
    base = 1024
    kv_pool_size = base + sum_total + kv_pool_slack
    pool = torch.randperm(kv_pool_size - base, generator=g)[:sum_total] + base

    req_to_token_indexs = torch.zeros((batch, max_seq_len_in_batch), dtype=torch.int32)
    p = 0
    for r_logical, req_id in enumerate(b_req_idx.tolist()):
        L = int(seq_lens[r_logical].item())
        req_to_token_indexs[req_id, :L] = pool[p : p + L].to(torch.int32)
        p += L

    # Randomly place contiguous image-token spans inside each batch's new-Q region.
    b_image_token_tag = torch.zeros(sum_q, dtype=torch.bool)
    for i in range(batch):
        M = int(q_seq_lens[i].item())
        if M < 2:
            continue
        if torch.rand((), generator=g).item() > image_prob:
            continue
        n_spans = int(torch.randint(1, num_image_spans_max + 1, (1,), generator=g).item())
        start_pack = int(b_q_start_loc[i].item())
        for _ in range(n_spans):
            span_len = int(
                torch.randint(1, max(2, image_span_len_max) + 1, (1,), generator=g).item()
            )
            span_len = min(span_len, M)
            s_rel = int(torch.randint(0, M - span_len + 1, (1,), generator=g).item())
            b_image_token_tag[start_pack + s_rel : start_pack + s_rel + span_len] = True

    b_seq_len = seq_lens.to(torch.int32)
    b_prompt_cache_len = prompt_cache_lens.to(torch.int32)

    # Per-batch last image-token index in *batch-local* packed-q coordinates.
    # Matches NeoChatInferStateInfo._compute_b_max_image_q_idx semantics:
    #   shape int32[batch]; value == -1 means that batch has no image tokens.
    # Computed on CPU (b_image_token_tag is still on CPU here) so no D2H sync
    # — keeps the eventual flash_attn_with_kvcache call CUDA-graph-safe.
    b_max_image_q_idx_cpu = torch.full((batch,), -1, dtype=torch.int32)
    for b in range(batch):
        start = int(b_q_start_loc[b].item())
        length = int(q_seq_lens[b].item())
        seg = b_image_token_tag[start : start + length]
        idx = torch.nonzero(seg, as_tuple=False)
        if idx.numel() > 0:
            b_max_image_q_idx_cpu[b] = int(idx[-1, 0].item())

    q = torch.randn((sum_q, Hq, D), dtype=dtype, device=device)
    k = torch.randn((kv_pool_size, Hk, D), dtype=dtype, device=device)
    v = torch.randn((kv_pool_size, Hk, D), dtype=dtype, device=device)

    return dict(
        q=q,
        k=k,
        v=v,
        b_req_idx=b_req_idx.to(device),
        b_q_start_loc=b_q_start_loc.to(device),
        b_seq_len=b_seq_len.to(device),
        b_prompt_cache_len=b_prompt_cache_len.to(device),
        max_seq_len_in_batch=max_seq_len_in_batch,
        max_q_seq_len_in_batch=max_q_seq_len_in_batch,
        req_to_token_indexs=req_to_token_indexs.to(device),
        b_image_token_tag=b_image_token_tag.to(device),
        b_max_image_q_idx=b_max_image_q_idx_cpu.to(device),
        q_seq_lens=q_seq_lens,
        prompt_cache_lens=prompt_cache_lens,
    )


def _fa3_prefill_with_image_tag(inputs: dict) -> torch.Tensor:
    """Drive ``flash_attn_with_kvcache`` with the same prefill semantics as
    ``Fa3PrefillAttState._nomarl_prefill_att`` plus an optional
    ``image_token_tag`` kwarg for image-token bidirectional attention.
    """
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    device = q.device

    # Build page_table[b, p] = req_to_token_indexs[b_req_idx[b], p].
    page_table = inputs["req_to_token_indexs"][
        inputs["b_req_idx"].long(), : inputs["max_seq_len_in_batch"]
    ].to(torch.int32)

    q_seq_lens_t = inputs["b_seq_len"].to(torch.int32) - inputs["b_prompt_cache_len"].to(torch.int32)

    cu_seqlens_q = torch.zeros(q_seq_lens_t.shape[0] + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = q_seq_lens_t.cumsum(0).to(torch.int32)

    cu_seqlens_k = torch.zeros(inputs["b_seq_len"].shape[0] + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = inputs["b_seq_len"].cumsum(0).to(torch.int32)

    sm_scale = 1.0 / math.sqrt(q.shape[-1])

    # page_size = 1 paged KV cache view.
    k_cache = k.view(k.shape[0], 1, k.shape[1], k.shape[2])
    v_cache = v.view(v.shape[0], 1, v.shape[1], v.shape[2])

    o = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=inputs["b_seq_len"],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=inputs["max_q_seq_len_in_batch"],
        softmax_scale=sm_scale,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        k_descale=None,
        v_descale=None,
        return_softmax_lse=False,
        # image-token bidirectional attention. Packed like q (shape [sum_q],
        # bool). Rows where the tag is True are allowed to attend to every
        # real key in the request (not just the causal prefix).
        #
        # b_max_image_q_idx is int32[batch]: per-batch last image-token index
        # in batch-local coordinates, -1 if no image in that batch. Lets the
        # fa3 kernel skip n_block_max extension for text-only requests, which
        # is critical for mixed-modality batches. Pre-computed in
        # _build_inputs (host-side) so this call is CUDA-graph safe.
        image_token_tag=inputs["b_image_token_tag"],
        max_image_q_idx=inputs["b_max_image_q_idx"],
    )
    return o


def _report_per_batch_error(out_fa3, out_ref, q_seq_lens, b_q_start_loc, image_tag, tag=""):
    print(f"\n[{tag}] per-batch error breakdown (abs / rel / cos):")
    for i in range(q_seq_lens.shape[0]):
        s = int(b_q_start_loc[i].item())
        m = int(q_seq_lens[i].item())
        if m == 0:
            continue
        a = out_fa3[s : s + m].float()
        b = out_ref[s : s + m].float()
        abs_err = (a - b).abs().max().item()
        denom = b.abs().max().item() + 1e-6
        rel_err = abs_err / denom
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        n_img = int(image_tag[s : s + m].sum().item())
        print(
            f"  batch {i:02d} | M={m:4d} | image_tokens={n_img:4d} | "
            f"max_abs={abs_err:.4e} | max_rel={rel_err:.4e} | cos={cos:.6f}"
        )


def _run_case(
    batch: int,
    Hq: int,
    Hk: int,
    D: int,
    dtype: torch.dtype,
    seed: int,
    max_q_seq_len: int,
    max_prompt_cache_len: int,
    atol: float = 5e-2,
    rtol: float = 5e-2,
    cos_threshold: float = 0.99,
    verbose: bool = True,
):
    assert Hq % Hk == 0
    device = "cuda"

    inputs = _build_inputs(
        batch=batch,
        Hq=Hq,
        Hk=Hk,
        D=D,
        dtype=dtype,
        device=device,
        max_q_seq_len=max_q_seq_len,
        max_prompt_cache_len=max_prompt_cache_len,
        seed=seed,
    )

    out_fa3 = _fa3_prefill_with_image_tag(inputs)

    out_ref = torch_reference_context_attention_neo(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["b_req_idx"],
        inputs["b_q_start_loc"],
        inputs["b_seq_len"],
        inputs["b_prompt_cache_len"],
        inputs["req_to_token_indexs"],
        inputs["b_image_token_tag"],
    )

    a = out_fa3.float().reshape_as(out_ref.float())
    b = out_ref.float()
    abs_err = (a - b).abs().max().item()
    denom = b.abs().max().item() + 1e-6
    rel_err = abs_err / denom
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    n_image = int(inputs["b_image_token_tag"].sum().item())
    n_tokens = int(inputs["b_image_token_tag"].numel())
    if verbose:
        print(
            f"\ncase: batch={batch} Hq={Hq} Hk={Hk} D={D} dtype={dtype} "
            f"seed={seed} image_tokens={n_image}/{n_tokens}"
        )
        print(
            f"  global: max_abs={abs_err:.4e} max_rel={rel_err:.4e} cos={cos:.6f} "
            f"(allclose atol={atol}, rtol={rtol}? "
            f"{torch.allclose(a, b, atol=atol, rtol=rtol)})"
        )
        _report_per_batch_error(
            out_fa3,
            out_ref,
            inputs["q_seq_lens"],
            inputs["b_q_start_loc"],
            inputs["b_image_token_tag"],
            tag=f"seed={seed}",
        )

    return abs_err, rel_err, cos


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.skipif(flash_attn_with_kvcache is None, reason="fa3 not available")
@pytest.mark.parametrize(
    "batch,Hq,Hk,D,dtype,seed,max_q_seq_len,max_prompt_cache_len",
    [
        (4, 8, 2, 128, torch.bfloat16, 0, 128, 256),
        (4, 8, 2, 128, torch.bfloat16, 1, 256, 512),
        (8, 16, 4, 128, torch.bfloat16, 2, 256, 512),
        (16, 28, 4, 128, torch.bfloat16, 3, 128, 256),
        (4, 8, 2, 128, torch.float16, 4, 256, 512),
        (4, 8, 8, 64, torch.bfloat16, 5, 128, 256),
        (3, 8, 2, 128, torch.bfloat16, 6, 8, 1024),
    ],
)
def test_fa3_neo_prefill_with_image_tag(
    batch, Hq, Hk, D, dtype, seed, max_q_seq_len, max_prompt_cache_len
):
    abs_err, rel_err, cos = _run_case(
        batch=batch,
        Hq=Hq,
        Hk=Hk,
        D=D,
        dtype=dtype,
        seed=seed,
        max_q_seq_len=max_q_seq_len,
        max_prompt_cache_len=max_prompt_cache_len,
        verbose=True,
    )
    assert cos > 0.99, f"cosine similarity too low: {cos}"
    assert rel_err < 5e-2, f"max relative error too large: {rel_err}"


def _bench_case(
    batch: int,
    Hq: int,
    Hk: int,
    D: int,
    dtype: torch.dtype,
    seed: int,
    max_q_seq_len: int,
    max_prompt_cache_len: int,
    rep_ms: int = 100,
    warmup_iters: int = 3,
):
    """Compare FA3 (with image_token_tag) vs the original Triton
    ``context_attention_fwd_neo`` using ``triton.testing.do_bench_cudagraph``.

    Both kernels are captured into a CUDA graph so scheduling/launch overhead
    is minimized and the measurement reflects the kernel cost.
    """
    assert Hq % Hk == 0
    device = "cuda"

    inputs = _build_inputs(
        batch=batch,
        Hq=Hq,
        Hk=Hk,
        D=D,
        dtype=dtype,
        device=device,
        max_q_seq_len=max_q_seq_len,
        max_prompt_cache_len=max_prompt_cache_len,
        seed=seed,
    )

    # --- fa3 runner: output tensor is allocated inside flash_attn_with_kvcache.
    def fa3_run():
        return _fa3_prefill_with_image_tag(inputs)

    # --- triton runner: pre-allocate o & position_ids so the graph captures
    # only the kernel launch.
    o_triton = torch.empty_like(inputs["q"])
    # Kernel signature requires position_ids but the current masking path does
    # not read it; zeros are fine for perf measurement.
    position_ids_0 = torch.zeros(
        inputs["q"].shape[0], dtype=torch.int32, device=inputs["q"].device
    )

    def triton_run():
        context_attention_fwd_neo(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            o_triton,
            position_ids_0,
            inputs["b_req_idx"],
            inputs["b_q_start_loc"],
            inputs["b_seq_len"],
            inputs["b_prompt_cache_len"],
            inputs["max_q_seq_len_in_batch"],
            inputs["req_to_token_indexs"],
            inputs["b_image_token_tag"],
        )

    # Warm up outside the graph capture so lazy allocations / autotune happen.
    for _ in range(warmup_iters):
        fa3_run()
        triton_run()
    torch.cuda.synchronize()

    fa3_ms = triton_testing.do_bench_cudagraph(fa3_run, rep=rep_ms)
    triton_ms = triton_testing.do_bench_cudagraph(triton_run, rep=rep_ms)

    n_image = int(inputs["b_image_token_tag"].sum().item())
    n_tokens = int(inputs["b_image_token_tag"].numel())
    sum_kv = int(inputs["b_seq_len"].sum().item())
    speedup = triton_ms / fa3_ms if fa3_ms > 0 else float("inf")

    print(
        f"bench: batch={batch} Hq={Hq} Hk={Hk} D={D} dtype={str(dtype).split('.')[-1]:<8s} "
        f"max_q_seq_len={max_q_seq_len:4d} max_prompt_cache_len={max_prompt_cache_len:4d} "
        f"image_tokens={n_image:4d}/{n_tokens:5d} sum_kv={sum_kv:6d} | "
        f"fa3 {fa3_ms*1000:8.1f} us | triton {triton_ms*1000:8.1f} us | "
        f"speedup {speedup:5.2f}x"
    )

    return fa3_ms, triton_ms


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA available.")
        raise SystemExit(0)
    if flash_attn_with_kvcache is None:
        print("fa3 flash_attn_with_kvcache not available (sgl_kernel missing?).")
        raise SystemExit(0)

    torch.manual_seed(0)

    cases = [
        dict(batch=4, Hq=8, Hk=2, D=128, dtype=torch.bfloat16, seed=0, max_q_seq_len=128, max_prompt_cache_len=256),
        dict(batch=8, Hq=16, Hk=4, D=128, dtype=torch.bfloat16, seed=1, max_q_seq_len=256, max_prompt_cache_len=512),
        dict(batch=16, Hq=28, Hk=4, D=128, dtype=torch.bfloat16, seed=2, max_q_seq_len=128, max_prompt_cache_len=256),
        dict(batch=4, Hq=8, Hk=2, D=128, dtype=torch.float16, seed=3, max_q_seq_len=256, max_prompt_cache_len=512),
    ]

    print("=" * 100)
    print("Correctness")
    print("=" * 100)
    for cfg in cases:
        _run_case(**cfg, verbose=True)

    if triton_testing is None or context_attention_fwd_neo is None:
        print("\nSkipping benchmark: triton or context_attention_fwd_neo not available.")
        raise SystemExit(0)

    print("\n" + "=" * 100)
    print("Benchmark (triton.testing.do_bench_cudagraph)")
    print("=" * 100)

    # Cold-prefill sweep: max_prompt_cache_len=0 so seq_len == q_seq_len.
    # Head shape matches neo_chat_moe / Qwen3 llm_config:
    #   num_attention_heads = 32, num_key_value_heads = 8, head_dim = 128
    # (GQA ratio 4:1)
    bench_batches = [8, 16, 32, 64, 128]
    bench_q_seq_lens = [1024, 4096, 8192]

    bench_cases = [
        dict(
            batch=b,
            Hq=32,
            Hk=8,
            D=128,
            dtype=torch.bfloat16,
            seed=0,
            max_q_seq_len=s,
            max_prompt_cache_len=0,
        )
        for b in bench_batches
        for s in bench_q_seq_lens
    ]

    for cfg in bench_cases:
        _bench_case(**cfg)
