"""Unit test for ``context_attention_fwd_neo``.

Torch reference expresses the *semantics* of the attention, not the kernel's
internal block structure — it has no notion of BLOCK_N / BLOCK_M. For each
batch element we gather K/V for the whole request (prompt + new tokens) via
``req_to_token_indexs`` and apply::

    allow[m, k] = (k <= q_pos[m]) OR image_tag[m]          for k in [0, total)

i.e. normal queries are causal, image-token queries can see every real key in
the request. If the Triton kernel disagrees with this reference, the kernel is
wrong.

Run directly for quick debugging:

    python unit_tests/common/basemodel/triton_kernel/att/prefill_att/\
        test_context_attention_fwd_neo.py

or via pytest:

    pytest unit_tests/common/basemodel/triton_kernel/att/prefill_att/\
        test_context_attention_fwd_neo.py -x -s
"""

import math
import pytest
import torch

from lightllm.models.neo_chat_moe.triton_kernel.context_attention_fwd_neo import (
    context_attention_fwd_neo,
)


def torch_reference_context_attention_neo(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
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
        total = int(b_seq_len[b].item())
        prompt = int(b_prompt_cache_len[b].item())
        new = total - prompt
        if new <= 0:
            continue

        q_start = int(b_start_loc[b].item())
        q_blk = q[q_start : q_start + new]  # [M, Hq, D]
        image_tag = b_image_token_tag[q_start : q_start + new].to(torch.bool)

        token_locs = req_to_token_indexs[req_idx, :total].to(torch.int64)
        k_blk = k[token_locs]  # [total, Hk, D]
        v_blk = v[token_locs]

        q_pos = torch.arange(prompt, total, device=device, dtype=torch.int64)  # [M]
        k_pos = torch.arange(0, total, device=device, dtype=torch.int64)  # [total]
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

        out[q_start : q_start + new] = out_blk

    return out


def _build_inputs(
    batch: int,
    Hq: int,
    Hk: int,
    D: int,
    dtype: torch.dtype,
    device: str,
    max_new: int = 256,
    max_prompt: int = 512,
    image_prob: float = 0.7,
    num_image_spans_max: int = 3,
    image_span_len_max: int = 24,
    kv_pool_slack: int = 4096,
    seed: int = 0,
):
    g = torch.Generator(device="cpu").manual_seed(seed)

    new_lens = torch.randint(low=1, high=max_new + 1, size=(batch,), generator=g)
    prompt_lens = torch.randint(low=0, high=max_prompt + 1, size=(batch,), generator=g)
    total_lens = new_lens + prompt_lens

    sum_new = int(new_lens.sum().item())
    sum_total = int(total_lens.sum().item())
    max_total_len = int(total_lens.max().item())
    max_new_len = int(new_lens.max().item())

    b_start_loc = torch.zeros(batch, dtype=torch.int32)
    cur = 0
    for i in range(batch):
        b_start_loc[i] = cur
        cur += int(new_lens[i].item())

    # Permute so batch idx != request idx: exercises the Req_to_tokens indexing.
    b_req_idx = torch.randperm(batch, generator=g).to(torch.int32)

    # Global KV pool with scattered, non-contiguous slot assignment per request.
    base = 1024
    kv_pool_size = base + sum_total + kv_pool_slack
    pool = torch.randperm(kv_pool_size - base, generator=g)[:sum_total] + base

    req_to_token_indexs = torch.zeros((batch, max_total_len), dtype=torch.int32)
    p = 0
    for r_logical, req_id in enumerate(b_req_idx.tolist()):
        L = int(total_lens[r_logical].item())
        req_to_token_indexs[req_id, :L] = pool[p : p + L].to(torch.int32)
        p += L

    # Randomly place contiguous image-token spans inside each batch's NEW region.
    b_image_token_tag = torch.zeros(sum_new, dtype=torch.bool)
    for i in range(batch):
        M = int(new_lens[i].item())
        if M < 2:
            continue
        if torch.rand((), generator=g).item() > image_prob:
            continue
        n_spans = int(torch.randint(1, num_image_spans_max + 1, (1,), generator=g).item())
        start_pack = int(b_start_loc[i].item())
        for _ in range(n_spans):
            span_len = int(torch.randint(1, max(2, image_span_len_max) + 1, (1,), generator=g).item())
            span_len = min(span_len, M)
            s_rel = int(torch.randint(0, M - span_len + 1, (1,), generator=g).item())
            b_image_token_tag[start_pack + s_rel : start_pack + s_rel + span_len] = True

    b_seq_len = total_lens.to(torch.int32)
    b_prompt_cache_len = prompt_lens.to(torch.int32)

    # position_ids[0]: kernel API still requires it even though its current
    # mask logic only reads b_image_token_tag.
    position_ids_0 = torch.empty(sum_new, dtype=torch.int32)
    for i in range(batch):
        M = int(new_lens[i].item())
        P = int(prompt_lens[i].item())
        s = int(b_start_loc[i].item())
        position_ids_0[s : s + M] = torch.arange(P, P + M, dtype=torch.int32)

    q = torch.randn((sum_new, Hq, D), dtype=dtype, device=device)
    k = torch.randn((kv_pool_size, Hk, D), dtype=dtype, device=device)
    v = torch.randn((kv_pool_size, Hk, D), dtype=dtype, device=device)
    o = torch.empty_like(q)

    return dict(
        q=q,
        k=k,
        v=v,
        o=o,
        position_ids_0=position_ids_0.to(device),
        b_req_idx=b_req_idx.to(device),
        b_start_loc=b_start_loc.to(device),
        b_seq_len=b_seq_len.to(device),
        b_prompt_cache_len=b_prompt_cache_len.to(device),
        max_new_len=max_new_len,
        req_to_token_indexs=req_to_token_indexs.to(device),
        b_image_token_tag=b_image_token_tag.to(device),
        new_lens=new_lens,
        prompt_lens=prompt_lens,
    )


def _report_per_batch_error(out_triton, out_ref, new_lens, b_start_loc, image_tag, tag=""):
    print(f"\n[{tag}] per-batch error breakdown (abs / rel / cos):")
    for i in range(new_lens.shape[0]):
        s = int(b_start_loc[i].item())
        m = int(new_lens[i].item())
        if m == 0:
            continue
        a = out_triton[s : s + m].float()
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
    max_new: int,
    max_prompt: int,
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
        max_new=max_new,
        max_prompt=max_prompt,
        seed=seed,
    )

    context_attention_fwd_neo(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["o"],
        inputs["position_ids_0"],
        inputs["b_req_idx"],
        inputs["b_start_loc"],
        inputs["b_seq_len"],
        inputs["b_prompt_cache_len"],
        inputs["max_new_len"],
        inputs["req_to_token_indexs"],
        inputs["b_image_token_tag"],
    )
    out_triton = inputs["o"]

    out_ref = torch_reference_context_attention_neo(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["b_req_idx"],
        inputs["b_start_loc"],
        inputs["b_seq_len"],
        inputs["b_prompt_cache_len"],
        inputs["req_to_token_indexs"],
        inputs["b_image_token_tag"],
    )

    a = out_triton.float()
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
            out_triton,
            out_ref,
            inputs["new_lens"],
            inputs["b_start_loc"],
            inputs["b_image_token_tag"],
            tag=f"seed={seed}",
        )

    return abs_err, rel_err, cos


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "batch,Hq,Hk,D,dtype,seed,max_new,max_prompt",
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
def test_context_attention_fwd_neo(batch, Hq, Hk, D, dtype, seed, max_new, max_prompt):
    abs_err, rel_err, cos = _run_case(
        batch=batch,
        Hq=Hq,
        Hk=Hk,
        D=D,
        dtype=dtype,
        seed=seed,
        max_new=max_new,
        max_prompt=max_prompt,
        verbose=True,
    )
    assert cos > 0.99, f"cosine similarity too low: {cos}"
    assert rel_err < 5e-2, f"max relative error too large: {rel_err}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA available.")
        raise SystemExit(0)

    torch.manual_seed(0)

    cases = [
        dict(batch=4, Hq=8, Hk=2, D=128, dtype=torch.bfloat16, seed=0, max_new=128, max_prompt=256),
        dict(batch=8, Hq=16, Hk=4, D=128, dtype=torch.bfloat16, seed=1, max_new=256, max_prompt=512),
        dict(batch=16, Hq=28, Hk=4, D=128, dtype=torch.bfloat16, seed=2, max_new=128, max_prompt=256),
        dict(batch=4, Hq=8, Hk=2, D=128, dtype=torch.float16, seed=3, max_new=256, max_prompt=512),
    ]

    for cfg in cases:
        _run_case(**cfg, verbose=True)
