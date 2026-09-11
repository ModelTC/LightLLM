"""Correctness test for Neo's causal-plus-image mixed attention mask.

The independent Torch reference states the contract directly::

    allow[query, key] = (key <= query) or (key < image_end[query])

It also gathers K/V through the request page table, so the test covers prompt
cache, GQA, permuted request ids, and non-contiguous KV slots.
"""

import math

import pytest
import torch

from lightllm.models.neo_chat.triton_kernel.context_attention_fwd_neo import (
    context_attention_fwd_neo,
)


def _torch_reference(
    q,
    k,
    v,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_prompt_cache_len,
    req_to_token_indexs,
    b_image_token_end,
):
    output = torch.empty_like(q)
    q_head_count = q.shape[1]
    kv_head_count = k.shape[1]
    kv_group = q_head_count // kv_head_count
    scale = 1.0 / math.sqrt(q.shape[-1])

    for batch_index in range(b_seq_len.numel()):
        req_index = int(b_req_idx[batch_index])
        total_len = int(b_seq_len[batch_index])
        prompt_len = int(b_prompt_cache_len[batch_index])
        new_len = total_len - prompt_len
        q_start = int(b_start_loc[batch_index])

        q_block = q[q_start : q_start + new_len]
        kv_locations = req_to_token_indexs[req_index, :total_len].long()
        k_block = k[kv_locations]
        v_block = v[kv_locations]
        q_positions = torch.arange(
            prompt_len,
            total_len,
            dtype=torch.int64,
            device=q.device,
        )
        k_positions = torch.arange(total_len, dtype=torch.int64, device=q.device)
        image_ends = b_image_token_end[q_start : q_start + new_len].long()
        allow = (k_positions[None, :] <= q_positions[:, None]) | (
            k_positions[None, :] < image_ends[:, None]
        )

        for q_head in range(q_head_count):
            kv_head = q_head // kv_group
            scores = (
                q_block[:, q_head].float() @ k_block[:, kv_head].float().transpose(0, 1)
            ) * scale
            scores.masked_fill_(~allow, -1.0e8)
            output[q_start : q_start + new_len, q_head] = (
                torch.softmax(scores, dim=-1) @ v_block[:, kv_head].float()
            ).to(q.dtype)

    return output


def _build_inputs(*, new_lens, prompt_lens, q_heads, kv_heads, head_dim, dtype, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    new_lens = torch.tensor(new_lens, dtype=torch.int32)
    prompt_lens = torch.tensor(prompt_lens, dtype=torch.int32)
    total_lens = new_lens + prompt_lens
    batch_size = new_lens.numel()

    b_start_loc = torch.cat(
        (torch.zeros(1, dtype=torch.int32), new_lens.cumsum(0)[:-1])
    )
    # The logical request id intentionally differs from packed batch order.
    b_req_idx = torch.arange(batch_size - 1, -1, -1, dtype=torch.int32)

    sum_total = int(total_lens.sum())
    max_total = int(total_lens.max())
    base_slot = 37
    kv_pool_size = base_slot + sum_total + 257
    scattered_slots = (
        torch.randperm(kv_pool_size - base_slot, generator=generator)[:sum_total]
        + base_slot
    )
    req_to_token_indexs = torch.zeros(
        (batch_size, max_total),
        dtype=torch.int32,
    )
    slot_offset = 0
    for batch_index, req_index in enumerate(b_req_idx.tolist()):
        total_len = int(total_lens[batch_index])
        req_to_token_indexs[req_index, :total_len] = scattered_slots[
            slot_offset : slot_offset + total_len
        ].to(torch.int32)
        slot_offset += total_len

    # Two image spans per sufficiently long sequence, separated by causal text.
    # The long case crosses a Triton query block and exercises block_image_end.
    b_image_token_end = torch.zeros(int(new_lens.sum()), dtype=torch.int32)
    for batch_index in range(batch_size):
        new_len = int(new_lens[batch_index])
        prompt_len = int(prompt_lens[batch_index])
        packed_start = int(b_start_loc[batch_index])
        if new_len < 3:
            continue
        middle = max(2, new_len // 2)
        b_image_token_end[packed_start + 1 : packed_start + middle] = (
            prompt_len + middle
        )
        if middle + 1 < new_len:
            b_image_token_end[packed_start + middle + 1 : packed_start + new_len] = (
                prompt_len + new_len
            )

    device = "cuda"
    torch.manual_seed(seed)
    q = torch.randn(
        (int(new_lens.sum()), q_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        (kv_pool_size, kv_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    v = torch.randn_like(k)
    return {
        "q": q,
        "k": k,
        "v": v,
        "o": torch.empty_like(q),
        "b_req_idx": b_req_idx.to(device),
        "b_start_loc": b_start_loc.to(device),
        "b_seq_len": total_lens.to(device),
        "b_prompt_cache_len": prompt_lens.to(device),
        "max_new_len": int(new_lens.max()),
        "req_to_token_indexs": req_to_token_indexs.to(device),
        "b_image_token_end": b_image_token_end.to(device),
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Neo Triton attention requires CUDA"
)
@pytest.mark.parametrize(
    "new_lens,prompt_lens,q_heads,kv_heads,head_dim,dtype,seed",
    [
        ([3, 5, 9], [0, 4, 7], 8, 2, 128, torch.bfloat16, 0),
        ([141, 33], [17, 129], 8, 8, 64, torch.float16, 1),
    ],
)
def test_context_attention_fwd_neo_matches_torch_reference(
    new_lens,
    prompt_lens,
    q_heads,
    kv_heads,
    head_dim,
    dtype,
    seed,
):
    inputs = _build_inputs(
        new_lens=new_lens,
        prompt_lens=prompt_lens,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        seed=seed,
    )
    context_attention_fwd_neo(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["o"],
        inputs["b_req_idx"],
        inputs["b_start_loc"],
        inputs["b_seq_len"],
        inputs["b_prompt_cache_len"],
        inputs["max_new_len"],
        inputs["req_to_token_indexs"],
        inputs["b_image_token_end"],
    )
    expected = _torch_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["b_req_idx"],
        inputs["b_start_loc"],
        inputs["b_seq_len"],
        inputs["b_prompt_cache_len"],
        inputs["req_to_token_indexs"],
        inputs["b_image_token_end"],
    )

    torch.testing.assert_close(inputs["o"], expected, atol=5e-2, rtol=5e-2)
    cosine = torch.nn.functional.cosine_similarity(
        inputs["o"].float().flatten(),
        expected.float().flatten(),
        dim=0,
    )
    assert float(cosine) > 0.99
