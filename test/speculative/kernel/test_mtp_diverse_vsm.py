"""
Test MTP Diverse Attention VSM Version

Compares VSM version (token_decode_attention_mtp_diverse_vsm_single_token)
against original version (token_decode_attention_mtp_diverse_single_token).

VSM version optimizes computation by:
1. Only computing for batches where b_mark_shared_group > 0 (last in group)
2. Dynamic SM parallelism based on workload
3. Skipping unnecessary computation in stage2 for non-last requests
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse.mtp_diverse_attn import (
    token_decode_attention_mtp_diverse_single_token,
)
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse.mtp_diverse_attn_vsm import (
    token_decode_attention_mtp_diverse_vsm_single_token,
)


def test_config(name, b_seq_len, b_mark_shared_group, q, k, v, Req_to_tokens, B_req_idx, block_seq=256):
    """Test a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  b_seq_len: {b_seq_len.tolist()}")
    print(f"  b_mark_shared_group: {b_mark_shared_group.tolist()}")
    print(f"{'='*60}")

    # Run original version
    out_original = token_decode_attention_mtp_diverse_single_token(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        Req_to_tokens=Req_to_tokens,
        B_req_idx=B_req_idx,
        b_seq_len=b_seq_len.clone(),
        b_mark_shared_group=b_mark_shared_group.clone(),
        block_seq=block_seq,
    )

    # Run VSM version
    out_vsm = token_decode_attention_mtp_diverse_vsm_single_token(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        Req_to_tokens=Req_to_tokens,
        B_req_idx=B_req_idx,
        b_seq_len=b_seq_len.clone(),
        b_mark_shared_group=b_mark_shared_group.clone(),
    )

    # Compare - only check batches where b_mark_shared_group > 0
    mask = b_mark_shared_group > 0
    out_original_masked = out_original[mask]
    out_vsm_masked = out_vsm[mask]

    diff = (out_original_masked - out_vsm_masked).abs()
    max_diff = diff.max().item()
    ref_norm = out_original_masked.abs().max().item()
    rel_error = max_diff / (ref_norm + 1e-8)

    print(f"  Max abs diff: {max_diff:.6e}")
    print(f"  Relative error: {rel_error:.6e}")

    atol = 5e-3
    rtol = 1e-2
    passed = max_diff < atol or rel_error < rtol

    if passed:
        print(f"  Result: PASSED")
    else:
        print(f"  Result: FAILED")
        print(f"  Original output sample: {out_original_masked[0, 0, :5]}")
        print(f"  VSM output sample: {out_vsm_masked[0, 0, :5]}")

    return passed, rel_error


def run_all_tests():
    """Run all test configurations"""
    q_head_num = 32
    kv_head_num = 8
    head_dim = 128
    max_kv_len = 1024
    dtype = torch.bfloat16

    all_passed = True

    # Category 1: Independent Requests
    print("\n" + "="*70)
    print("Category 1: Independent Requests (all b_mark_shared_group=1)")
    print("="*70)

    batch_size = 8
    q = torch.randn(batch_size, q_head_num, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    Req_to_tokens = torch.arange(0, max_kv_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, max_kv_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    b_seq_len = torch.randint(100, max_kv_len, (batch_size,), dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    passed, _ = test_config("All independent (batch=8)", b_seq_len, b_mark_shared_group, q, k, v, Req_to_tokens, B_req_idx)
    all_passed = all_passed and passed

    # Category 2: Single Large Group
    print("\n" + "="*70)
    print("Category 2: Single Large Group")
    print("="*70)

    batch_size = 8
    q = torch.randn(batch_size, q_head_num, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    Req_to_tokens = torch.arange(0, max_kv_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, max_kv_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    b_seq_len = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.tensor([0, 0, 0, 0, 0, 0, 0, 8], dtype=torch.int32, device="cuda")

    passed, _ = test_config("Single group of 8", b_seq_len, b_mark_shared_group, q, k, v, Req_to_tokens, B_req_idx)
    all_passed = all_passed and passed

    # Category 3: Multiple Fixed-Size Groups
    print("\n" + "="*70)
    print("Category 3: Multiple Fixed-Size Groups")
    print("="*70)

    batch_size = 12
    q = torch.randn(batch_size, q_head_num, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    Req_to_tokens = torch.arange(0, max_kv_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, max_kv_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    b_seq_len = torch.tensor([100, 200, 300, 150, 250, 350, 120, 220, 320, 180, 280, 380], dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.tensor([0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3], dtype=torch.int32, device="cuda")

    passed, _ = test_config("4 groups of 3", b_seq_len, b_mark_shared_group, q, k, v, Req_to_tokens, B_req_idx)
    all_passed = all_passed and passed

    # Category 4: Mixed Group Sizes
    print("\n" + "="*70)
    print("Category 4: Mixed Group Sizes")
    print("="*70)

    batch_size = 10
    q = torch.randn(batch_size, q_head_num, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    Req_to_tokens = torch.arange(0, max_kv_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, max_kv_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    b_seq_len = torch.tensor([100, 200, 300, 150, 250, 350, 180, 280, 380, 480], dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.tensor([1, 0, 2, 0, 0, 3, 0, 0, 0, 4], dtype=torch.int32, device="cuda")

    passed, _ = test_config("Mixed: 1,2,3,4", b_seq_len, b_mark_shared_group, q, k, v, Req_to_tokens, B_req_idx)
    all_passed = all_passed and passed

    # Category 5: Pairs (group_size=2)
    print("\n" + "="*70)
    print("Category 5: Pairs (group_size=2)")
    print("="*70)

    batch_size = 8
    q = torch.randn(batch_size, q_head_num, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    Req_to_tokens = torch.arange(0, max_kv_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, max_kv_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    b_seq_len = torch.tensor([100, 200, 150, 250, 120, 220, 180, 280], dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.tensor([0, 2, 0, 2, 0, 2, 0, 2], dtype=torch.int32, device="cuda")

    passed, _ = test_config("4 pairs", b_seq_len, b_mark_shared_group, q, k, v, Req_to_tokens, B_req_idx)
    all_passed = all_passed and passed

    # Category 6: Edge Cases
    print("\n" + "="*70)
    print("Category 6: Edge Cases")
    print("="*70)

    # Single request
    batch_size = 1
    q = torch.randn(batch_size, q_head_num, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    Req_to_tokens = torch.arange(0, max_kv_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, max_kv_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    b_seq_len = torch.tensor([512], dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.tensor([1], dtype=torch.int32, device="cuda")

    passed, _ = test_config("Single request", b_seq_len, b_mark_shared_group, q, k, v, Req_to_tokens, B_req_idx)
    all_passed = all_passed and passed

    # Short seq_len
    batch_size = 8
    q = torch.randn(batch_size, q_head_num, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    Req_to_tokens = torch.arange(0, max_kv_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, max_kv_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    b_seq_len_short = torch.randint(10, 50, (batch_size,), dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    passed, _ = test_config("Short seq_len (10-50)", b_seq_len_short, b_mark_shared_group, q, k, v, Req_to_tokens, B_req_idx)
    all_passed = all_passed and passed

    # Long seq_len
    batch_size = 8
    q = torch.randn(batch_size, q_head_num, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(max_kv_len * batch_size, kv_head_num, head_dim, dtype=dtype, device="cuda")
    Req_to_tokens = torch.arange(0, max_kv_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, max_kv_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    b_seq_len_long = torch.randint(800, 1000, (batch_size,), dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    passed, _ = test_config("Long seq_len (800-1000)", b_seq_len_long, b_mark_shared_group, q, k, v, Req_to_tokens, B_req_idx)
    all_passed = all_passed and passed

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    torch.cuda.set_device(0)
    run_all_tests()
