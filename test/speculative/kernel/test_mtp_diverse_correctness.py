"""
Test MTP Diverse Attention Correctness

验证 MTP diverse attention 的输出是否正确：
- 组内第 i 个请求的输出应该等于逐请求计算该请求对前 i+1 个 KV 的 attention
"""
import torch
import os
import sys

os.environ['PYTHONPATH'] = '/data/nvme0/chenjunyi/project/lightllm'
sys.path.insert(0, '/data/nvme0/chenjunyi/project/lightllm')

import json
os.environ['LIGHTLLM_START_ARGS'] = json.dumps({
    'mtp_step': 1,
    'model_dir': '/tmp/test_model',
    'max_total_token_num': 10000,
})

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse import (
    mtp_diverse_stage1_single_token,
    mtp_diverse_stage2_single_token,
)


def setup_mtp_test_data(kv_len, group_size, batch_groups, test_dtype=torch.bfloat16, device="cuda", seed=42):
    """
    Setup MTP test data for single token mode.

    在静态 MTP 模式下：
    - 每个请求有 1 个 Q token
    - 组内第 i 个请求只能看到前 i+1 个 KV
    - 组内请求共享相同的 KV 前缀
    """
    torch.manual_seed(seed)

    num_heads = 32
    kv_head_num = 4  # gqa_group_size = 8
    head_dim = 128

    batch_size = batch_groups * group_size

    # KV pool
    kv_pool_size = batch_groups * kv_len
    k = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)
    v = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)

    # req_to_tokens: [batch, max_kv_len]
    max_kv_len = group_size
    req_to_tokens = torch.zeros((batch_size, max_kv_len), dtype=torch.int32, device=device)

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Q: [batch_size, num_heads, head_dim] - 每个请求 1 个 Q token
    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device=device)

    # Setup KV indices
    for group_idx in range(batch_groups):
        group_start = group_idx * group_size
        kv_base = group_idx * kv_len

        for member_idx in range(group_size):
            batch_idx = group_start + member_idx
            # 第 member_idx 个请求可以看到前 member_idx+1 个 KV
            b_seq_len[batch_idx] = member_idx + 1

            # Setup KV indices - 组内请求共享相同的 KV 前缀
            for kv_pos in range(member_idx + 1):
                req_to_tokens[batch_idx, kv_pos] = kv_base + kv_pos

    # b_mark_shared_group: 组内最后一个请求标记组大小，其他为 0
    b_mark_shared_group = torch.zeros(batch_size, dtype=torch.int32, device=device)
    for group_idx in range(batch_groups):
        group_start = group_idx * group_size
        group_end = group_start + group_size - 1
        b_mark_shared_group[group_end] = group_size

    return q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group


def reference_attention(q, k, v, seq_len):
    """
    Reference implementation: compute attention for a single Q against first seq_len KVs.

    Args:
        q: [num_heads, head_dim]
        k: [seq_len, kv_head_num, head_dim]
        v: [seq_len, kv_head_num, head_dim]
        seq_len: number of visible KVs

    Returns:
        out: [num_heads, head_dim]
    """
    num_heads = q.shape[0]
    kv_head_num = k.shape[1]
    head_dim = q.shape[1]
    gqa_group_size = num_heads // kv_head_num

    sm_scale = 1.0 / (head_dim ** 0.5)

    out = torch.zeros_like(q, dtype=torch.float32)

    for h in range(num_heads):
        kv_h = h // gqa_group_size
        q_h = q[h].float()  # [head_dim]
        k_h = k[:seq_len, kv_h].float()  # [seq_len, head_dim]
        v_h = v[:seq_len, kv_h].float()  # [seq_len, head_dim]

        # Compute attention scores: [seq_len]
        scores = torch.matmul(q_h, k_h.T) * sm_scale  # [seq_len]

        # Softmax
        scores_max = scores.max(dim=-1, keepdim=True).values
        scores_exp = torch.exp(scores - scores_max)
        scores_sum = scores_exp.sum()
        scores_norm = scores_exp / scores_sum

        # Weighted sum: [head_dim]
        out[h] = torch.matmul(scores_norm, v_h)

    return out


def test_mtp_diverse_correctness():
    """Test MTP diverse attention correctness against reference implementation"""
    print("=" * 80)
    print("Testing MTP Diverse Attention Correctness")
    print("=" * 80)

    kv_len = 5  # Small for easy verification
    group_size = 2  # 2 requests per group: [q1, q2] with seq_len [1, 2]
    batch_groups = 4  # 4 groups for simplicity

    q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group = setup_mtp_test_data(
        kv_len, group_size, batch_groups, test_dtype=torch.float32, device="cuda", seed=42
    )

    print(f"kv_len: {kv_len}, group_size: {group_size}, batch_groups: {batch_groups}")
    print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
    print(f"b_seq_len: {b_seq_len}")
    print(f"b_mark_shared_group: {b_mark_shared_group}")
    print(f"req_to_tokens:\n{req_to_tokens}")

    batch_size = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    max_kv_len = b_seq_len.max().item()
    block_seq = 256

    # Run MTP diverse attention
    from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse.mtp_diverse_attn import (
        token_decode_attention_mtp_diverse_single_token
    )

    mtp_output = token_decode_attention_mtp_diverse_single_token(
        q=q,
        k=k,
        v=v,
        Req_to_tokens=req_to_tokens,
        B_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        block_seq=block_seq,
    )

    print(f"\nMTP Output shape: {mtp_output.shape}")

    # Compute reference output for each request
    print("\nComputing reference implementation...")
    reference_outputs = []
    for batch_idx in range(batch_size):
        seq_len = b_seq_len[batch_idx].item()
        q_single = q[batch_idx]  # [num_heads, head_dim]

        # Get KV indices for this request
        kv_indices = req_to_tokens[batch_idx, :seq_len]  # [seq_len]

        # Gather K and V
        k_single = k[kv_indices]  # [seq_len, kv_head_num, head_dim]
        v_single = v[kv_indices]  # [seq_len, kv_head_num, head_dim]

        ref_out = reference_attention(q_single, k_single, v_single, seq_len)
        reference_outputs.append(ref_out)

    reference_tensor = torch.stack(reference_outputs)  # [batch_size, num_heads, head_dim]

    print(f"Reference Output shape: {reference_tensor.shape}")

    # Compare MTP output with reference
    print("\n" + "=" * 80)
    print("Comparison Results:")
    print("=" * 80)

    max_diff = (mtp_output - reference_tensor).abs().max().item()
    mean_diff = (mtp_output - reference_tensor).abs().mean().item()

    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    # Check if results match (within floating point tolerance)
    # 对于 float32 的 flash attention，1e-2 是合理的 tolerance
    tolerance = 1e-2
    if max_diff < tolerance:
        print(f"\n✓ SUCCESS: MTP output matches reference (max_diff={max_diff:.6e} < {tolerance})")
        return True
    else:
        print(f"\n✗ FAILURE: MTP output does NOT match reference (max_diff={max_diff:.6e} >= {tolerance})")

        # Print per-request differences
        print("\nPer-request max differences:")
        for batch_idx in range(batch_size):
            req_diff = (mtp_output[batch_idx] - reference_tensor[batch_idx]).abs().max().item()
            print(f"  Request {batch_idx} (seq_len={b_seq_len[batch_idx].item()}): {req_diff:.6e}")

        return False


if __name__ == "__main__":
    success = test_mtp_diverse_correctness()
    sys.exit(0 if success else 1)
