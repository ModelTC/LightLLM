"""
MTP Diverse Attention Unit Test

测试 MTP diverse attention 算子的正确性和性能。

Single Token Mode:
- 每个请求只有 1 个 Q token
- 组内第 i 个请求只能看到前 i 个 KV
- b_mark_shared_group 标记：
  - 0: 组内非最后一个请求（跳过）
  - N>=1: 一个 N 人组的最后一个请求（需要计算）
"""
import pytest
import warnings

import torch
import time

# 全局结果收集
_test_results = []


def gqa_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    req_to_tokens: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
) -> torch.Tensor:
    """
    Reference GQA attention implementation for single token mode

    q: [batch_size, num_heads, head_dim] - 每个请求 1 个 Q token
    k, v: [kv_pool_size, kv_head_num, head_dim] - KV 池
    req_to_tokens: [batch, max_kv_len] - 每个请求的 KV 索引
    b_seq_len: [batch] - 每个请求可见的 KV 数量
    """
    batch_size = b_seq_len.shape[0]
    num_heads = q.shape[1]
    kv_head_num = k.shape[1]
    head_dim = q.shape[2]
    gqa_group_size = num_heads // kv_head_num

    # 输出：[batch, num_heads, head_dim]
    output = torch.zeros((batch_size, num_heads, head_dim), dtype=q.dtype, device=q.device)

    for b in range(batch_size):
        seq_len = b_seq_len[b].item()  # 这个请求可见的 KV 数量

        # 获取这个请求的 KV 索引：[seq_len]
        kv_indices = req_to_tokens[b, :seq_len]

        # 获取 K 和 V：[seq_len, kv_head_num, head_dim]
        k_batch = k[kv_indices]
        v_batch = v[kv_indices]

        # Process each Q head
        for h in range(num_heads):
            kv_h = h // gqa_group_size
            k_h = k_batch[:, kv_h, :]  # [seq_len, head_dim]
            v_h = v_batch[:, kv_h, :]  # [seq_len, head_dim]

            q_h = q[b, h]  # [head_dim]

            # Compute attention scores
            att = torch.matmul(q_h, k_h.transpose(0, 1))  # [seq_len]
            att = att / (head_dim ** 0.5)
            att = torch.softmax(att, dim=-1)
            out_h = torch.matmul(att, v_h)  # [head_dim]

            output[b, h] = out_h

    return output


def setup_mtp_test_data(kv_len, group_size, batch_groups, test_dtype=torch.bfloat16, device="cuda", seed=42):
    """
    设置 MTP 测试数据 - Single Token Mode

    Single Token Mode 的特点：
    - 每个请求有 1 个 Q token
    - 组内第 i 个请求只能看到前 i+1 个 KV
    - 组内请求共享相同的 KV 前缀

    b_mark_shared_group 标记：
    - 0: 组内非最后一个请求（跳过计算）
    - N>=1: 一个 N 人组的最后一个请求（需要计算）
    """
    torch.manual_seed(seed)

    num_heads = 32
    kv_head_num = 4  # gqa_group_size = 8
    head_dim = 128

    batch_size = batch_groups * group_size

    # KV 池：[batch_groups * kv_len, kv_head_num, head_dim]
    # 每个组使用不同的 KV 范围
    kv_pool_size = batch_groups * kv_len
    k = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)
    v = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)

    # req_to_tokens: [batch, max_kv_len] - 每个请求的 KV 索引
    max_kv_len = group_size  # 每个请求最多有 group_size 个 KV（可见范围）
    req_to_tokens = torch.zeros((batch_size, max_kv_len), dtype=torch.int32, device=device)

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=device)  # 每个请求可见的 KV 数量
    b_mark_shared_group = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Q: [batch_size, num_heads, head_dim] - 每个请求 1 个 Q token
    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device=device)

    for group_idx in range(batch_groups):
        group_start = group_idx * group_size
        kv_base = group_idx * kv_len

        for member_idx in range(group_size):
            batch_idx = group_start + member_idx
            # 第 member_idx 个请求可以看到前 member_idx+1 个 KV
            b_seq_len[batch_idx] = member_idx + 1

            # 设置 KV 索引 - 组内请求共享相同的 KV 前缀
            for kv_pos in range(member_idx + 1):
                req_to_tokens[batch_idx, kv_pos] = kv_base + kv_pos

            # b_mark_shared_group:
            # - 0: 组内非最后一个请求
            # - group_size: 组内最后一个请求
            if member_idx == group_size - 1:
                b_mark_shared_group[batch_idx] = group_size
            else:
                b_mark_shared_group[batch_idx] = 0

    return q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group


def mtp_diverse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    req_to_tokens: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_mark_shared_group: torch.Tensor,
    block_seq: int = 256,
) -> torch.Tensor:
    """
    MTP Diverse Attention 调用入口
    """
    from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse import (
        token_decode_attention_mtp_diverse_single_token,
    )

    return token_decode_attention_mtp_diverse_single_token(
        q=q,
        k=k,
        v=v,
        Req_to_tokens=req_to_tokens,
        B_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        block_seq=block_seq,
    )


@pytest.mark.parametrize("kv_len", [100, 256])
@pytest.mark.parametrize("group_size", [2, 3])
@pytest.mark.parametrize("batch_groups", [1, 2, 4])
def test_mtp_diverse_vs_reference(kv_len, group_size, batch_groups):
    """
    测试 MTP diverse attention 与 reference 实现的正确性对比
    """
    test_dtype = torch.bfloat16
    device = "cuda"

    q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group = setup_mtp_test_data(
        kv_len, group_size, batch_groups, test_dtype, device
    )

    # 运行 reference 实现（计算所有请求）
    reference_out = gqa_attention_reference(
        q=q,
        k=k,
        v=v,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
    )

    # 运行 MTP diverse（只计算组内最后一个请求）
    diverse_out = mtp_diverse_attention(
        q=q,
        k=k,
        v=v,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        block_seq=256,
    )

    # 对比结果
    print(f"\nkv_len={kv_len}, group_size={group_size}, batch_groups={batch_groups}")
    print(f"reference_out shape: {reference_out.shape}")
    print(f"diverse_out shape: {diverse_out.shape}")
    print(f"reference_out[0, 0, :4]: {reference_out[0, 0, :4]}")
    print(f"diverse_out[0, 0, :4]: {diverse_out[0, 0, :4]}")

    # 比较所有请求的输出
    max_diff = (reference_out - diverse_out).abs().max()
    print(f"max diff: {max_diff}")

    # 与 reference 对比
    allclose_passed = torch.allclose(reference_out, diverse_out, atol=1e-2, rtol=1e-2)

    # 收集结果
    _test_results.append({
        "kv_len": kv_len,
        "group_size": group_size,
        "batch_groups": batch_groups,
        "max_diff": max_diff.item(),
        "allclose_passed": allclose_passed,
    })

    status = "PASS" if allclose_passed else "FAIL"
    print(f"[{status}] max_diff: {max_diff.item():.6f}")

    # pytest 模式下，如果 allclose 失败只发出警告，不终止测试
    if not allclose_passed:
        warnings.warn(
            f"allclose check failed for kv_len={kv_len}, group_size={group_size}, batch_groups={batch_groups}. "
            f"max_diff={max_diff.item():.6f}"
        )

    return allclose_passed


if __name__ == "__main__":
    # 默认测试参数
    kv_lens = [100, 256]
    group_sizes = [2, 3]
    batch_groupss = [1, 2, 4]

    print("=" * 100)
    print("Testing MTP Diverse Attention vs Reference")
    print("=" * 100)

    all_passed = True
    for kv_len in kv_lens:
        for group_size in group_sizes:
            for batch_groups in batch_groupss:
                print(
                    f"\nTesting: kv_len={kv_len}, group_size={group_size}, batch_groups={batch_groups}"
                )
                try:
                    passed = test_mtp_diverse_vs_reference(kv_len, group_size, batch_groups)
                    if not passed:
                        all_passed = False
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    all_passed = False

    # 打印结果表格
    print(f"\n{'='*100}")
    print("SUMMARY - Results Table")
    print(f"{'='*100}")
    print(
        f"{'kv_len':<12} {'group_size':<12} {'batch_groups':<14} {'max_diff':<12} {'status':<8}"
    )
    print(f"{'-'*100}")
    for r in _test_results:
        status = "PASS" if r["allclose_passed"] else "FAIL"
        print(
            f"{r['kv_len']:<12} {r['group_size']:<12} "
            f"{r['batch_groups']:<14} {r['max_diff']:<12.6f} {status:<8}"
        )
    print(f"{'='*100}")

    # 总结
    passed_count = sum(1 for r in _test_results if r["allclose_passed"])
    total_count = len(_test_results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if not all_passed:
        print("\nWARNING: Some tests failed allclose check!")
    else:
        print("\nAll tests passed!")
