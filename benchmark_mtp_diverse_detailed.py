"""
MTP Diverse Attention Detailed Performance Analysis

详细分析不同verify group大小对性能的影响，包括：
1. 固定总batch_size，比较不同group_size的性能
2. 固定kv_len，观察batch size变化趋势
3. 不同kv_len下的性能表现
4. 计算效率分析（实际计算量vs理论计算量）
"""
import torch
import time
import json
import sys
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/data/nvme0/chenjunyi/project/lightllm')

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse import (
    token_decode_attention_mtp_diverse_single_token,
)


def warmup_gpu(device="cuda"):
    """GPU warmup"""
    torch.cuda.synchronize(device)
    for _ in range(10):
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
    torch.cuda.synchronize(device)


def setup_mtp_test_data(kv_len: int, group_size: int, batch_groups: int,
                        num_heads: int = 32, kv_head_num: int = 4, head_dim: int = 128,
                        test_dtype=torch.bfloat16, device="cuda", seed=42):
    """设置 MTP 测试数据"""
    torch.manual_seed(seed)

    batch_size = batch_groups * group_size
    gqa_group_size = num_heads // kv_head_num

    kv_pool_size = batch_groups * kv_len
    k = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)
    v = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)

    max_kv_len = group_size
    req_to_tokens = torch.zeros((batch_size, max_kv_len), dtype=torch.int32, device=device)

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    b_mark_shared_group = torch.zeros(batch_size, dtype=torch.int32, device=device)

    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device=device)

    for group_idx in range(batch_groups):
        group_start = group_idx * group_size
        kv_base = group_idx * kv_len

        for member_idx in range(group_size):
            batch_idx = group_start + member_idx
            b_seq_len[batch_idx] = member_idx + 1

            for kv_pos in range(member_idx + 1):
                req_to_tokens[batch_idx, kv_pos] = kv_base + kv_pos

            if member_idx == group_size - 1:
                b_mark_shared_group[batch_idx] = group_size
            else:
                b_mark_shared_group[batch_idx] = 0

    return q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group


def benchmark_mtp_diverse(q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group,
                          block_seq: int = 256, warmup_iters: int = 20, test_iters: int = 200,
                          device="cuda"):
    """基准测试MTP diverse算子"""
    torch.cuda.synchronize(device)

    # Warmup
    for _ in range(warmup_iters):
        _ = token_decode_attention_mtp_diverse_single_token(
            q=q, k=k, v=v,
            Req_to_tokens=req_to_tokens,
            B_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mark_shared_group=b_mark_shared_group,
            block_seq=block_seq,
        )
    torch.cuda.synchronize(device)

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(test_iters):
        _ = token_decode_attention_mtp_diverse_single_token(
            q=q, k=k, v=v,
            Req_to_tokens=req_to_tokens,
            B_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mark_shared_group=b_mark_shared_group,
            block_seq=block_seq,
        )
    end_event.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / test_iters

    return avg_time_ms


def run_single_config(kv_len, group_size, batch_groups, num_heads, kv_head_num, head_dim, device):
    """运行单个配置测试"""
    try:
        q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group = setup_mtp_test_data(
            kv_len=kv_len,
            group_size=group_size,
            batch_groups=batch_groups,
            num_heads=num_heads,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            device=device,
        )

        avg_time_ms = benchmark_mtp_diverse(
            q=q, k=k, v=v,
            req_to_tokens=req_to_tokens,
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mark_shared_group=b_mark_shared_group,
            block_seq=256,
            warmup_iters=20,
            test_iters=200,
            device=device,
        )

        total_batch = group_size * batch_groups
        actual_computed = batch_groups  # 只有组内最后一个请求被计算
        tokens_per_sec = actual_computed / (avg_time_ms / 1000)

        # 计算理论计算量（浮点运算次数）
        # Attention计算: Q @ K^T: batch_groups * num_heads * group_size * head_dim * avg_kv_len
        # 平均KV长度 = (1 + group_size) / 2
        avg_kv_len = (1 + group_size) / 2
        gqa_group_size = num_heads // kv_head_num

        # FLOPs = 2 * num_heads * batch_groups * (QK + PV计算)
        # QK: batch_groups * num_heads * avg_kv_len * head_dim
        # PV: batch_groups * num_heads * head_dim * avg_kv_len
        flops_per_request = 2 * num_heads * avg_kv_len * head_dim * 2  # *2 for QK and PV
        total_flops = flops_per_request * batch_groups

        # 计算带宽（假设数据需要从HBM读取）
        # 读取: Q + K + V + indices
        # Q: batch_size * num_heads * head_dim
        # K, V: batch_groups * avg_kv_len * kv_head_num * head_dim * 2
        # indices: batch_size * avg_kv_len
        q_bytes = total_batch * num_heads * head_dim * 2  # bfloat16 = 2 bytes
        kv_bytes = batch_groups * avg_kv_len * kv_head_num * head_dim * 2 * 2  # K + V
        indices_bytes = total_batch * avg_kv_len * 4  # int32
        total_bytes = (q_bytes + kv_bytes + indices_bytes) / 1e9  # GB

        return {
            "kv_len": kv_len,
            "group_size": group_size,
            "batch_groups": batch_groups,
            "total_batch_size": total_batch,
            "avg_time_ms": avg_time_ms,
            "tokens_per_sec": tokens_per_sec,
            "total_flops": total_flops,
            "memory_gb": total_bytes,
            "avg_kv_len": avg_kv_len,
            "status": "SUCCESS",
        }
    except Exception as e:
        return {
            "kv_len": kv_len,
            "group_size": group_size,
            "batch_groups": batch_groups,
            "total_batch_size": group_size * batch_groups,
            "avg_time_ms": None,
            "tokens_per_sec": None,
            "status": f"FAILED: {str(e)}",
        }


def test_fixed_total_batch(fixed_total_batch: int = 96, kv_len: int = 1000):
    """
    测试固定总batch_size下，不同group_size的性能

    例如：总batch=96
    - group_size=6: batch_groups=16
    - group_size=12: batch_groups=8
    """
    print(f"\n{'='*80}")
    print(f"Test 1: Fixed Total Batch Size = {fixed_total_batch}, KV Len = {kv_len}")
    print(f"{'='*80}")

    results = []
    group_sizes = [1, 2, 3, 4, 6, 8, 12, 16, 24]

    print(f"\n{'group_size':<12} {'batch_groups':<14} {'total_batch':<12} {'time(ms)':<12} {'tokens/s':<12}")
    print("-" * 80)

    for gs in group_sizes:
        if fixed_total_batch % gs == 0:
            bg = fixed_total_batch // gs
            result = run_single_config(kv_len, gs, bg, 32, 4, 128, "cuda")
            results.append(result)
            if result["status"] == "SUCCESS":
                print(f"{gs:<12} {bg:<14} {result['total_batch_size']:<12} "
                      f"{result['avg_time_ms']:<12.3f} {result['tokens_per_sec']:<12.2f}")

    return results


def test_different_kv_lens(group_sizes=[6, 12], kv_lens=[256, 512, 1000, 2000, 4000]):
    """测试不同KV长度下的性能"""
    print(f"\n{'='*80}")
    print(f"Test 2: Different KV Lengths Impact")
    print(f"{'='*80}")

    results = []
    batch_groups_list = [1, 4, 16]

    print(f"\n{'kv_len':<10} {'group_size':<12} {'batch_groups':<14} {'time(ms)':<12} {'tokens/s':<12}")
    print("-" * 80)

    for kv_len in kv_lens:
        for gs in group_sizes:
            for bg in batch_groups_list:
                result = run_single_config(kv_len, gs, bg, 32, 4, 128, "cuda")
                results.append(result)
                if result["status"] == "SUCCESS":
                    print(f"{kv_len:<10} {gs:<12} {bg:<14} "
                          f"{result['avg_time_ms']:<12.3f} {result['tokens_per_sec']:<12.2f}")

    return results


def test_batch_scaling(group_sizes=[6, 12], kv_len=1000):
    """测试batch size扩展性能"""
    print(f"\n{'='*80}")
    print(f"Test 3: Batch Size Scaling (KV Len = {kv_len})")
    print(f"{'='*80}")

    results = []
    batch_groups_list = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f"\n{'group_size':<12} {'batch_groups':<14} {'total_batch':<12} {'time(ms)':<12} "
          f"{'tokens/s':<12} {'efficiency':<12}")
    print("-" * 90)

    for gs in group_sizes:
        base_time = None
        for bg in batch_groups_list:
            result = run_single_config(kv_len, gs, bg, 32, 4, 128, "cuda")
            results.append(result)
            if result["status"] == "SUCCESS":
                if base_time is None:
                    base_time = result["avg_time_ms"]
                    efficiency = 1.0
                else:
                    # 理想情况下，batch扩大N倍，时间也应该扩大N倍
                    expected_time = base_time * bg
                    actual_time = result["avg_time_ms"]
                    efficiency = expected_time / actual_time if actual_time > 0 else 0

                print(f"{gs:<12} {bg:<14} {result['total_batch_size']:<12} "
                      f"{result['avg_time_ms']:<12.3f} {result['tokens_per_sec']:<12.2f} "
                      f"{efficiency:<12.2f}")

    return results


def test_varying_group_sizes(kv_len=1000, batch_groups=16):
    """测试不同group_size的详细对比"""
    print(f"\n{'='*80}")
    print(f"Test 4: Varying Group Sizes (KV Len = {kv_len}, Batch Groups = {batch_groups})")
    print(f"{'='*80}")

    results = []
    group_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]

    print(f"\n{'group_size':<12} {'total_batch':<12} {'avg_kv_len':<12} {'time(ms)':<12} "
          f"{'tokens/s':<12} {'latency/token':<14}")
    print("-" * 90)

    for gs in group_sizes:
        result = run_single_config(kv_len, gs, batch_groups, 32, 4, 128, "cuda")
        results.append(result)
        if result["status"] == "SUCCESS":
            latency_per_token = result["avg_time_ms"] / batch_groups * 1000  # us
            print(f"{gs:<12} {result['total_batch_size']:<12} {result['avg_kv_len']:<12.1f} "
                  f"{result['avg_time_ms']:<12.3f} {result['tokens_per_sec']:<12.2f} "
                  f"{latency_per_token:<14.2f}")

    return results


def analyze_results(results: List[Dict]):
    """分析测试结果"""
    print(f"\n{'='*80}")
    print("Analysis Summary")
    print(f"{'='*80}")

    # 按测试分组
    from collections import defaultdict
    by_kv_gs = defaultdict(list)

    for r in results:
        if r["status"] == "SUCCESS":
            key = (r["kv_len"], r["group_size"])
            by_kv_gs[key].append(r)

    # 计算扩展效率
    print("\nScaling Efficiency (theoretical linear scaling = 1.0):")
    print(f"{'kv_len':<10} {'group_size':<12} {'1->4x':<10} {'4->16x':<10} {'1->16x':<10}")
    print("-" * 60)

    for (kv_len, gs), items in sorted(by_kv_gs.items()):
        items_by_bg = {r["batch_groups"]: r for r in items}

        eff_1_4 = "N/A"
        eff_4_16 = "N/A"
        eff_1_16 = "N/A"

        if 1 in items_by_bg and 4 in items_by_bg:
            t1 = items_by_bg[1]["avg_time_ms"]
            t4 = items_by_bg[4]["avg_time_ms"]
            eff_1_4 = f"{(t1 * 4 / t4):.2f}"

        if 4 in items_by_bg and 16 in items_by_bg:
            t4 = items_by_bg[4]["avg_time_ms"]
            t16 = items_by_bg[16]["avg_time_ms"]
            eff_4_16 = f"{(t4 * 4 / t16):.2f}"

        if 1 in items_by_bg and 16 in items_by_bg:
            t1 = items_by_bg[1]["avg_time_ms"]
            t16 = items_by_bg[16]["avg_time_ms"]
            eff_1_16 = f"{(t1 * 16 / t16):.2f}"

        print(f"{kv_len:<10} {gs:<12} {eff_1_4:<10} {eff_4_16:<10} {eff_1_16:<10}")


def export_results(results: List[Dict], filename: str = "mtp_diverse_detailed_results.json"):
    """导出结果"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to {filename}")


def main():
    """主函数"""
    print("MTP Diverse Attention Detailed Performance Analysis")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Warmup
    print("\nWarming up GPU...")
    warmup_gpu(device)

    all_results = []

    # Test 1: Fixed total batch
    results1 = test_fixed_total_batch(fixed_total_batch=96, kv_len=1000)
    all_results.extend(results1)

    # Test 2: Different KV lengths
    results2 = test_different_kv_lens(group_sizes=[6, 12], kv_lens=[256, 512, 1000, 2000, 4000])
    all_results.extend(results2)

    # Test 3: Batch scaling
    results3 = test_batch_scaling(group_sizes=[6, 12], kv_len=1000)
    all_results.extend(results3)

    # Test 4: Varying group sizes
    results4 = test_varying_group_sizes(kv_len=1000, batch_groups=16)
    all_results.extend(results4)

    # Analysis
    analyze_results(all_results)

    # Export
    export_results(all_results)

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
