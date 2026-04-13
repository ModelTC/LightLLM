"""
MTP Diverse Attention Performance Benchmark

测试不同verify group大小和batch size下的性能表现。

关键参数：
- kv_len: KV缓存长度
- group_size (verify group): 验证组大小（组内请求数）
- batch_groups: 组的数量，总batch_size = group_size * batch_groups
"""
import torch
import time
import json
import sys
from typing import List, Tuple, Dict
import torch.cuda.profiler as profiler

# 添加项目路径
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
    """
    设置 MTP 测试数据

    Args:
        kv_len: 每个组的KV长度
        group_size: 验证组大小（组内请求数）
        batch_groups: 组的数量
        num_heads: Q头数
        kv_head_num: KV头数
        head_dim: 头维度
    """
    torch.manual_seed(seed)

    batch_size = batch_groups * group_size
    gqa_group_size = num_heads // kv_head_num

    # KV池：[batch_groups * kv_len, kv_head_num, head_dim]
    kv_pool_size = batch_groups * kv_len
    k = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)
    v = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)

    # req_to_tokens: [batch, max_kv_len] - 每个请求的KV索引
    max_kv_len = group_size  # 每个请求最多可见group_size个KV
    req_to_tokens = torch.zeros((batch_size, max_kv_len), dtype=torch.int32, device=device)

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    b_mark_shared_group = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Q: [batch_size, num_heads, head_dim] - 每个请求1个Q token
    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device=device)

    for group_idx in range(batch_groups):
        group_start = group_idx * group_size
        kv_base = group_idx * kv_len

        for member_idx in range(group_size):
            batch_idx = group_start + member_idx
            # 第member_idx个请求可以看到前member_idx+1个KV
            b_seq_len[batch_idx] = member_idx + 1

            # 设置KV索引 - 组内请求共享相同的KV前缀
            for kv_pos in range(member_idx + 1):
                req_to_tokens[batch_idx, kv_pos] = kv_base + kv_pos

            # b_mark_shared_group: 0表示非最后一个，N表示组内最后一个
            if member_idx == group_size - 1:
                b_mark_shared_group[batch_idx] = group_size
            else:
                b_mark_shared_group[batch_idx] = 0

    return q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group


def benchmark_mtp_diverse(q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group,
                          block_seq: int = 256, warmup_iters: int = 10, test_iters: int = 100,
                          device="cuda"):
    """
    基准测试MTP diverse算子

    Returns:
        avg_time_ms: 平均执行时间(ms)
    """
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


def run_benchmark_suite(configs: List[Dict], num_heads: int = 32, kv_head_num: int = 4,
                        head_dim: int = 128, device="cuda") -> List[Dict]:
    """
    运行一系列基准测试

    Args:
        configs: 测试配置列表，每个配置包含kv_len, group_size, batch_groups
    """
    results = []

    # 首先warmup一次
    print("Warming up GPU...")
    warmup_gpu(device)

    for i, cfg in enumerate(configs):
        kv_len = cfg["kv_len"]
        group_size = cfg["group_size"]
        batch_groups = cfg["batch_groups"]
        batch_size = group_size * batch_groups

        print(f"\n[{i+1}/{len(configs)}] Testing: kv_len={kv_len}, group_size={group_size}, "
              f"batch_groups={batch_groups}, total_batch={batch_size}")

        try:
            # 准备数据
            q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group = setup_mtp_test_data(
                kv_len=kv_len,
                group_size=group_size,
                batch_groups=batch_groups,
                num_heads=num_heads,
                kv_head_num=kv_head_num,
                head_dim=head_dim,
                device=device,
            )

            # 运行测试
            avg_time_ms = benchmark_mtp_diverse(
                q=q, k=k, v=v,
                req_to_tokens=req_to_tokens,
                b_req_idx=b_req_idx,
                b_seq_len=b_seq_len,
                b_mark_shared_group=b_mark_shared_group,
                block_seq=256,
                warmup_iters=10,
                test_iters=100,
                device=device,
            )

            # 计算吞吐量指标
            # 实际计算的token数（只有组内最后一个请求被计算）
            actual_computed_tokens = batch_groups
            tokens_per_sec = actual_computed_tokens / (avg_time_ms / 1000)

            result = {
                "kv_len": kv_len,
                "group_size": group_size,
                "batch_groups": batch_groups,
                "total_batch_size": batch_size,
                "avg_time_ms": avg_time_ms,
                "tokens_per_sec": tokens_per_sec,
                "status": "SUCCESS",
            }

            print(f"  -> avg_time: {avg_time_ms:.3f} ms, throughput: {tokens_per_sec:.2f} tokens/s")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            result = {
                "kv_len": kv_len,
                "group_size": group_size,
                "batch_groups": batch_groups,
                "total_batch_size": batch_size,
                "avg_time_ms": None,
                "tokens_per_sec": None,
                "status": f"FAILED: {str(e)}",
            }

        results.append(result)

        # 清理显存
        torch.cuda.empty_cache()

    return results


def generate_test_configs(fixed_kv_len: int = 1000) -> List[Dict]:
    """
    生成测试配置

    主要对比：
    1. 固定kv_len=1000，对比group_size=6和group_size=12在不同batch_groups下的性能
    2. 逐渐增加batch_groups，观察性能变化趋势
    """
    configs = []

    # 测试组大小对比: 6 vs 12
    group_sizes = [6, 12]

    # batch_groups从1逐渐增加到32，总batch_size = group_size * batch_groups
    batch_groups_list = [1, 2, 4, 8, 16, 32, 64]

    for group_size in group_sizes:
        for batch_groups in batch_groups_list:
            configs.append({
                "kv_len": fixed_kv_len,
                "group_size": group_size,
                "batch_groups": batch_groups,
            })

    return configs


def print_results_table(results: List[Dict]):
    """打印结果表格"""
    print("\n" + "=" * 120)
    print("MTP Diverse Attention Performance Benchmark Results")
    print("=" * 120)
    print(f"{'kv_len':<10} {'group_size':<12} {'batch_groups':<14} {'total_batch':<12} "
          f"{'avg_time(ms)':<14} {'tokens/sec':<14} {'status':<10}")
    print("-" * 120)

    for r in results:
        if r["status"] == "SUCCESS":
            print(f"{r['kv_len']:<10} {r['group_size']:<12} {r['batch_groups']:<14} "
                  f"{r['total_batch_size']:<12} {r['avg_time_ms']:<14.3f} "
                  f"{r['tokens_per_sec']:<14.2f} {r['status']:<10}")
        else:
            print(f"{r['kv_len']:<10} {r['group_size']:<12} {r['batch_groups']:<14} "
                  f"{r['total_batch_size']:<12} {'N/A':<14} {'N/A':<14} {r['status']:<10}")

    print("=" * 120)


def analyze_group_size_impact(results: List[Dict]):
    """分析group_size对性能的影响"""
    print("\n" + "=" * 80)
    print("Group Size Impact Analysis")
    print("=" * 80)

    # 按batch_groups分组，比较group_size=6和group_size=12的性能
    from collections import defaultdict
    grouped = defaultdict(lambda: {})

    for r in results:
        if r["status"] == "SUCCESS":
            key = (r["kv_len"], r["batch_groups"])
            grouped[key][r["group_size"]] = r

    print(f"{'kv_len':<10} {'batch_groups':<14} {'group_size=6':<15} {'group_size=12':<15} {'ratio(12/6)':<12}")
    print("-" * 80)

    for (kv_len, batch_groups), sizes in sorted(grouped.items()):
        if 6 in sizes and 12 in sizes:
            time_6 = sizes[6]["avg_time_ms"]
            time_12 = sizes[12]["avg_time_ms"]
            ratio = time_12 / time_6 if time_6 > 0 else 0
            print(f"{kv_len:<10} {batch_groups:<14} {time_6:<15.3f} {time_12:<15.3f} {ratio:<12.2f}")

    print("=" * 80)


def export_results(results: List[Dict], filename: str = "mtp_diverse_benchmark_results.json"):
    """导出结果到JSON文件"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to {filename}")


def main():
    """主函数"""
    print("MTP Diverse Attention Performance Benchmark")
    print("=" * 80)

    # 检查GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # 模型配置 (Qwen3 8B配置)
    num_heads = 32
    kv_head_num = 4
    head_dim = 128

    print(f"\nModel Config: num_heads={num_heads}, kv_head_num={kv_head_num}, head_dim={head_dim}")

    # 生成测试配置
    configs = generate_test_configs(fixed_kv_len=1000)

    print(f"\nTotal test cases: {len(configs)}")
    print("\nTest configurations:")
    for cfg in configs:
        batch_size = cfg["group_size"] * cfg["batch_groups"]
        print(f"  kv_len={cfg['kv_len']}, group_size={cfg['group_size']}, "
              f"batch_groups={cfg['batch_groups']}, total_batch={batch_size}")

    # 运行测试
    print("\n" + "=" * 80)
    print("Starting benchmark...")
    print("=" * 80)

    results = run_benchmark_suite(
        configs=configs,
        num_heads=num_heads,
        kv_head_num=kv_head_num,
        head_dim=head_dim,
        device=device,
    )

    # 打印结果
    print_results_table(results)

    # 分析group_size影响
    analyze_group_size_impact(results)

    # 导出结果
    export_results(results)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
