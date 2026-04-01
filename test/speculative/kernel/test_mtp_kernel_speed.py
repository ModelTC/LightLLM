"""
Test MTP Diverse Attention with FA3 - Speed Benchmark

Qwen3-32B 模型结构配置下，测试 MTP diverse attention 和 FA3 extend attention 的速度。

测试场景说明:
- 一个验证组包含 group_size 个请求
- 每个请求比前一个多 1 个 token (seq_len 递增)
- 只有最后一个请求 (b_mark_shared_group = group_size) 触发计算

MTP Diverse 方式:
- batch_size = group_size 个请求
- 每个请求 1 个 Q token
- 只有 b_marked != 0 的请求执行 kernel 计算

FA3 Extend 方式:
- 最后一个请求有 group_size 个 Q token
- 一次性 extend attention 处理所有 Q token
- 每个 Q token 对应不同的 KV 可见范围

例如 group_size=4, base_len=100:
组内请求：
- req_0: seq_len=100, b_marked=0 (跳过)
- req_1: seq_len=101, b_marked=0 (跳过)
- req_2: seq_len=102, b_marked=0 (跳过)
- req_3: seq_len=103, b_marked=4 (计算)

MTP Diverse: 4 个 batch，每个 1 个 Q，只有第 4 个 batch 计算
FA3 Extend:  1 个 batch，4 个 Q token，一次性处理
"""
import torch
import time
import os
import json

os.environ["PYTHONPATH"] = "/data/chenjunyi/project/lightllm"
os.environ["LIGHTLLM_START_ARGS"] = json.dumps(
    {
        "mtp_step": 4,
        "model_dir": "/tmp/test_model",
        "max_total_token_num": 100000,
    }
)

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse import (
    token_decode_attention_mtp_diverse_single_token,
)
from lightllm.utils.sgl_utils import flash_attn_with_kvcache


def setup_qwen3_32b_config():
    """Qwen3-32B 模型配置"""
    return {
        "num_heads": 64,       # Qwen3-32B 的 attention heads
        "kv_head_num": 8,      # GQA 的 KV heads，group size = 8
        "head_dim": 128,       # head dimension
        "gqa_group_size": 8,   # 64 / 8 = 8
    }


def setup_mtp_diverse_data(
    base_len,
    group_size,
    num_groups,
    config,
    test_dtype=torch.bfloat16,
    device="cuda",
    seed=42,
):
    """
    设置 MTP Diverse 测试数据

    MTP Diverse 模式:
    - 每组有 group_size 个请求
    - 组内第 i 个请求的 seq_len = base_len + i
    - 每个请求 1 个 Q token
    - 只有最后一个请求 (b_marked = group_size) 触发计算

    例如 group_size=4, base_len=100:
    - req_0: seq_len=100, b_marked=0 (跳过)
    - req_1: seq_len=101, b_marked=0 (跳过)
    - req_2: seq_len=102, b_marked=0 (跳过)
    - req_3: seq_len=103, b_marked=4 (计算，1 个 Q token)
    """
    torch.manual_seed(seed)

    num_heads = config["num_heads"]
    kv_head_num = config["kv_head_num"]
    head_dim = config["head_dim"]

    batch_size = num_groups * group_size

    # KV 池：每组需要 base_len + group_len 个 KV
    kv_per_group = base_len + group_size
    kv_pool_size = num_groups * kv_per_group

    k = torch.randn(
        size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device
    )
    v = torch.randn(
        size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device
    )

    # req_to_tokens: [batch, max_kv_len]
    max_kv_len = kv_per_group
    req_to_tokens = torch.zeros(
        (batch_size, max_kv_len), dtype=torch.int32, device=device
    )

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    b_mark_shared_group = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Q: [batch_size, num_heads, head_dim] - 每个请求 1 个 Q token
    q = torch.randn(
        size=(batch_size, num_heads, head_dim), dtype=test_dtype, device=device
    )

    # 设置每个验证组的数据
    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        kv_base = group_idx * kv_per_group

        for member_idx in range(group_size):
            batch_idx = group_start + member_idx

            # 组内第 i 个请求：seq_len = base_len + i
            b_seq_len[batch_idx] = base_len + member_idx

            # 设置 KV 索引
            for kv_pos in range(base_len + member_idx):
                req_to_tokens[batch_idx, kv_pos] = kv_base + kv_pos

            # b_mark_shared_group: 只有最后一个请求需要计算
            if member_idx == group_size - 1:
                b_mark_shared_group[batch_idx] = group_size
            else:
                b_mark_shared_group[batch_idx] = 0

    return {
        "q": q,                      # [batch, num_heads, head_dim]
        "k": k,                      # [kv_pool, kv_head_num, head_dim]
        "v": v,
        "req_to_tokens": req_to_tokens,
        "b_req_idx": b_req_idx,
        "b_seq_len": b_seq_len,
        "b_mark_shared_group": b_mark_shared_group,
        "config": config,
        "num_groups": num_groups,
        "group_size": group_size,
        "base_len": base_len,
    }


def setup_fa3_extend_data(
    base_len,
    group_size,
    num_groups,
    config,
    test_dtype=torch.bfloat16,
    device="cuda",
    seed=42,
):
    """
    设置 FA3 Extend 测试数据

    FA3 Extend 模式:
    - 每组只有 1 个请求（最后一个）
    - 这个请求有 group_size 个 Q token
    - 每个 Q token 对应不同的 KV 可见范围

    例如 group_size=4, base_len=100:
    - 1 个 batch，4 个 Q tokens
    - Q[0] 对 seq_len=100 的 KV 做 attention
    - Q[1] 对 seq_len=101 的 KV 做 attention
    - Q[2] 对 seq_len=102 的 KV 做 attention
    - Q[3] 对 seq_len=103 的 KV 做 attention
    """
    torch.manual_seed(seed)

    num_heads = config["num_heads"]
    kv_head_num = config["kv_head_num"]
    head_dim = config["head_dim"]

    # FA3: 每组只有 1 个 batch（最后一个请求）
    batch_size = num_groups

    # KV 池：每组需要 base_len + group_size 个 KV
    kv_per_group = base_len + group_size
    kv_pool_size = num_groups * kv_per_group

    k = torch.randn(
        size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device
    )
    v = torch.randn(
        size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device
    )

    # req_to_tokens: [batch, max_kv_len]
    max_kv_len = kv_per_group
    req_to_tokens = torch.zeros(
        (batch_size, max_kv_len), dtype=torch.int32, device=device
    )

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Q: [batch_size, group_size, num_heads, head_dim]
    # 每个请求有 group_size 个 Q token
    q = torch.randn(
        size=(batch_size, group_size, num_heads, head_dim), dtype=test_dtype, device=device
    )

    # 设置每个验证组的数据
    for group_idx in range(num_groups):
        kv_base = group_idx * kv_per_group

        # 只有最后一个请求，它的 seq_len 是最大的
        batch_idx = group_idx
        b_seq_len[batch_idx] = base_len + group_size - 1

        # 设置 KV 索引
        for kv_pos in range(base_len + group_size):
            req_to_tokens[batch_idx, kv_pos] = kv_base + kv_pos

    return {
        "q": q,                      # [batch, group_size, num_heads, head_dim]
        "k": k,                      # [kv_pool, kv_head_num, head_dim]
        "v": v,
        "req_to_tokens": req_to_tokens,
        "b_req_idx": b_req_idx,
        "b_seq_len": b_seq_len,
        "config": config,
        "num_groups": num_groups,
        "group_size": group_size,
        "base_len": base_len,
    }


def benchmark_mtp_diverse(data, block_seq=256, num_warmup=10, num_iters=100):
    """
    Benchmark MTP Diverse Attention - triton kernel 实现

    只计算 b_marked != 0 的位置
    """
    q = data["q"]
    k = data["k"]
    v = data["v"]
    req_to_tokens = data["req_to_tokens"]
    b_req_idx = data["b_req_idx"]
    b_seq_len = data["b_seq_len"]
    b_mark_shared_group = data["b_mark_shared_group"]

    # Warmup
    for _ in range(num_warmup):
        _ = token_decode_attention_mtp_diverse_single_token(
            q=q,
            k=k,
            v=v,
            Req_to_tokens=req_to_tokens,
            B_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mark_shared_group=b_mark_shared_group,
            block_seq=block_seq,
        )

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Benchmark
    for _ in range(num_iters):
        _ = token_decode_attention_mtp_diverse_single_token(
            q=q,
            k=k,
            v=v,
            Req_to_tokens=req_to_tokens,
            B_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mark_shared_group=b_mark_shared_group,
            block_seq=block_seq,
        )

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    avg_time_ms = (elapsed_time / num_iters) * 1000
    return avg_time_ms


def benchmark_fa3_extend(data, num_warmup=10, num_iters=100):
    """
    Benchmark FA3 Extend Attention

    FA3 extend attention 模式:
    - 每个 batch 有 group_size 个 Q token
    - 每个 Q token 对应不同的 KV 可见范围 (cumulative)
    - 一次性处理所有 Q token
    """
    q = data["q"]  # [batch, group_size, num_heads, head_dim]
    k = data["k"]
    v = data["v"]
    req_to_tokens = data["req_to_tokens"]
    b_seq_len = data["b_seq_len"]  # 每个 batch 的最大 seq_len
    group_size = data["group_size"]
    num_groups = data["num_groups"]
    base_len = data["base_len"]
    config = data["config"]

    num_heads = config["num_heads"]
    kv_head_num = config["kv_head_num"]
    head_dim = config["head_dim"]

    batch_size = num_groups

    # 构建 page_table: [batch, max_kv_len]
    max_kv_len = base_len + group_size
    page_table = req_to_tokens[:, :max_kv_len]

    # Q reshape for FA3: [batch * group_size, num_heads, head_dim]
    q_fa3 = q.reshape(batch_size * group_size, num_heads, head_dim)

    # cumulative lengths for FA3
    # cu_seqlens_q: 每个 batch 有 group_size 个 Q token
    cu_seqlens_q = torch.arange(
        0, batch_size * group_size + 1, group_size,
        dtype=torch.int32, device=q.device
    )

    # cu_seqlens_k_new: 每个 Q token 对应的 KV 累积长度
    # 对于每个 group，KV 长度是 base_len, base_len+1, ..., base_len+group_size-1
    # 所以 cumulative 是 base_len, base_len+(base_len+1), ...
    kv_lens = []
    for g in range(num_groups):
        for i in range(group_size):
            kv_lens.append(base_len + i)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=q.device)
    cu_seqlens_k_new = torch.cat([
        torch.tensor([0], dtype=torch.int32, device=q.device),
        kv_lens.cumsum(dim=0)
    ])

    # cache_seqlens: 每个 batch 的最大 KV 长度
    cache_seqlens = b_seq_len

    Lq = head_dim
    sm_scale = 1.0 / (Lq ** 0.5)

    # Warmup
    for _ in range(num_warmup):
        _ = flash_attn_with_kvcache(
            q=q_fa3,
            k_cache=k.view(k.shape[0], 1, k.shape[1], k.shape[2]),
            v_cache=v.view(v.shape[0], 1, v.shape[1], v.shape[2]),
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=group_size,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            sinks=None,
        )

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Benchmark
    for _ in range(num_iters):
        _ = flash_attn_with_kvcache(
            q=q_fa3,
            k_cache=k.view(k.shape[0], 1, k.shape[1], k.shape[2]),
            v_cache=v.view(v.shape[0], 1, v.shape[1], v.shape[2]),
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=group_size,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            sinks=None,
        )

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    avg_time_ms = (elapsed_time / num_iters) * 1000
    return avg_time_ms


def test_fa3_extend_vs_mtp_diverse():
    """
    对比 MTP Diverse Attention 和 FA3 Extend Attention 的速度
    """
    print("=" * 120)
    print("Testing MTP Diverse Attention vs FA3 Extend Attention - Speed Benchmark")
    print("Qwen3-32B Model Configuration")
    print("=" * 120)

    config = setup_qwen3_32b_config()
    print(f"\nModel Config: {config}")
    print("\nTest Mode:")
    print("- Each verification group has {group_size} requests")
    print("- Request i has seq_len = base_len + i")
    print("- Only the last request (b_marked != 0) computes attention")
    print("\nMTP Diverse:")
    print("- batch_size = group_size, each with 1 Q token")
    print("- Only b_marked != 0 requests execute kernel")
    print("\nFA3 Extend:")
    print("- batch_size = 1 (last request only)")
    print("- This request has {group_size} Q tokens")
    print("- Process all Q tokens in one extend attention call")

    # 测试参数
    test_configs = [
        # (base_len, group_size, num_groups)
        (1024, 2, 1),
        (1024, 4, 1),
        (1024, 8, 1),
        (2048, 2, 1),
        (2048, 4, 1),
        (2048, 8, 1),
        (4096, 2, 1),
        (4096, 4, 1),
        (4096, 8, 1),
    ]

    print(f"\n{'='*120}")
    print(f"{'base_len':<12} {'group_size':<12} {'diverse(ms)':<16} {'fa3(ms)':<16} {'speedup(fa3)':<14}")
    print(f"{'='*120}")

    results = []

    for base_len, group_size, num_groups in test_configs:
        print(f"\nTesting: base_len={base_len}, group_size={group_size}, num_groups={num_groups}")

        # Setup MTP diverse data
        mtp_data = setup_mtp_diverse_data(
            base_len=base_len,
            group_size=group_size,
            num_groups=num_groups,
            config=config,
        )

        # Setup FA3 extend data
        fa3_data = setup_fa3_extend_data(
            base_len=base_len,
            group_size=group_size,
            num_groups=num_groups,
            config=config,
        )

        print(f"  MTP: batch_size={mtp_data['q'].shape[0]}, active={(mtp_data['b_mark_shared_group'] > 0).sum().item()}")
        print(f"  FA3: batch_size={fa3_data['q'].shape[0]}, q_tokens_per_batch={fa3_data['q'].shape[1]}")

        # Benchmark MTP diverse
        diverse_time = benchmark_mtp_diverse(mtp_data, block_seq=256)

        # Benchmark FA3 extend
        fa3_time = benchmark_fa3_extend(fa3_data)

        speedup = diverse_time / fa3_time if fa3_time > 0 else float('inf')

        results.append({
            "base_len": base_len,
            "group_size": group_size,
            "num_groups": num_groups,
            "diverse_time_ms": diverse_time,
            "fa3_time_ms": fa3_time,
            "speedup": speedup,
        })

        print(f"  MTP Diverse: {diverse_time:.4f} ms, FA3 Extend: {fa3_time:.4f} ms, FA3 Speedup: {speedup:.2f}x")

    # 打印结果表格
    print(f"\n{'='*120}")
    print("SUMMARY - Speed Benchmark Results")
    print(f"{'='*120}")
    print(
        f"{'base_len':<12} {'group_size':<12} {'num_groups':<12} "
        f"{'diverse(ms)':<14} {'fa3(ms)':<14} {'speedup':<10}"
    )
    print(f"{'-'*120}")

    for r in results:
        print(
            f"{r['base_len']:<12} {r['group_size']:<12} {r['num_groups']:<12} "
            f"{r['diverse_time_ms']:<14.4f} {r['fa3_time_ms']:<14.4f} {r['speedup']:<10.2f}x"
        )

    print(f"{'='*120}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MTP Diverse Attention FA3 Speed Benchmark")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "compare"],
                        help="Which test to run")
    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

    if args.test in ["all", "compare"]:
        test_fa3_extend_vs_mtp_diverse()

    print("\nBenchmark completed!")
