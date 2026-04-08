"""
MTP Diverse Triton Kernel 性能基准测试 - 动态 MTP vs 静态 MTP vs FA3

测试目标:
- 动态 MTP (Triton): 验证组大小为 2 个 token
- 静态 MTP (Triton): 验证组大小固定为 4/8/12 个 token
- FA3 Decode: 每个验证组做 group_size 次 decode attention

关键：num_groups 相同，total_reqs 不同
- 动态 MTP: total_reqs = num_groups × 2
- 静态 MTP: total_reqs = num_groups × 4/8/12
- FA3: total_reqs = num_groups (每个请求做 group_size 次 attention)

模型配置：Qwen3-8B
- num_attention_heads: 32
- num_key_value_heads: 8
- head_dim: 128
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


def setup_qwen3_8b_config():
    """Qwen3-8B 模型配置"""
    return {
        "num_heads": 32,
        "kv_head_num": 8,
        "head_dim": 128,
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
    设置 MTP Diverse 测试数据 (Triton kernel 用)

    num_groups 相同，total_reqs 不同:
    - 动态 MTP (gs=2): num_groups × 2 个请求
    - 静态 MTP (gs=8): num_groups × 8 个请求
    """
    torch.manual_seed(seed)

    num_heads = config["num_heads"]
    kv_head_num = config["kv_head_num"]
    head_dim = config["head_dim"]

    batch_size = num_groups * group_size
    kv_per_group = base_len + group_size
    kv_pool_size = num_groups * kv_per_group

    k = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)
    v = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)

    max_kv_len = kv_per_group
    req_to_tokens = torch.zeros((batch_size, max_kv_len), dtype=torch.int32, device=device)

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    b_mark_shared_group = torch.zeros(batch_size, dtype=torch.int32, device=device)

    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device=device)

    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        kv_base = group_idx * kv_per_group

        for member_idx in range(group_size):
            batch_idx = group_start + member_idx
            b_seq_len[batch_idx] = base_len + member_idx

            for kv_pos in range(base_len + member_idx):
                req_to_tokens[batch_idx, kv_pos] = kv_base + kv_pos

            if member_idx == group_size - 1:
                b_mark_shared_group[batch_idx] = group_size
            else:
                b_mark_shared_group[batch_idx] = 0

    return {
        "q": q,
        "k": k,
        "v": v,
        "req_to_tokens": req_to_tokens,
        "b_req_idx": b_req_idx,
        "b_seq_len": b_seq_len,
        "b_mark_shared_group": b_mark_shared_group,
        "config": config,
        "num_groups": num_groups,
        "group_size": group_size,
        "base_len": base_len,
        "total_requests": batch_size,
    }


def setup_fa3_data(
    base_len,
    group_size,
    num_groups,
    config,
    test_dtype=torch.bfloat16,
    device="cuda",
    seed=42,
):
    """
    设置 FA3 测试数据

    FA3 decode 模式：每个验证组做 group_size 次 attention
    - num_groups 个请求
    - 每个请求连续做 group_size 次 decode attention
    - 每次 attention 的 KV 可见范围递增
    """
    torch.manual_seed(seed)

    num_heads = config["num_heads"]
    kv_head_num = config["kv_head_num"]
    head_dim = config["head_dim"]

    # FA3: num_groups 个请求，每个请求做 group_size 次 attention
    kv_per_group = base_len + group_size
    kv_pool_size = num_groups * kv_per_group

    k = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)
    v = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)

    max_kv_len = kv_per_group
    req_to_tokens = torch.zeros((num_groups, max_kv_len), dtype=torch.int32, device=device)

    b_seq_len = torch.zeros(num_groups, dtype=torch.int32, device=device)

    # Q: [num_groups, num_heads, head_dim]
    q = torch.randn(size=(num_groups, num_heads, head_dim), dtype=test_dtype, device=device)

    for group_idx in range(num_groups):
        kv_base = group_idx * kv_per_group
        b_seq_len[group_idx] = base_len + group_size - 1

        for kv_pos in range(base_len + group_size):
            req_to_tokens[group_idx, kv_pos] = kv_base + kv_pos

    return {
        "q": q,
        "k": k,
        "v": v,
        "req_to_tokens": req_to_tokens,
        "config": config,
        "num_groups": num_groups,
        "group_size": group_size,
        "base_len": base_len,
    }


def benchmark_mtp_diverse(data, block_seq=256, num_warmup=10, num_iters=100):
    """Benchmark MTP Diverse Attention (Triton kernel)"""
    q = data["q"]
    k = data["k"]
    v = data["v"]
    req_to_tokens = data["req_to_tokens"]
    b_req_idx = data["b_req_idx"]
    b_seq_len = data["b_seq_len"]
    b_mark_shared_group = data["b_mark_shared_group"]

    for _ in range(num_warmup):
        _ = token_decode_attention_mtp_diverse_single_token(
            q=q, k=k, v=v,
            Req_to_tokens=req_to_tokens,
            B_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mark_shared_group=b_mark_shared_group,
            block_seq=block_seq,
        )

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iters):
        _ = token_decode_attention_mtp_diverse_single_token(
            q=q, k=k, v=v,
            Req_to_tokens=req_to_tokens,
            B_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mark_shared_group=b_mark_shared_group,
            block_seq=block_seq,
        )

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time
    return (elapsed_time / num_iters) * 1000


def benchmark_fa3_decode(data, num_warmup=10, num_iters=100):
    """
    Benchmark FA3 Decode Attention

    FA3 decode 模式：每个验证组做 group_size 次 attention
    每次 attention 处理 1 个 Q token，KV 可见范围递增

    使用 flash_attn_with_kvcache 的正确调用方式
    """
    q = data["q"]  # [num_groups, num_heads, head_dim]
    k = data["k"]  # [kv_pool, kv_head_num, head_dim]
    v = data["v"]
    req_to_tokens = data["req_to_tokens"]
    group_size = data["group_size"]
    num_groups = data["num_groups"]
    base_len = data["base_len"]
    config = data["config"]

    head_dim = config["head_dim"]
    kv_head_num = config["kv_head_num"]

    # 为 FA3 准备数据
    # FA3 decode 需要：batch_size=num_groups, 每个请求做 group_size 次 attention
    # Q reshape: [num_groups * group_size, num_heads, head_dim]
    q_expand = q.repeat_interleave(group_size, dim=0).contiguous()  # [num_groups * group_size, num_heads, head_dim]

    # k/v cache 形状：[kv_pool, 1, kv_head_num, head_dim]
    k_cache = k.view(k.shape[0], 1, kv_head_num, head_dim).contiguous()
    v_cache = v.view(v.shape[0], 1, kv_head_num, head_dim).contiguous()

    # page_table: [num_groups, max_kv_len]
    max_kv_len = base_len + group_size
    page_table = req_to_tokens.contiguous()

    # cache_seqlens: [num_groups] 每个请求的最大 KV 长度
    cache_seqlens = torch.full((num_groups,), base_len + group_size - 1, dtype=torch.int32, device=q.device)

    # cu_seqlens_q: 累积 Q 长度 [0, group_size, 2*group_size, ...]
    cu_seqlens_q = torch.arange(0, num_groups * group_size + 1, group_size, dtype=torch.int32, device=q.device)

    # cu_seqlens_k_new: 每个 Q token 对应的 KV 累积长度
    # 对于每个 group，KV 长度是 base_len, base_len+1, ..., base_len+group_size-1
    kv_lens = []
    for i in range(group_size):
        for _ in range(num_groups):
            kv_lens.append(base_len + i)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=q.device)
    cu_seqlens_k_new = torch.cat([
        torch.tensor([0], dtype=torch.int32, device=q.device),
        kv_lens.cumsum(dim=0)
    ])

    Lq = head_dim
    sm_scale = 1.0 / (Lq ** 0.5)

    def run_fa3():
        _ = flash_attn_with_kvcache(
            q=q_expand,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=1,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            sinks=None,
        )

    # Warmup
    for _ in range(num_warmup):
        run_fa3()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Benchmark
    for _ in range(num_iters):
        run_fa3()

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time
    return (elapsed_time / num_iters) * 1000


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = setup_qwen3_8b_config()
    print(f"模型：Qwen3-8B (num_heads={config['num_heads']}, kv_head_num={config['kv_head_num']}, head_dim={config['head_dim']})")

    base_len = 1024
    dynamic_gs = 2      # 动态 MTP: 每次验证 2 个 token
    static_gs_list = [1, 2, 3, 4]  # 静态 MTP: 每次验证 1, 2, 3, 4 个 token
    num_groups_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    print(f"\nbase_len={base_len}, 动态 MTP group_size={dynamic_gs}")
    print(f"静态 MTP group_size={static_gs_list}")
    print(f"FA3 group_size=与静态 MTP 相同 (每个验证组做 static_gs 次 decode)")
    print(f"\n注意：num_groups 相同，total_reqs 不同")
    print(f"  动态 MTP: total_reqs = num_groups × {dynamic_gs}")
    print(f"  静态 MTP: total_reqs = num_groups × static_gs")
    print(f"  FA3:      total_reqs = num_groups (但每个请求做 {dynamic_gs} 次 attention)")

    for static_gs in static_gs_list:
        print(f"\n{'='*140}")
        print(f"动态 MTP (gs={dynamic_gs}) vs 静态 MTP (gs={static_gs}) vs FA3 (gs={static_gs})")
        print(f"{'='*140}")
        print(f"{'num_groups':<12} {'dyn_reqs':<10} {'stat_reqs':<10} {'fa3_reqs':<10} {'dyn(ms)':<10} {'stat(ms)':<10} {'fa3(ms)':<10} {'dyn<stat':<8} {'dyn<fa3'}")
        print(f"{'-'*140}")

        crossover_stat = None
        crossover_fa3 = None

        for num_groups in num_groups_list:
            # Setup data
            dynamic_data = setup_mtp_diverse_data(
                base_len=base_len,
                group_size=dynamic_gs,
                num_groups=num_groups,
                config=config,
            )

            static_data = setup_mtp_diverse_data(
                base_len=base_len,
                group_size=static_gs,
                num_groups=num_groups,
                config=config,
            )

            fa3_data = setup_fa3_data(
                base_len=base_len,
                group_size=static_gs,  # FA3 gs 与静态 MTP 相同
                num_groups=num_groups,
                config=config,
            )

            # Benchmark
            dynamic_time = benchmark_mtp_diverse(dynamic_data, block_seq=256)
            static_time = benchmark_mtp_diverse(static_data, block_seq=256)
            fa3_time = benchmark_fa3_decode(fa3_data)

            dynamic_total_reqs = num_groups * dynamic_gs
            static_total_reqs = num_groups * static_gs
            fa3_total_reqs = num_groups

            dyn_wins_stat = dynamic_time < static_time
            dyn_wins_fa3 = dynamic_time < fa3_time

            if dyn_wins_stat and crossover_stat is None:
                crossover_stat = num_groups
            if dyn_wins_fa3 and crossover_fa3 is None:
                crossover_fa3 = num_groups

            print(f"{num_groups:<12} {dynamic_total_reqs:<10} {static_total_reqs:<10} {fa3_total_reqs:<10} {dynamic_time:<10.4f} {static_time:<10.4f} {fa3_time:<10.4f} {str(dyn_wins_stat):<8} {dyn_wins_fa3}")

        print()
        if crossover_stat:
            print(f"--> 拐点 (vs 静态 MTP gs={static_gs}): num_groups >= {crossover_stat} 时动态 MTP 更快")
        else:
            print(f"--> 拐点 (vs 静态 MTP gs={static_gs}): 测试范围内动态 MTP 未超越静态 MTP")

        if crossover_fa3:
            print(f"--> 拐点 (vs FA3 gs={static_gs}): num_groups >= {crossover_fa3} 时动态 MTP 更快")
        else:
            print(f"--> 拐点 (vs FA3 gs={static_gs}): 测试范围内动态 MTP 未超越 FA3")

    print(f"\n{'='*140}")
    print("测试完成!")


if __name__ == "__main__":
    main()
