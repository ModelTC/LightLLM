"""
MTP Kernel Performance Comparison Benchmark

比较不同 Decode kernel 在不同负载下的性能：
1. MTP Diverse (original) - 原始 MTP 实现
2. MTP Diverse VSM - VSM 优化 MTP 实现
3. Flash Decoding GQA - 普通 GQA Flash Decoding
4. Flash Decoding VSM - VSM 优化的 GQA Flash Decoding

实验设置基于 blog 描述：
- Llama-3: avg_input_len=1024, variance=256, max_len=2048
- DeepSeek-V2: avg_input_len=10240, variance=2560, max_len=128000
- 测试不同 batch size 下的性能表现
- 测试 Cuda Graph 启用/禁用的影响

Blog 关键观点：
- 原始 kernel 基于 attention heads 和 batch size 分配 thread blocks
- 当处理不均匀长度的 batch 时，thread blocks 负载不均衡，影响性能
- VSM 重新设计：固定 thread blocks 数量，将 context 分成固定大小的块
- 每个 thread block 迭代所有块，动态变化长度转为固定迭代次数
- 固定大小块确保负载平衡，处理不均匀长度 batch 性能更好
"""
import torch
import time
import json
import sys
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

# 添加项目路径
sys.path.insert(0, '/data/nvme0/chenjunyi/project/lightllm')

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse import (
    token_decode_attention_mtp_diverse_single_token,
)
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse.mtp_diverse_attn_vsm import (
    token_decode_attention_mtp_diverse_vsm_single_token,
)
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding import (
    gqa_token_decode_attention_flash_decoding,
)
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_vsm import (
    gqa_token_decode_attention_flash_decoding_vsm,
)


@dataclass
class TestConfig:
    """测试配置"""
    name: str
    batch_size: int
    avg_seq_len: int
    seq_len_variance: int
    max_seq_len: int
    num_heads: int = 32
    kv_head_num: int = 4
    head_dim: int = 128
    dtype: str = "bfloat16"
    # MTP 特定配置
    group_size: int = 1  # 1 表示普通 decode，>1 表示 MTP 组
    num_groups: int = 1  # 组的数量
    # 负载分布类型
    load_distribution: str = "normal"  # normal / uniform / outlier


@dataclass
class BenchmarkResult:
    """benchmark 结果"""
    config_name: str
    kernel_name: str
    batch_size: int
    avg_seq_len: float
    group_size: int
    avg_time_ms: float
    std_time_ms: float
    tokens_per_sec: float
    status: str
    error_msg: Optional[str] = None
    cuda_graph_enabled: bool = False


class SimpleInferState:
    """简单的 InferState 用于 Flash Decoding 测试"""
    def __init__(self, b_seq_len, req_to_tokens, b_req_idx, max_kv_seq_len=None):
        self.b_seq_len = b_seq_len
        self.total_token_num = b_seq_len.sum().item()
        self.max_kv_seq_len = max_kv_seq_len if max_kv_seq_len is not None else b_seq_len.max().item()
        self.batch_size = b_seq_len.shape[0]

        class ReqManager:
            def __init__(self, req_to_tokens, b_req_idx):
                self.req_to_token_indexs = req_to_tokens
                self.b_req_idx = b_req_idx

        self.req_manager = ReqManager(req_to_tokens, b_req_idx)
        self.b_req_idx = b_req_idx  # 直接暴露 b_req_idx


def warmup_gpu(device="cuda", num_warmup=20):
    """GPU warmup"""
    torch.cuda.synchronize(device)
    for _ in range(num_warmup):
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def generate_seq_lens(config: TestConfig, seed: int = 42) -> np.ndarray:
    """
    生成 seq_len 数组

    根据 config 中的 load_distribution 生成不同分布的 seq_len:
    - normal: 正态分布，均值和方差由 config 指定
    - uniform: 均匀分布
    - outlier: 大部分请求长度较短，少量异常长请求
    """
    np.random.seed(seed)
    batch_size = config.batch_size

    if config.load_distribution == "normal":
        # 正态分布
        seq_lens = np.random.normal(config.avg_seq_len, config.seq_len_variance, batch_size)
        seq_lens = np.clip(seq_lens, 1, config.max_seq_len).astype(int)

    elif config.load_distribution == "uniform":
        # 均匀分布
        min_len = max(1, config.avg_seq_len - config.seq_len_variance)
        max_len = min(config.max_seq_len, config.avg_seq_len + config.seq_len_variance)
        seq_lens = np.random.randint(min_len, max_len + 1, batch_size)

    elif config.load_distribution == "outlier":
        # 存在异常值的分布：90% 的请求长度为 avg_seq_len，10% 为 max_seq_len
        num_outliers = max(1, batch_size // 10)
        num_normal = batch_size - num_outliers

        normal_lens = np.full(num_normal, config.avg_seq_len, dtype=int)
        outlier_lens = np.full(num_outliers, config.max_seq_len, dtype=int)

        seq_lens = np.concatenate([normal_lens, outlier_lens])
        np.random.shuffle(seq_lens)

    else:
        raise ValueError(f"Unknown load_distribution: {config.load_distribution}")

    return seq_lens


def setup_mtp_diverse_data(config: TestConfig, device="cuda", seed=42):
    """
    设置 MTP Diverse 测试数据

    组内请求共享相同的 KV 前缀，每个请求 1 个 Q token，但可见 KV 范围不同
    """
    torch.manual_seed(seed)

    batch_size = config.batch_size
    num_heads = config.num_heads
    kv_head_num = config.kv_head_num
    head_dim = config.head_dim
    group_size = config.group_size
    num_groups = config.num_groups
    test_dtype = getattr(torch, config.dtype)

    # KV 池大小
    kv_pool_size = num_groups * config.max_seq_len
    k = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)
    v = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)

    # req_to_tokens: [batch, max_seq_len]
    max_kv_len = config.max_seq_len
    req_to_tokens = torch.zeros((batch_size, max_kv_len), dtype=torch.int32, device=device)

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    b_mark_shared_group = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Q: [batch_size, num_heads, head_dim]
    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device=device)

    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        kv_base = group_idx * config.max_seq_len

        for member_idx in range(group_size):
            batch_idx = group_start + member_idx
            if batch_idx >= batch_size:
                break

            # 第 member_idx 个请求可以看到前 member_idx+1 个 KV
            seq_len = member_idx + 1
            b_seq_len[batch_idx] = seq_len

            # 设置 KV 索引 - 组内请求共享相同的 KV 前缀
            for kv_pos in range(member_idx + 1):
                req_to_tokens[batch_idx, kv_pos] = kv_base + kv_pos

            # b_mark_shared_group: 组内最后一个标记为 group_size
            if member_idx == group_size - 1:
                b_mark_shared_group[batch_idx] = group_size
            else:
                b_mark_shared_group[batch_idx] = 0

    return q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group


def setup_decode_data(config: TestConfig, device="cuda", seed=42):
    """
    设置普通 Decode 测试数据（非 MTP 模式）

    每个请求独立，seq_len 符合指定分布
    """
    torch.manual_seed(seed)

    batch_size = config.batch_size
    num_heads = config.num_heads
    kv_head_num = config.kv_head_num
    head_dim = config.head_dim
    test_dtype = getattr(torch, config.dtype)

    # 生成 seq_len
    seq_lens = generate_seq_lens(config, seed)

    # KV 池大小
    total_kv_len = seq_lens.sum()
    k = torch.randn(size=(total_kv_len, kv_head_num, head_dim), dtype=test_dtype, device=device)
    v = torch.randn(size=(total_kv_len, kv_head_num, head_dim), dtype=test_dtype, device=device)

    # req_to_tokens: [batch, max_seq_len]
    max_kv_len = config.max_seq_len
    req_to_tokens = torch.zeros((batch_size, max_kv_len), dtype=torch.int32, device=device)

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    # Q: [batch_size, num_heads, head_dim]
    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device=device)

    # 设置连续的 KV 索引
    kv_offset = 0
    for i in range(batch_size):
        req_to_tokens[i, :seq_lens[i]] = torch.arange(kv_offset, kv_offset + seq_lens[i], dtype=torch.int32, device=device)
        kv_offset += seq_lens[i]

    return q, k, v, req_to_tokens, b_req_idx, b_seq_len


def benchmark_kernel_with_times(func, warmup_iters=10, test_iters=100, device="cuda") -> Tuple[float, float]:
    """
    基准测试 kernel 函数，返回多次测试的平均时间和标准差
    """
    torch.cuda.synchronize(device)

    # Warmup
    for _ in range(warmup_iters):
        _ = func()
    torch.cuda.synchronize(device)

    # 测试
    times = []
    for _ in range(test_iters):
        start = time.perf_counter()
        _ = func()
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)

    return avg_ms, std_ms


def benchmark_kernel_cuda_graph(func, graph_size=100, device="cuda") -> Tuple[float, float]:
    """
    使用 CUDA Graph 进行基准测试
    """
    torch.cuda.synchronize(device)

    # Warmup
    for _ in range(10):
        _ = func()
    torch.cuda.synchronize(device)

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = func()

    # Benchmark graph replay
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(graph_size):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / graph_size

    return avg_ms, 0.0  # CUDA Graph 模式下标准差为 0


# ============================================================================
# Kernel 测试函数
# ============================================================================

def test_mtp_diverse_original(q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group, block_seq=256):
    """MTP Diverse Original"""
    return token_decode_attention_mtp_diverse_single_token(
        q=q, k=k, v=v,
        Req_to_tokens=req_to_tokens,
        B_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        block_seq=block_seq,
    )


def test_mtp_diverse_vsm(q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group):
    """MTP Diverse VSM"""
    return token_decode_attention_mtp_diverse_vsm_single_token(
        q=q, k=k, v=v,
        Req_to_tokens=req_to_tokens,
        B_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
    )


def test_flash_decoding_gqa(q, k, v, infer_state):
    """Flash Decoding GQA"""
    return gqa_token_decode_attention_flash_decoding(
        q.clone(), infer_state, k.clone(), v.clone(), out=torch.empty_like(q)
    )


def test_flash_decoding_vsm(q, k, v, infer_state):
    """Flash Decoding VSM"""
    return gqa_token_decode_attention_flash_decoding_vsm(
        q.clone(), k.clone(), v.clone(), infer_state, out=torch.empty_like(q)
    )


# ============================================================================
# Benchmark 函数
# ============================================================================

def benchmark_mtp_kernels(config: TestConfig, device="cuda", use_cuda_graph=False) -> List[BenchmarkResult]:
    """
    测试 MTP kernels 性能
    """
    results = []

    # 准备数据
    q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group = setup_mtp_diverse_data(config, device)

    block_seq = 256

    # 定义 kernels
    kernels = [
        ("MTP_Diverse_Original", lambda: test_mtp_diverse_original(
            q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group, block_seq
        )),
        ("MTP_Diverse_VSM", lambda: test_mtp_diverse_vsm(
            q, k, v, req_to_tokens, b_req_idx, b_seq_len, b_mark_shared_group
        )),
    ]

    for kernel_name, kernel_func in kernels:
        try:
            if use_cuda_graph:
                avg_ms, std_ms = benchmark_kernel_cuda_graph(kernel_func, device=device)
            else:
                avg_ms, std_ms = benchmark_kernel_with_times(kernel_func, device=device)

            # 计算吞吐量（实际计算的 token 数，只有组内最后一个请求被计算）
            actual_tokens = config.num_groups
            tokens_per_sec = actual_tokens / (avg_ms / 1000)

            results.append(BenchmarkResult(
                config_name=config.name,
                kernel_name=kernel_name,
                batch_size=config.batch_size,
                avg_seq_len=b_seq_len.float().mean().item(),
                group_size=config.group_size,
                avg_time_ms=avg_ms,
                std_time_ms=std_ms,
                tokens_per_sec=tokens_per_sec,
                status="SUCCESS",
                cuda_graph_enabled=use_cuda_graph,
            ))
        except Exception as e:
            results.append(BenchmarkResult(
                config_name=config.name,
                kernel_name=kernel_name,
                batch_size=config.batch_size,
                avg_seq_len=config.avg_seq_len,
                group_size=config.group_size,
                avg_time_ms=0,
                std_time_ms=0,
                tokens_per_sec=0,
                status="FAILED",
                error_msg=str(e),
                cuda_graph_enabled=use_cuda_graph,
            ))

    return results


def benchmark_decode_kernels(config: TestConfig, device="cuda", use_cuda_graph=False) -> List[BenchmarkResult]:
    """
    测试普通 Decode kernels 性能（非 MTP 模式）

    测试 Flash Decoding 和 Flash Decoding VSM 的性能
    """
    results = []

    # 准备数据
    q, k, v, req_to_tokens, b_req_idx, b_seq_len = setup_decode_data(config, device)

    # 创建 infer_state
    infer_state = SimpleInferState(b_seq_len, req_to_tokens, b_req_idx)

    # 定义 kernels
    kernels = [
        ("FlashDecoding_GQA", lambda: test_flash_decoding_gqa(q, k, v, infer_state)),
        ("FlashDecoding_VSM", lambda: test_flash_decoding_vsm(q, k, v, infer_state)),
    ]

    for kernel_name, kernel_func in kernels:
        try:
            if use_cuda_graph:
                avg_ms, std_ms = benchmark_kernel_cuda_graph(kernel_func, device=device)
            else:
                avg_ms, std_ms = benchmark_kernel_with_times(kernel_func, device=device)

            # 计算吞吐量
            tokens_per_sec = config.batch_size / (avg_ms / 1000)

            results.append(BenchmarkResult(
                config_name=config.name,
                kernel_name=kernel_name,
                batch_size=config.batch_size,
                avg_seq_len=b_seq_len.float().mean().item(),
                group_size=1,
                avg_time_ms=avg_ms,
                std_time_ms=std_ms,
                tokens_per_sec=tokens_per_sec,
                status="SUCCESS",
                cuda_graph_enabled=use_cuda_graph,
            ))
        except Exception as e:
            results.append(BenchmarkResult(
                config_name=config.name,
                kernel_name=kernel_name,
                batch_size=config.batch_size,
                avg_seq_len=config.avg_seq_len,
                group_size=1,
                avg_time_ms=0,
                std_time_ms=0,
                tokens_per_sec=0,
                status="FAILED",
                error_msg=str(e),
                cuda_graph_enabled=use_cuda_graph,
            ))

    return results


# ============================================================================
# 配置生成函数
# ============================================================================

def generate_llama3_configs() -> List[TestConfig]:
    """
    生成 Llama-3 测试配置
    avg_input_len=1024, variance=256, max_len=2048
    """
    configs = []

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for bs in batch_sizes:
        configs.append(TestConfig(
            name=f"Llama3_bs{bs}",
            batch_size=bs,
            avg_seq_len=1024,
            seq_len_variance=256,
            max_seq_len=2048,
        ))

    return configs


def generate_deepseek_v2_configs() -> List[TestConfig]:
    """
    生成 DeepSeek-V2 测试配置
    avg_input_len=10240, variance=2560, max_len=128000
    """
    configs = []

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for bs in batch_sizes:
        configs.append(TestConfig(
            name=f"DeepSeekV2_bs{bs}",
            batch_size=bs,
            avg_seq_len=10240,
            seq_len_variance=2560,
            max_seq_len=128000,
        ))

    return configs


def generate_load_distribution_configs(model_type: str = "llama3") -> List[TestConfig]:
    """
    生成不同负载分布的测试配置

    测试不同负载分布对 kernel 性能的影响：
    - normal: 正态分布
    - uniform: 均匀分布
    - outlier: 存在异常值（大部分短请求 + 少量长请求）
    """
    configs = []

    if model_type == "llama3":
        avg_seq_len = 1024
        seq_len_variance = 256
        max_seq_len = 2048
    else:  # deepseek_v2
        avg_seq_len = 10240
        seq_len_variance = 2560
        max_seq_len = 128000

    batch_sizes = [8, 32, 128]
    distributions = ["normal", "uniform", "outlier"]

    for bs in batch_sizes:
        for dist in distributions:
            configs.append(TestConfig(
                name=f"{model_type}_bs{bs}_{dist}",
                batch_size=bs,
                avg_seq_len=avg_seq_len,
                seq_len_variance=seq_len_variance,
                max_seq_len=max_seq_len,
                load_distribution=dist,
            ))

    return configs


def generate_mtp_configs() -> List[TestConfig]:
    """
    生成 MTP 特定测试配置
    测试不同 group_size 的影响
    """
    configs = []

    group_sizes = [2, 4, 6, 8, 12]
    batch_sizes = [8, 16, 32, 64, 128]

    for gs in group_sizes:
        for bs in batch_sizes:
            if bs % gs == 0:
                num_groups = bs // gs
                configs.append(TestConfig(
                    name=f"MTP_gs{gs}_bs{bs}",
                    batch_size=bs,
                    avg_seq_len=gs // 2 + 1,
                    seq_len_variance=0,
                    max_seq_len=gs,
                    group_size=gs,
                    num_groups=num_groups,
                ))

    return configs


# ============================================================================
# 结果输出函数
# ============================================================================

def print_results_table(results: List[BenchmarkResult]):
    """打印结果表格"""
    print("\n" + "=" * 150)
    print("MTP Kernel Performance Comparison Results")
    print("=" * 150)
    print(f"{'Config':<28} {'Kernel':<25} {'Batch':<8} {'Avg_Seq':<10} {'Time(ms)':<12} {'Std(ms)':<10} {'Tokens/s':<12} {'Status':<10}")
    print("-" * 150)

    for r in results:
        status_str = r.status[:10] if len(r.status) > 10 else r.status
        if r.status == "SUCCESS":
            print(f"{r.config_name:<28} {r.kernel_name:<25} {r.batch_size:<8} {r.avg_seq_len:<10.1f} "
                  f"{r.avg_time_ms:<12.3f} {r.std_time_ms:<10.3f} {r.tokens_per_sec:<12.2f} {status_str:<10}")
        else:
            print(f"{r.config_name:<28} {r.kernel_name:<25} {r.batch_size:<8} {r.avg_seq_len:<10} "
                  f"{'N/A':<12} {'N/A':<10} {'N/A':<12} {status_str:<10}")

    print("=" * 150)


def analyze_results(results: List[BenchmarkResult]) -> Dict:
    """分析结果"""
    analysis = {}

    # 按 config 分组
    by_config = defaultdict(list)
    for r in results:
        if r.status == "SUCCESS":
            by_config[r.config_name].append(r)

    # 计算每个 config 下各 kernel 的相对性能
    for config_name, kernel_results in by_config.items():
        if len(kernel_results) < 2:
            continue

        # 找到最快的 kernel 作为基准
        fastest = min(kernel_results, key=lambda x: x.avg_time_ms)

        analysis[config_name] = {
            "fastest_kernel": fastest.kernel_name,
            "fastest_time": fastest.avg_time_ms,
            "speedups": {r.kernel_name: fastest.avg_time_ms / r.avg_time_ms for r in kernel_results}
        }

    # 分析 CUDA Graph 的影响
    cuda_graph_comparison = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.status == "SUCCESS":
            key = (r.config_name, r.kernel_name)
            cuda_graph_comparison[key][r.cuda_graph_enabled].append(r.avg_time_ms)

    cuda_graph_impact = {}
    for key, data in cuda_graph_comparison.items():
        config_name, kernel_name = key
        if False in data and True in data:
            normal_avg = np.mean(data[False])
            graph_avg = np.mean(data[True])
            speedup = normal_avg / graph_avg if graph_avg > 0 else 0
            cuda_graph_impact[f"{config_name}_{kernel_name}"] = {
                "normal_time": normal_avg,
                "cuda_graph_time": graph_avg,
                "speedup": speedup,
            }

    analysis["cuda_graph_impact"] = cuda_graph_impact

    return analysis


def export_results(results: List[BenchmarkResult], analysis: Dict, filename: str = "mtp_kernel_benchmark_results.json"):
    """导出结果到 JSON"""
    export_data = {
        "results": [asdict(r) for r in results],
        "analysis": analysis,
    }
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"\nResults exported to {filename}")


def print_analysis_summary(analysis: Dict):
    """打印分析摘要"""
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)

    # Kernel 性能对比
    print("\n--- Kernel Speedup Summary ---")
    for config_name, info in analysis.items():
        if config_name == "cuda_graph_impact":
            continue
        print(f"\n{config_name}:")
        print(f"  Fastest: {info['fastest_kernel']} ({info['fastest_time']:.3f} ms)")
        for kernel, speedup in sorted(info['speedups'].items(), key=lambda x: -x[1]):
            print(f"  {kernel}: {speedup:.2f}x")

    # CUDA Graph 影响
    if "cuda_graph_impact" in analysis:
        print("\n--- CUDA Graph Impact Summary ---")
        cuda_data = analysis["cuda_graph_impact"]
        if cuda_data:
            speedups = [v["speedup"] for v in cuda_data.values()]
            print(f"Average CUDA Graph speedup: {np.mean(speedups):.2f}x")
            print(f"Max CUDA Graph speedup: {max(speedups):.2f}x")
            print(f"Min CUDA Graph speedup: {min(speedups):.2f}x")
        else:
            print("No CUDA Graph comparison data available")


def main():
    parser = argparse.ArgumentParser(description="MTP Kernel Performance Benchmark")
    parser.add_argument("--model", type=str, default="all",
                        choices=["llama3", "deepseek_v2", "load_dist", "mtp", "all"],
                        help="Model type to benchmark")
    parser.add_argument("--cuda-graph", action="store_true", help="Enable CUDA Graph")
    parser.add_argument("--output", type=str, default="mtp_kernel_benchmark_results.json",
                        help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="CUDA device")
    parser.add_argument("--skip-decode", action="store_true", help="Skip decode kernel benchmarks")
    parser.add_argument("--skip-mtp", action="store_true", help="Skip MTP kernel benchmarks")
    args = parser.parse_args()

    print("=" * 80)
    print("MTP Kernel Performance Comparison Benchmark")
    print("=" * 80)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    device = args.device
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Graph: {args.cuda_graph}")

    # 生成配置
    all_configs = []
    if args.model in ["llama3", "all"]:
        print("\nGenerating Llama-3 configs...")
        all_configs.extend(generate_llama3_configs())

    if args.model in ["deepseek_v2", "all"]:
        print("Generating DeepSeek-V2 configs...")
        all_configs.extend(generate_deepseek_v2_configs())

    if args.model in ["load_dist", "all"]:
        print("Generating load distribution configs...")
        all_configs.extend(generate_load_distribution_configs("llama3"))
        all_configs.extend(generate_load_distribution_configs("deepseek_v2"))

    if args.model in ["mtp", "all"]:
        print("Generating MTP configs...")
        all_configs.extend(generate_mtp_configs())

    print(f"\nTotal configs: {len(all_configs)}")

    # Warmup GPU
    print("\nWarming up GPU...")
    warmup_gpu(device)

    # 运行测试
    all_results = []

    print("\n" + "=" * 80)
    print("Running benchmarks...")
    print("=" * 80)

    for i, config in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] {config.name}: bs={config.batch_size}, "
              f"avg_seq={config.avg_seq_len}, max_seq={config.max_seq_len}, "
              f"distribution={config.load_distribution}, group_size={config.group_size}")

        if config.group_size > 1 and not args.skip_mtp:
            # MTP 模式
            results = benchmark_mtp_kernels(config, device, args.cuda_graph)
            all_results.extend(results)

        if config.group_size == 1 and not args.skip_decode:
            # 普通 Decode 模式
            results = benchmark_decode_kernels(config, device, args.cuda_graph)
            all_results.extend(results)

        # 清理显存
        torch.cuda.empty_cache()

    # 打印结果
    print_results_table(all_results)

    # 分析结果
    analysis = analyze_results(all_results)

    # 打印分析摘要
    print_analysis_summary(analysis)

    # 导出结果
    export_results(all_results, analysis, args.output)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
