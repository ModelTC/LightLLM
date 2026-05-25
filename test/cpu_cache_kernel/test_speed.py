"""
Speed benchmark for copy_cpu_cache_to_kv_buffer in linear_att_cpu_cache_copy.py.

Test configuration (matching the user's LinearAttCacheConfig):
    tp_world_size=8, full_att_all_num_kv_heads=2, full_att_dtype=torch.bfloat16,
    full_att_num_kv_heads=1, full_att_head_dim=256,
    num_linear_k_heads=2, num_linear_v_heads=8,
    head_linear_k_dim=128, head_linear_v_dim=128,
    conv_kernel_size=4, linear_layer_num=36,
    conv_state_dtype=torch.bfloat16, ssm_state_dtype=torch.float32,
    full_attention_interval=4, all_layer_num=48
"""

import os
import json
import time
import triton
import torch
from easydict import EasyDict

# ---------------------------------------------------------------------------
# Step 0 – set up environment args BEFORE any import that calls
#          get_env_start_args() / LinearAttCacheConfig.load_from_args().
# ---------------------------------------------------------------------------
_env_args = {
    "cpu_cache_token_page_size": 2048 * 8,  # big_page_token_num
    "linear_att_hash_page_size": 2048,
    "linear_att_page_block_num": 8,  # 512 * 1 == 512
    "data_type": "bfloat16",
    "linear_att_ssm_data_type": "float32",
    "model_dir": "/tmp/fake_model",  # dummy – not used when config is built directly
    "tp": 8,
    "dp": 1,
    "running_max_req_size": 2048,
    "enable_cpu_cache": True,
}
os.environ["LIGHTLLM_START_ARGS"] = json.dumps(_env_args)

# ---------------------------------------------------------------------------
# Step 1 – build LinearAttCacheConfig directly (avoids needing a real model dir)
# ---------------------------------------------------------------------------
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

linear_config = LinearAttCacheConfig(
    tp_world_size=8,
    full_att_all_num_kv_heads=2,
    full_att_dtype=torch.bfloat16,
    full_att_num_kv_heads=1,
    full_att_head_dim=256,
    num_linear_k_heads=2,
    num_linear_v_heads=8,
    head_linear_k_dim=128,
    head_linear_v_dim=128,
    conv_kernel_size=4,
    linear_layer_num=36,
    conv_state_dtype=torch.bfloat16,
    ssm_state_dtype=torch.float32,
    full_attention_interval=4,
    all_layer_num=48,
)
print(f"LinearAttCacheConfig:\n{linear_config}\n", flush=True)

# ---------------------------------------------------------------------------
# Step 2 – derive sizes from the config
# ---------------------------------------------------------------------------
big_page_token_num = _env_args["cpu_cache_token_page_size"]  # 512
full_att_layer_num = linear_config.all_layer_num // linear_config.full_attention_interval  # 12

full_att_bytes = linear_config.get_cpu_cache_full_att_bytes()  # per big page
conv_bytes = linear_config.get_cpu_cache_conv_bytes()
ssm_bytes = linear_config.get_cpu_cache_ssm_bytes()
total_bytes = full_att_bytes + conv_bytes + ssm_bytes
print(
    f"Per-page bytes  full_att={full_att_bytes:,}  conv={conv_bytes:,}  " f"ssm={ssm_bytes:,}  total={total_bytes:,}",
    flush=True,
)
total_bytes = linear_config.get_cpu_cache_big_page_bytes()

# ---------------------------------------------------------------------------
# Step 3 – allocate tensors
# ---------------------------------------------------------------------------
grid_num = 8
PAGE_NUM = 1  # number of big pages to copy per call
SEQ_LEN = 2048 * 8  # total sequence length in gpu_full_att_kv_state dim-1
BIG_PAGE_COUNT = PAGE_NUM  # big_page_buffer_ids length == page_indexes length

# --- GPU tensors ---
mem_indexes = torch.arange(0, big_page_token_num * PAGE_NUM, dtype=torch.int64, device="cpu")
big_page_buffer_ids = torch.arange(0, BIG_PAGE_COUNT, dtype=torch.int64, device="cpu")
page_indexes = torch.arange(0, PAGE_NUM, dtype=torch.int32, device="cpu")

gpu_full_att_kv_state = torch.empty(
    (
        full_att_layer_num,
        SEQ_LEN,
        2 * max(1, linear_config.full_att_num_kv_heads // linear_config.tp_world_size),
        linear_config.full_att_head_dim,
    ),
    dtype=linear_config.full_att_dtype,
    device="cuda",
)

# --- CPU tensors ---
buffer_count = triton.cdiv(SEQ_LEN, big_page_token_num) + 2  # matches Qwen3NextMemManager


conv_shape = linear_config.get_conv_state_shape()
cpu_kv_conv_state = torch.empty(
    (buffer_count, linear_config.linear_layer_num, *conv_shape),
    dtype=linear_config.conv_state_dtype,
    device="cpu",
    pin_memory=True,
)

ssm_shape = linear_config.get_ssm_state_shape()  # (num_linear_v_heads, head_linear_k_dim, head_linear_v_dim)
cpu_kv_ssm_state = torch.empty(
    (buffer_count, linear_config.linear_layer_num, *ssm_shape),
    dtype=linear_config.ssm_state_dtype,
    device="cpu",
    pin_memory=True,
)

# conv_shape = linear_config.get_conv_state_shape()
# cpu_kv_conv_state = torch.empty(
#     (buffer_count, linear_config.linear_layer_num, *conv_shape),
#     dtype=linear_config.conv_state_dtype, device="cuda",
# )

# ssm_shape = linear_config.get_ssm_state_shape()  # (num_linear_v_heads, head_linear_k_dim, head_linear_v_dim)
# cpu_kv_ssm_state = torch.empty(
#     (buffer_count, linear_config.linear_layer_num, *ssm_shape),
#     dtype=linear_config.ssm_state_dtype, device="cuda",
# )


# cpu_cache_tensor: [page_num, 1, 1, 1, total_bytes]
cpu_cache_tensor = torch.empty(
    (PAGE_NUM, 1, 1, 1, total_bytes),
    dtype=torch.uint8,
    device="cpu",
    pin_memory=True,
)

# Move GPU tensors to CUDA
mem_indexes_cuda = mem_indexes.cuda(non_blocking=True)
big_page_buffer_ids_cuda = big_page_buffer_ids.cuda(non_blocking=True)
page_indexes_cuda = page_indexes.cuda(non_blocking=True)
gpu_full_att_kv_state = gpu_full_att_kv_state.cuda(non_blocking=True)

torch.cuda.synchronize()
print("All tensors allocated and moved to GPU.\n", flush=True)

# ---------------------------------------------------------------------------
# Step 4 – import and warm-up the triton kernel
# ---------------------------------------------------------------------------
from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
    copy_cpu_cache_to_kv_buffer,
)

print("Warming up …", flush=True)
copy_cpu_cache_to_kv_buffer(
    mem_indexes=mem_indexes_cuda,
    big_page_buffer_ids=big_page_buffer_ids_cuda,
    page_indexes=page_indexes_cuda,
    gpu_full_att_kv_state=gpu_full_att_kv_state,
    cpu_kv_conv_state=cpu_kv_conv_state,
    cpu_kv_ssm_state=cpu_kv_ssm_state,
    cpu_cache_tensor=cpu_cache_tensor,
    tp_rank=0,
    tp_world_size=linear_config.tp_world_size,
    big_page_token_num=big_page_token_num,
    linear_config=linear_config,
    grid_num=grid_num,
)
torch.cuda.synchronize()
print("Warm-up done.\n", flush=True)

# ---------------------------------------------------------------------------
# Step 5 – benchmark
# ---------------------------------------------------------------------------
WARMUP_ITERS = 10
BENCH_ITERS = 100

print(f"Benchmarking  ({BENCH_ITERS} iterations, {PAGE_NUM} pages / {big_page_token_num} tokens each) …", flush=True)

# Warm-up
for _ in range(WARMUP_ITERS):
    copy_cpu_cache_to_kv_buffer(
        mem_indexes=mem_indexes_cuda,
        big_page_buffer_ids=big_page_buffer_ids_cuda,
        page_indexes=page_indexes_cuda,
        gpu_full_att_kv_state=gpu_full_att_kv_state,
        cpu_kv_conv_state=cpu_kv_conv_state,
        cpu_kv_ssm_state=cpu_kv_ssm_state,
        cpu_cache_tensor=cpu_cache_tensor,
        tp_rank=0,
        tp_world_size=linear_config.tp_world_size,
        big_page_token_num=big_page_token_num,
        linear_config=linear_config,
        grid_num=grid_num,
    )
torch.cuda.synchronize()

# Timed runs
times = []
for _ in range(BENCH_ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    copy_cpu_cache_to_kv_buffer(
        mem_indexes=mem_indexes_cuda,
        big_page_buffer_ids=big_page_buffer_ids_cuda,
        page_indexes=page_indexes_cuda,
        gpu_full_att_kv_state=gpu_full_att_kv_state,
        cpu_kv_conv_state=cpu_kv_conv_state,
        cpu_kv_ssm_state=cpu_kv_ssm_state,
        cpu_cache_tensor=cpu_cache_tensor,
        tp_rank=0,
        tp_world_size=linear_config.tp_world_size,
        big_page_token_num=big_page_token_num,
        linear_config=linear_config,
        grid_num=grid_num,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

# ---------------------------------------------------------------------------
# Step 6 – report
# ---------------------------------------------------------------------------
import statistics

times_ms = [t * 1e3 for t in times]
total_tokens = PAGE_NUM * big_page_token_num

# Calculate head_scale_size (same logic as in copy_cpu_cache_to_kv_buffer)
if linear_config.full_att_all_num_kv_heads % linear_config.tp_world_size == 0:
    head_scale_size = 1
else:
    head_scale_size = linear_config.tp_world_size // linear_config.full_att_all_num_kv_heads

# Each TP rank copies:
# - full_att_bytes / head_scale_size (full attention is sharded by head_scale_size)
# - conv_bytes / tp_world_size (conv state is sharded by tp_rank)
# - ssm_bytes / tp_world_size (ssm state is sharded by tp_rank)
full_att_bytes = linear_config.get_cpu_cache_full_att_bytes()
conv_bytes = linear_config.get_cpu_cache_conv_bytes()
ssm_bytes = linear_config.get_cpu_cache_ssm_bytes()

bytes_per_page_per_tp = (
    full_att_bytes
    * max(1, linear_config.full_att_all_num_kv_heads // linear_config.tp_world_size)
    / linear_config.full_att_all_num_kv_heads
    + conv_bytes // linear_config.tp_world_size
    + ssm_bytes // linear_config.tp_world_size
)
total_bytes_copied = PAGE_NUM * bytes_per_page_per_tp

print()
print("=" * 60)
print(f"  copy_cpu_cache_to_kv_buffer  speed benchmark")
print("=" * 60)
print(f"  Pages / call              : {PAGE_NUM}")
print(f"  Tokens / page             : {big_page_token_num}")
print(f"  Total tokens / call       : {total_tokens}")
print(f"  Bytes / page (total)      : {total_bytes:,}")
print(f"  Bytes / page (per TP)     : {bytes_per_page_per_tp:,}")
print(f"  Total bytes / call        : {total_bytes_copied:,}  ({total_bytes_copied / 1024**3:.3f} GB)")
print(f"  Iterations                : {BENCH_ITERS}")
print(f"  Mean latency              : {statistics.mean(times_ms):.3f} ms")
print(f"  Median latency            : {statistics.median(times_ms):.3f} ms")
print(f"  Std  latency              : {statistics.stdev(times_ms):.3f} ms")
print(f"  Min  latency              : {min(times_ms):.3f} ms")
print(f"  Max  latency              : {max(times_ms):.3f} ms")
print(f"  Throughput (tokens/s)     : {total_tokens / statistics.mean(times_ms) * 1e3:,.0f}")
print(f"  Throughput (GB/s)         : {total_bytes_copied / 1024**3 / statistics.mean(times_ms) * 1e3:.3f}")
print("=" * 60)
