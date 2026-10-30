"""Measure the public CPU-page restore path for one TP shard of a hybrid model.

Run from the repository root, for example:
    PYTHONPATH=. python test/cpu_cache_kernel/test_speed.py --tokens-per-page 16384 --iterations 100
The reported time includes validation, KV transfers, and CPU state-page repacking.
"""

import argparse
import statistics
import time

import torch

from lightllm.common.att_resources.allocators import FullAttKVAllocator
from lightllm.common.att_resources.full_att import (
    GQAFullAttentionState,
    FullAttentionConfig,
)
from lightllm.models.qwen3next.config import Qwen3NextResourceConfig
from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import StatePageAllocator
from lightllm.common.att_resources.bounded_att import create_gdn_state, ConvSSMStateConfig
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import set_env_start_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-per-page", type=int, default=16384)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if min(args.tokens_per_page, args.pages, args.iterations) <= 0 or args.warmup < 0:
        parser.error("page sizes and iterations must be positive; warmup must be nonnegative")
    set_env_start_args(StartArgs(cpu_cache_token_page_size=args.tokens_per_page))
    cfg = Qwen3NextResourceConfig(
        tp_world_size=8,
        full_att_all_num_kv_heads=2,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=256,
        global_linear_k_heads=16,
        global_linear_v_heads=64,
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
    tokens = args.tokens_per_page * args.pages
    full = GQAFullAttentionState(
        FullAttentionConfig(
            logical_layer_groups=tuple((layer,) for layer in tuple(range(12))),
            global_head_num=1,
            head_dim=256,
            dtype=torch.bfloat16,
            tp_rank=0,
            tp_world_size=8,
        ),
    )
    full.initialize_storage(tokens)
    resources = AttResourceManager(request_capacity=1)
    bounded_state = create_gdn_state(
        config=ConvSSMStateConfig(
            logical_layer_ids=tuple(layer for layer in range(cfg.all_layer_num) if (layer + 1) % cfg.full_attention_interval),
            global_k_heads=cfg.global_linear_k_heads,
            global_v_heads=cfg.global_linear_v_heads,
            key_dim=cfg.head_linear_k_dim,
            value_dim=cfg.head_linear_v_dim,
            conv_width=cfg.conv_kernel_size - 1,
            conv_dtype=cfg.conv_state_dtype,
            ssm_dtype=cfg.ssm_state_dtype,
            tp_rank=0,
            tp_world_size=cfg.tp_world_size,
        ),
        request_capacity=1,
        mtp_step=0,
        device="cuda",
    )
    bounded_state.initialize_page_pools(big_page_capacity=args.pages + 2, small_page_capacity=0)
    resources.set_att_state("main", full)
    resources.set_att_state("main", bounded_state)
    resources.full_att_allocator = FullAttKVAllocator(tokens)
    resources.bounded_att_big_page_allocator = StatePageAllocator(args.pages + 2, reserved_slots=2)
    resources.bounded_att_cpu_load_big_page_slot = args.pages
    resources.bounded_att_cpu_offload_big_page_slot = args.pages + 1
    resources.finish_state_creation()
    pages = list(range(args.pages))
    slots = resources.bounded_att_big_page_allocator.alloc(args.pages)
    cpu_page_buffers = torch.zeros((args.pages, resources.get_cpu_page_bytes()), dtype=torch.uint8, pin_memory=True)
    # Populate every TP region through the production API before measuring restores.
    full.k_storage.zero_()
    full.v_storage.zero_()
    resources.save_bounded_att_pages([0] * args.pages, page_kind="big", slot_ids=slots)
    torch.cuda.synchronize()
    token_indices = torch.arange(tokens, dtype=torch.int64, device="cuda")
    resources.write_req_to_cpu(
        token_indices=token_indices,
        cpu_page_buffers=cpu_page_buffers,
        cpu_page_indices=pages,
        layouts=resources.get_cpu_transfer_layouts(),
        big_page_slot_ids=slots,
    )
    torch.cuda.synchronize()

    def restore():
        resources.read_req_from_cpu(
            cpu_page_buffers=cpu_page_buffers,
            cpu_page_indices=pages,
            token_indices=token_indices,
            layouts=resources.get_cpu_transfer_layouts(),
            big_page_slot_ids=slots,
            req_idx=0,
        )

    for _ in range(args.warmup):
        restore()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        restore()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    token_bytes = (
        full.k_storage[:, :tokens].numel() * full.k_storage.element_size()
        + full.v_storage[:, :tokens].numel() * full.v_storage.element_size()
    )
    pool = (
        resources.main_kda_conv_bf16_ssm_fp32_state if hasattr(resources, "main_kda_conv_bf16_ssm_fp32_state") else resources.main_gdn_conv_bf16_ssm_fp32_state
    ).big_pages_pool
    state_bytes = args.pages * sum(t[0].numel() * t.element_size() for t in (pool.conv_state, pool.ssm_state))
    median = statistics.median(times)
    print(f"Pages: {args.pages}; tokens/page: {args.tokens_per_page}; CPU bytes/page: {resources.get_cpu_page_bytes():,}")
    print(f"Mean: {statistics.mean(times) * 1000:.3f} ms; median: {median * 1000:.3f} ms")
    print(f"KV + state payload throughput: {(token_bytes + state_bytes) / median / 1024**3:.3f} GiB/s")


if __name__ == "__main__":
    main()
