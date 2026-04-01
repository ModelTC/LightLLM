import argparse
import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_diverse import (
    gqa_token_decode_attention_flash_decoding_diverse,
)
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_diverse_stage1 import (
    GQADiverseDecodeStage1KernelConfig,
)
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_diverse_stage2 import (
    GQADiverseDecodeStage2KernelConfig,
)

KERNEL_TUNING_DIR = str((Path(__file__).resolve().parents[1] / "../kernel").resolve())
if KERNEL_TUNING_DIR not in sys.path:
    sys.path.insert(0, KERNEL_TUNING_DIR)

tune_stage1_one_shape = importlib.import_module("llama_gqa_diverse_decode_stage1_fp_tuning").tune_one_shape
tune_stage2_one_shape = importlib.import_module("llama_gqa_diverse_decode_stage2_fp_tuning").tune_one_shape


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MockReqManager:
    def __init__(self, req_to_token_indexs: torch.Tensor):
        self.req_to_token_indexs = req_to_token_indexs


@dataclass
class MockInferState:
    batch_size: int
    max_kv_seq_len: int
    req_manager: MockReqManager
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_shared_seq_len: torch.Tensor
    b_mark_shared_group: torch.Tensor


def build_case(
    batch_size: int,
    shared_prefix_len: int,
    total_seq_len: int,
    max_len_in_batch: int,
    max_batch_group_size: int,
    num_heads: int,
    kv_head_num: int,
    head_dim: int,
    dtype: torch.dtype,
):
    tail_len = total_seq_len - shared_prefix_len
    if tail_len < 0:
        raise ValueError("total_seq_len must be >= shared_prefix_len")
    tails = [tail_len for _ in range(batch_size)]
    seq_lens = [shared_prefix_len + tail for tail in tails]
    max_kv_seq_len = max_len_in_batch
    total_unique_tokens = shared_prefix_len + sum(tails)

    q = torch.randn((batch_size, num_heads, head_dim), dtype=dtype, device="cuda")
    cache_k = torch.randn((total_unique_tokens, kv_head_num, head_dim), dtype=dtype, device="cuda")
    cache_v = torch.randn((total_unique_tokens, kv_head_num, head_dim), dtype=dtype, device="cuda")

    req_to_tokens = torch.zeros((batch_size, max_len_in_batch), dtype=torch.int32, device="cuda")
    shared_positions = torch.arange(shared_prefix_len, dtype=torch.int32, device="cuda")
    tail_offset = shared_prefix_len
    for batch_idx, tail_len in enumerate(tails):
        req_to_tokens[batch_idx, :shared_prefix_len] = shared_positions
        req_to_tokens[batch_idx, shared_prefix_len : shared_prefix_len + tail_len] = torch.arange(
            tail_offset, tail_offset + tail_len, dtype=torch.int32, device="cuda"
        )
        tail_offset += tail_len

    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_shared_seq_len = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    group_start = 0
    while group_start < batch_size:
        group_size = min(max_batch_group_size, batch_size - group_start)
        group_end = group_start + group_size
        if group_size > 1:
            b_shared_seq_len[group_start:group_end] = shared_prefix_len
        b_mark_shared_group[group_end - 1] = group_size
        group_start = group_end

    infer_state = MockInferState(
        batch_size=batch_size,
        max_kv_seq_len=max_kv_seq_len,
        req_manager=MockReqManager(req_to_tokens),
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_shared_seq_len=b_shared_seq_len,
        b_mark_shared_group=b_mark_shared_group,
    )

    kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(b_seq_len, dim=0)
    kv_indices = torch.empty(int(b_seq_len.sum().item()), dtype=torch.int32, device="cuda")
    write_offset = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        kv_indices[write_offset : write_offset + seq_len] = req_to_tokens[batch_idx, :seq_len]
        write_offset += seq_len
    kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

    return {
        "q": q,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "infer_state": infer_state,
        "tails": tails,
        "seq_lens": seq_lens,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
    }


def build_flashinfer_wrapper(case, num_heads: int, kv_head_num: int, head_dim: int):
    import flashinfer

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=case["kv_indptr"],
        paged_kv_indices_buffer=case["kv_indices"],
        paged_kv_last_page_len_buffer=case["kv_last_page_len"],
    )
    wrapper.plan(
        case["kv_indptr"],
        case["kv_indices"],
        case["kv_last_page_len"],
        num_heads,
        kv_head_num,
        head_dim,
        1,
        q_data_type=case["q"].dtype,
        kv_data_type=case["cache_k"].dtype,
        non_blocking=True,
    )
    return wrapper


def maybe_tune_configs(
    *,
    tune: bool,
    seq_len: int,
    shared_seq_len: int,
    max_len_in_batch: int,
    batch_size: int,
    num_heads: int,
    kv_head_num: int,
    head_dim: int,
    block_seq: int,
    max_batch_group_size: int,
    dtype: torch.dtype,
):
    gqa_group_size = num_heads // kv_head_num
    stage1_config = None
    stage2_config = None

    if tune:
        stage1_config = tune_stage1_one_shape(
            block_seq=block_seq,
            batch_size=batch_size,
            seq_len=seq_len,
            shared_seq_len=shared_seq_len,
            max_len_in_batch=max_len_in_batch,
            gqa_group_size=gqa_group_size,
            q_head_dim=head_dim,
            max_batch_group_size=max_batch_group_size,
            dtype=dtype,
            test_count=1,
        )
        stage2_config = tune_stage2_one_shape(
            block_seq=block_seq,
            batch_size=batch_size,
            seq_len=seq_len,
            shared_seq_len=shared_seq_len,
            max_len_in_batch=max_len_in_batch,
            num_heads=num_heads,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            dtype=dtype,
            test_count=1,
        )
        return stage1_config, stage2_config

    GQADiverseDecodeStage1KernelConfig.get_the_config.cache_clear()
    GQADiverseDecodeStage1KernelConfig.try_to_get_best_config.cache_clear()
    GQADiverseDecodeStage2KernelConfig.get_the_config.cache_clear()
    GQADiverseDecodeStage2KernelConfig.try_to_get_best_config.cache_clear()

    stage1_config = GQADiverseDecodeStage1KernelConfig.try_to_get_best_config(
        batch_size=batch_size,
        avg_seq_len_in_batch=seq_len,
        gqa_group_size=gqa_group_size,
        max_batch_group_size=max_batch_group_size,
        q_head_dim=head_dim,
        block_seq=block_seq,
        out_dtype=dtype,
    )
    stage2_config = GQADiverseDecodeStage2KernelConfig.try_to_get_best_config(
        batch_size=batch_size,
        avg_seq_len_in_batch=seq_len,
        gqa_group_size=gqa_group_size,
        q_head_dim=head_dim,
        block_seq=block_seq,
        out_dtype=dtype,
    )
    return stage1_config, stage2_config


def benchmark_one_case(
    case,
    num_heads: int,
    kv_head_num: int,
    head_dim: int,
    rep: int,
    max_batch_group_size: int,
    stage1_run_config: dict,
    stage2_run_config: dict,
):
    wrapper = build_flashinfer_wrapper(case, num_heads=num_heads, kv_head_num=kv_head_num, head_dim=head_dim)
    diverse_out = torch.empty_like(case["q"])
    flashinfer_out = torch.empty_like(case["q"])
    shared_streams_dict = {}

    def run_diverse():
        gqa_token_decode_attention_flash_decoding_diverse(
            q=case["q"],
            infer_state=case["infer_state"],
            cache_k=case["cache_k"],
            cache_v=case["cache_v"],
            out=diverse_out,
            shared_streams_dict=shared_streams_dict,
            max_batch_group_size=max_batch_group_size,
            stage1_run_config=stage1_run_config,
            stage2_run_config=stage2_run_config,
        )

    def run_flashinfer():
        wrapper.run(case["q"], (case["cache_k"].unsqueeze(1), case["cache_v"].unsqueeze(1)), out=flashinfer_out)

    run_diverse()
    run_flashinfer()
    torch.cuda.synchronize()

    max_diff = (diverse_out - flashinfer_out).abs().max().item()
    mean_diff = (diverse_out - flashinfer_out).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(diverse_out.flatten(), flashinfer_out.flatten(), dim=0).item()

    diverse_ms = triton.testing.do_bench_cudagraph(run_diverse, rep=rep)
    flashinfer_ms = triton.testing.do_bench_cudagraph(run_flashinfer, rep=rep)

    return {
        "diverse_ms": diverse_ms,
        "flashinfer_ms": flashinfer_ms,
        "speedup": flashinfer_ms / diverse_ms,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-prefix-len", type=int, default=2900)
    parser.add_argument("--total-seq-len", type=int, default=3072)
    parser.add_argument("--max-len-in-batch", type=int, default=8192)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 10, 16, 24, 32])
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--kv-head-num", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[8])
    parser.add_argument(
        "--mode",
        choices=["benchmark", "joint-tune-benchmark"],
        default="joint-tune-benchmark",
        help="benchmark: use saved configs/defaults; joint-tune-benchmark: tune stage1/stage2 for each combo before benchmarking",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    block_seq = 256

    print(
        f"device={torch.cuda.get_device_name(0)} dtype={dtype} shared_prefix_len={args.shared_prefix_len} "
        f"total_seq_len={args.total_seq_len} max_len_in_batch={args.max_len_in_batch} rep={args.rep} "
        f"group_sizes={args.group_sizes} mode={args.mode}"
    )
    print(
        f"{'batch':>5} {'group':>5} {'avg_seq':>8} {'diverse_ms':>12} {'flashinfer_ms':>14} "
        f"{'speedup':>9} {'max_diff':>10} {'mean_diff':>10} {'cos_sim':>10} {'stage1_cfg':>30} {'stage2_cfg':>30}"
    )

    for batch_size in args.batches:
        valid_group_sizes = [value for value in args.group_sizes if value <= batch_size]
        best_result = None
        for max_batch_group_size in valid_group_sizes:
            case = build_case(
                batch_size=batch_size,
                shared_prefix_len=args.shared_prefix_len,
                total_seq_len=args.total_seq_len,
                max_len_in_batch=args.max_len_in_batch,
                max_batch_group_size=max_batch_group_size,
                num_heads=args.num_heads,
                kv_head_num=args.kv_head_num,
                head_dim=args.head_dim,
                dtype=dtype,
            )
            avg_seq = round(sum(case["seq_lens"]) / len(case["seq_lens"]))
            stage1_run_config, stage2_run_config = maybe_tune_configs(
                tune=args.mode == "joint-tune-benchmark",
                seq_len=args.total_seq_len,
                shared_seq_len=args.shared_prefix_len,
                max_len_in_batch=args.max_len_in_batch,
                batch_size=batch_size,
                num_heads=args.num_heads,
                kv_head_num=args.kv_head_num,
                head_dim=args.head_dim,
                block_seq=block_seq,
                max_batch_group_size=max_batch_group_size,
                dtype=dtype,
            )
            metrics = benchmark_one_case(
                case=case,
                num_heads=args.num_heads,
                kv_head_num=args.kv_head_num,
                head_dim=args.head_dim,
                rep=args.rep,
                max_batch_group_size=max_batch_group_size,
                stage1_run_config=stage1_run_config,
                stage2_run_config=stage2_run_config,
            )
            print(
                f"{batch_size:5d} {max_batch_group_size:5d} {avg_seq:8.1f} "
                f"{metrics['diverse_ms']:12.4f} {metrics['flashinfer_ms']:14.4f} "
                f"{metrics['speedup']:9.3f} {metrics['max_diff']:10.4e} "
                f"{metrics['mean_diff']:10.4e} {metrics['cos_sim']:10.6f} "
                f"{str(stage1_run_config):>30} {str(stage2_run_config):>30}"
            )
            if best_result is None or metrics["diverse_ms"] < best_result["diverse_ms"]:
                best_result = {
                    "group_size": max_batch_group_size,
                    "stage1_run_config": stage1_run_config,
                    "stage2_run_config": stage2_run_config,
                    **metrics,
                }

        print(
            f"BEST batch={batch_size} group={best_result['group_size']} "
            f"stage1={best_result['stage1_run_config']} stage2={best_result['stage2_run_config']} "
            f"diverse_ms={best_result['diverse_ms']:.4f} flashinfer_ms={best_result['flashinfer_ms']:.4f} "
            f"speedup={best_result['speedup']:.3f}"
        )


if __name__ == "__main__":
    main()
