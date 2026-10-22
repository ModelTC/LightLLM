"""实测空 H2D 能否像单元素 H2D 一样保护 raw batch 使用的 pinned storage。

示例：CUDA_VISIBLE_DEVICES=6 python test/benchmark/benchmark_pinned_empty_h2d.py
每种方法和规模在独立子进程运行，避免其他案例留下的 allocator 状态影响观察。
只操作实验自身申请的内存，不启动模型、不修改服务的生命周期保护实现。
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


METHODS = ("none", "empty_copy", "empty_to", "one_element_copy", "empty_copy_keep_view")


def protect_source(target, source, method):
    if method == "empty_copy":
        target[:0].copy_(source[:0], non_blocking=True)
    elif method == "empty_to":
        # 与 empty_copy 一样丢弃临时视图和结果；只有 keep_view 对照延长源引用。
        source[:0].to(target.device, non_blocking=True)
    elif method == "one_element_copy":
        target[:1].copy_(source[:1], non_blocking=True)
    elif method == "empty_copy_keep_view":
        # 此对照有意保留 CPU 空视图：即使 numel 为 0，它仍持有整个原 storage。
        # 如果这组成功，不能据此认为空 H2D 向 allocator 登记了流。
        view = source[:0]
        target[:0].copy_(view, non_blocking=True)
        return view
    return None


def submit_raw_batch(runtime, target, source, stream):
    attrs = runtime.cudaMemcpyAttributes()
    # 与服务相同的 Stream 模式：真正读取源数据的时间遵循上传流顺序，
    # raw batch 不会替调用方向 PyTorch host allocator 登记源内存。
    attrs.srcAccessOrder = runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    (error,) = runtime.cudaMemcpyBatchAsync(
        [target.data_ptr()],
        [source.data_ptr()],
        [source.numel() * source.element_size()],
        1,
        [attrs],
        [0],
        1,
        stream.cuda_stream,
    )
    if error != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpyBatchAsync failed: {error}")


def run_trial(torch, runtime, stream, method, token_count, repeat, delay_cycles, attempts):
    offset = 64
    allocation_size = token_count + offset * 2
    expected_value, overwritten_value = repeat + 100, -repeat - 1000
    owner = torch.full((allocation_size,), expected_value, dtype=torch.int32, pin_memory=True, requires_grad=False)
    source = owner[offset : offset + token_count]
    source_storage_ptr = owner.untyped_storage().data_ptr()
    empty_view = source[:0]
    empty_metadata = {
        "numel": empty_view.numel(),
        "data_ptr": empty_view.data_ptr(),
        "shares_source_storage": empty_view.untyped_storage().data_ptr() == source_storage_ptr,
    }
    del empty_view
    target = torch.empty(token_count, dtype=torch.int32, device="cuda", requires_grad=False)
    stream.wait_stream(torch.cuda.default_stream())
    # 同一流先延时，再提交真实 raw H2D 和待验证的保护操作。
    # 这样 CPU 能在 GPU 尚未读取源数据时释放原 Tensor、触发 allocator 复用。
    with torch.cuda.stream(stream):
        torch.cuda._sleep(delay_cycles)
        submit_raw_batch(runtime, target, source, stream)
        kept_view = protect_source(target, source, method)
    del owner, source

    held_allocations = []
    reused_while_pending = False
    reuse_seen = False
    overwrite_finished_while_pending = False
    pending_checks = 0
    try:
        for _ in range(attempts):
            if stream.query():
                break
            replacement = torch.empty(allocation_size, dtype=torch.int32, pin_memory=True, requires_grad=False)
            # 保留所有新分配块。若旧块被复用并改写，也让它活到上传流结束，
            # 将风险局限为本实验的数据被覆盖，不让 DMA 访问已经归还驱动的内存。
            held_allocations.append(replacement)
            still_pending = not stream.query()
            pending_checks += int(still_pending)
            if replacement.untyped_storage().data_ptr() == source_storage_ptr:
                reuse_seen = True
                reused_while_pending = still_pending
                if still_pending:
                    replacement.fill_(overwritten_value)
                    overwrite_finished_while_pending = not stream.query()
                break
    finally:
        # 包括异常路径也先等待本实验的上传流，再释放 replacement 和保留的空视图。
        stream.synchronize()
    del kept_view
    result = target.cpu()
    matching_tokens = int(result.eq(expected_value).sum().item())
    overwritten_tokens = int(result.eq(overwritten_value).sum().item())

    reused_after_completion = None
    if not reuse_seen:
        # 正向对照还应证明上传完成后原分配块可以再次使用，而非被永久保留。
        reused_after_completion = False
        for _ in range(attempts):
            replacement = torch.empty(allocation_size, dtype=torch.int32, pin_memory=True, requires_grad=False)
            held_allocations.append(replacement)
            if replacement.untyped_storage().data_ptr() == source_storage_ptr:
                reused_after_completion = True
                break
    return {
        "repeat": repeat,
        "method": method,
        "token_count": token_count,
        "payload_bytes": token_count * 4,
        "source_storage_ptr": source_storage_ptr,
        "empty_view": empty_metadata,
        "pending_allocation_checks": pending_checks,
        "source_reused_while_pending": reused_while_pending,
        "overwrite_finished_while_pending": overwrite_finished_while_pending,
        "matching_tokens": matching_tokens,
        "overwritten_tokens": overwritten_tokens,
        "output_correct": matching_tokens == token_count,
        "source_reused_after_completion": reused_after_completion,
    }


def run_worker(args):
    import torch
    from cuda.bindings import runtime

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    # 首次初始化必须在延时试验前完成，不能让导入/驱动初始化耗时掩盖复用窗口。
    source = torch.ones(256, dtype=torch.int32, pin_memory=True, requires_grad=False)
    target = torch.empty(256, dtype=torch.int32, device="cuda", requires_grad=False)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1)
        submit_raw_batch(runtime, target, source, stream)
        protect_source(target, source, args.method)
    stream.synchronize()
    del source, target
    rows = [
        run_trial(torch, runtime, stream, args.method, args.token_count, repeat, args.delay_cycles, args.attempts)
        for repeat in range(args.repeats)
    ]
    print(json.dumps({"torch_version": torch.__version__, "cuda_version": torch.version.cuda, "trials": rows}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8192, 262144], help="展开后的 int32 token 数")
    parser.add_argument("--delay-cycles", type=int, default=200_000_000)
    parser.add_argument("--attempts", type=int, default=64)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--token-count", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    # 显式限制到一张卡，防止实验误用用户禁止的 GPU 3 或其他运行中的服务。
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "3":
        parser.error("请通过 CUDA_VISIBLE_DEVICES 显式选择一张空闲 GPU，不能使用 3 号卡")
    if args.repeats < 1 or min(args.sizes) < 1 or args.delay_cycles < 1 or args.attempts < 1:
        parser.error("重复数、规模、延时和分配尝试次数必须大于 0")
    if args.worker:
        run_worker(args)
        return

    root = args.output_dir or Path("/tmp") / (
        "pinned_empty_h2d_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    root.mkdir(parents=True, exist_ok=False)
    all_trials, summaries = [], []
    version = None
    for token_count in args.sizes:
        for method in METHODS:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--method",
                method,
                "--token-count",
                str(token_count),
                "--repeats",
                str(args.repeats),
                "--delay-cycles",
                str(args.delay_cycles),
                "--attempts",
                str(args.attempts),
            ]
            child = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
            (root / f"{token_count}_{method}.json").write_text(child.stdout)
            if child.stderr:
                (root / f"{token_count}_{method}.stderr").write_text(child.stderr)
            data = json.loads(child.stdout)
            version = {key: data[key] for key in ("torch_version", "cuda_version")}
            rows = data["trials"]
            all_trials.extend(rows)
            summary = {
                "token_count": token_count,
                "method": method,
                "repeats": len(rows),
                "pending_window_trials": sum(row["pending_allocation_checks"] > 0 for row in rows),
                "reused_while_pending": sum(row["source_reused_while_pending"] for row in rows),
                "overwrite_finished_while_pending": sum(row["overwrite_finished_while_pending"] for row in rows),
                "correct_outputs": sum(row["output_correct"] for row in rows),
                "fully_overwritten_outputs": sum(row["overwritten_tokens"] == token_count for row in rows),
                "reusable_after_completion": sum(row["source_reused_after_completion"] is True for row in rows),
            }
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    evidence = {
        **version,
        "cuda_visible_devices": visible,
        "delay_cycles": args.delay_cycles,
        "allocation_attempts": args.attempts,
        "source_slice_offset_tokens": 64,
        "raw_copy": "cudaMemcpyBatchAsync with cudaMemcpySrcAccessOrderStream",
        "isolated_subprocess_per_method_and_size": True,
        "replacement_allocations_retained_until_stream_completion": True,
        "summary": summaries,
        "trials": all_trials,
    }
    (root / "result.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    with (root / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"实验结果：{root}", flush=True)


if __name__ == "__main__":
    main()
