"""实测 CUDA Event 等待是否释放 GIL，以及服务保活线程等待时主线程能否执行 Python。

示例：CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python test/benchmark/benchmark_event_synchronize_gil.py
所有等待都在 cuda_wrapper 的同一个后台线程池执行，主线程只做 Python 计数和时间采样。
PyDLL.usleep 是持有 GIL 的负对照，CDLL.usleep 是释放 GIL 的正对照。
"""

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time


METHODS = (
    "pydll_holds_gil",
    "cdll_releases_gil",
    "event_blocking_true",
    "event_blocking_false",
    "service_release_worker",
)
COUNTER_CHUNK = 4096


def calibrate_delay(torch, stream, wait_ms):
    """先预热，再按实际 GPU 时间估算约 wait_ms 的延时，避免依赖空闲时的 SM 频率。"""
    reference_cycles = 50_000_000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(reference_cycles)
    stream.synchronize()
    durations = []
    for _ in range(3):
        with torch.cuda.stream(stream):
            start.record(stream)
            torch.cuda._sleep(reference_cycles)
            end.record(stream)
        end.synchronize()
        durations.append(start.elapsed_time(end))
    return int(reference_cycles * wait_ms / statistics.median(durations)), durations


def timed_wait(wait, state):
    state["thread"] = threading.current_thread().name
    state["start_ns"] = time.perf_counter_ns()
    try:
        wait()
    finally:
        state["end_ns"] = time.perf_counter_ns()
        state["finished"] = True


def count_python_work(state):
    # 此循环不调用 torch、CUDA、sleep、锁或 Future；只有 Python 整数加法和时间采样。
    # 每 4096 次采样一次，既能验证持续推进，也避免保存过多时间戳。
    samples = []
    counter = 0
    while not state["finished"]:
        for _ in range(COUNTER_CHUNK):
            counter += 1
        samples.append((time.perf_counter_ns(), counter))
    return samples


def run_trial(torch, wrapper, stream, host_waits, cycles, method, repeat, wait_ms, margin_ms):
    event = None
    timing_start, timing_end = None, None
    sources = None
    pending_before_wait = None
    if method in host_waits:
        wait = lambda: host_waits[method](int(wait_ms * 1000))
    else:
        # 被测 Event 与服务保持相同的 enable_timing=False 配置；计时用独立事件。
        event = torch.cuda.Event(blocking=method != "event_blocking_false")
        timing_start = torch.cuda.Event(enable_timing=True)
        timing_end = torch.cuda.Event(enable_timing=True)
        if method == "service_release_worker":
            sources = [torch.empty(16, dtype=torch.int32, device="cpu", pin_memory=True, requires_grad=False)]
        with torch.cuda.stream(stream):
            timing_start.record(stream)
            torch.cuda._sleep(cycles)
            event.record(stream)
            timing_end.record(stream)
        pending_before_wait = not event.query()
        if method == "service_release_worker":
            wait = lambda: wrapper.PinnedH2DBatchCopier._wait_and_release_pinned_sources(event, sources)
        else:
            wait = event.synchronize

    state = {"finished": False}
    future = wrapper.PinnedH2DBatchCopier._get_release_pool().submit(timed_wait, wait, state)
    samples = count_python_work(state)
    future.result(timeout=10)
    gpu_delay_ms = None
    if event is not None:
        stream.synchronize()
        gpu_delay_ms = timing_start.elapsed_time(timing_end)

    # 严格截取等待内部区间，排除进入原生调用之前和返回 Python 之后的调度窗口。
    margin_ns = int(margin_ms * 1_000_000)
    window_start = state["start_ns"] + margin_ns
    window_end = state["end_ns"] - margin_ns
    interior_samples = [(stamp, count) for stamp, count in samples if window_start < stamp < window_end]
    increments = interior_samples[-1][1] - interior_samples[0][1] if len(interior_samples) > 1 else 0
    progress_span_ms = (
        (interior_samples[-1][0] - interior_samples[0][0]) / 1_000_000 if len(interior_samples) > 1 else 0.0
    )
    duration_ms = (state["end_ns"] - state["start_ns"]) / 1_000_000
    # 等待不足目标的 60% 或内部窗口不足 150ms 时不接受结果，避免过短等待误判 GIL 行为。
    valid_wait = duration_ms >= wait_ms * 0.6 and window_end - window_start >= 150_000_000
    expects_gil_release = method != "pydll_holds_gil"
    if expects_gil_release:
        # 正对照要求内部窗口持续推进至少 100ms；负对照要求内部完全没有 Python 采样。
        progress_matches = increments > 0 and progress_span_ms >= 100
    else:
        progress_matches = len(interior_samples) == 0
    passed = valid_wait and progress_matches and pending_before_wait is not False
    if sources is not None:
        passed = passed and len(sources) == 0

    return {
        "method": method,
        "repeat": repeat,
        "wait_ms": duration_ms,
        "gpu_delay_ms": gpu_delay_ms,
        "event_pending_before_wait": pending_before_wait,
        "interior_window_ms": max(0.0, (window_end - window_start) / 1_000_000),
        "interior_samples": len(interior_samples),
        "python_increments": increments,
        "progress_span_ms": progress_span_ms,
        "worker_thread": state["thread"],
        "sources_cleared": len(sources) == 0 if sources is not None else None,
        "valid_wait": valid_wait,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--wait-ms", type=float, default=500)
    parser.add_argument("--margin-ms", type=float, default=50)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "3":
        parser.error("请通过 CUDA_VISIBLE_DEVICES 显式选择一张空闲 GPU，不能使用 3 号卡")
    if args.repeats < 1 or args.wait_ms < 400 or not 0 < args.margin_ms < args.wait_ms / 4:
        parser.error("重复数须大于 0，等待至少 400ms，边缘排除区间须介于 0 和等待时间的四分之一之间")

    import torch
    from lightllm.common import cuda_wrapper

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    # ctypes 的两种加载方式调用同一个 libc 函数，唯一关注的区别是是否在原生等待中释放 GIL。
    libraries = {"pydll_holds_gil": ctypes.PyDLL(None), "cdll_releases_gil": ctypes.CDLL(None)}
    host_waits = {}
    for method, library in libraries.items():
        function = library.usleep
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_int
        host_waits[method] = function
    cuda_wrapper.PinnedH2DBatchCopier._get_release_pool().submit(lambda: None).result(timeout=10)
    cycles, calibration_ms = calibrate_delay(torch, stream, args.wait_ms)
    root = args.output_dir or Path("/tmp") / (
        "event_synchronize_gil_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    root.mkdir(parents=True, exist_ok=False)
    print(f"结果目录：{root}", flush=True)
    rows = []
    try:
        for repeat in range(args.repeats):
            for method in METHODS:
                rows.append(
                    run_trial(
                        torch, cuda_wrapper, stream, host_waits, cycles, method, repeat, args.wait_ms, args.margin_ms
                    )
                )
            print(f"已完成第 {repeat + 1}/{args.repeats} 轮", flush=True)
    finally:
        stream.synchronize()
        cuda_wrapper.PinnedH2DBatchCopier._get_release_pool().submit(lambda: None).result(timeout=10)
    summaries = []
    for method in METHODS:
        cases = [row for row in rows if row["method"] == method]
        summary = {
            "method": method,
            "passed": sum(row["passed"] for row in cases),
            "trials": len(cases),
            "wait_ms_median": statistics.median(row["wait_ms"] for row in cases),
            "python_increments_min": min(row["python_increments"] for row in cases),
            "python_increments_median": statistics.median(row["python_increments"] for row in cases),
            "python_increments_max": max(row["python_increments"] for row in cases),
            "progress_span_ms_min": min(row["progress_span_ms"] for row in cases),
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    evidence = {
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": visible,
            "gpu": torch.cuda.get_device_name(0),
            "python_switch_interval_s": sys.getswitchinterval(),
        },
        "config": {
            "repeats": args.repeats,
            "target_wait_ms": args.wait_ms,
            "margin_ms": args.margin_ms,
            "counter_chunk": COUNTER_CHUNK,
            "gpu_delay_cycles": cycles,
            "calibration_ms": calibration_ms,
        },
        "summary": summaries,
        "trials": rows,
        "passed": all(row["passed"] for row in rows),
    }
    (root / "result.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    if not evidence["passed"]:
        raise SystemExit("实验判据未全部满足，请查看 result.json 中的有效等待时长与内部窗口进度")


if __name__ == "__main__":
    main()
