from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
from queue import Empty, SimpleQueue
from statistics import fmean
import threading


@dataclass(frozen=True)
class StreamingWorkload:
    name: str
    prompt_ids: tuple[int, ...]
    expected_ids: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Workload requires a name")
        for field in ("prompt_ids", "expected_ids"):
            values = getattr(self, field)
            if not isinstance(values, (list, tuple)) or not values or any(type(x) is not int or x < 0 for x in values):
                raise ValueError("Workload requires exact nonnegative prompt and reference token IDs")
            object.__setattr__(self, field, tuple(values))


def distribution(values):
    if not values:
        return None
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency samples must be finite and nonnegative")
    ordered = sorted(values)
    result = dict(count=len(values), min=ordered[0], max=ordered[-1], mean=fmean(values))
    for percentile in (50, 90, 95, 99):
        position = (len(ordered) - 1) * percentile / 100
        left, right = math.floor(position), math.ceil(position)
        result[f"p{percentile}"] = ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    return result


def _validate_result(result):
    metrics, timeline = result["metrics"], result["timeline"]
    started, first, last, ended = (timeline[key] for key in ("started", "first_token", "last_token", "ended"))
    if not all(math.isfinite(value) for value in (started, first, last, ended)):
        raise ValueError("Request timeline must be finite")
    if not started <= first <= last <= ended or ended <= started:
        raise ValueError("Request timeline is incomplete or nonmonotonic")
    ids = result["token_ids"]
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Result requires actual integer output IDs")
    if type(metrics["input_tokens"]) is not int or metrics["input_tokens"] < 1:
        raise ValueError("Result requires the actual input token count")
    if type(metrics["output_tokens"]) is not int or metrics["output_tokens"] != len(ids):
        raise ValueError("Output count differs from actual token IDs")
    gaps = metrics["receiver_itl_s"]
    if len(gaps) != len(ids) - 1 or any(not math.isfinite(gap) or gap < 0 for gap in gaps):
        raise ValueError("Receiver gaps must account for every output position")
    expected = dict(ttft_s=first - started, e2e_s=ended - started)
    if len(ids) > 1:
        expected["tpot_s"] = (last - first) / (len(ids) - 1)
    elif metrics["tpot_s"] is not None:
        raise ValueError("A single output token has no TPOT")
    for key, value in expected.items():
        if not math.isclose(metrics[key], value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{key} differs from the captured request timeline")
    if not math.isclose(sum(gaps), last - first, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Receiver gaps differ from the token arrival interval")


def summarize_cohort(results):
    if not results:
        raise ValueError("An empty cohort cannot be measured")
    for result in results:
        _validate_result(result)
    elapsed = max(item["timeline"]["ended"] for item in results) - min(item["timeline"]["started"] for item in results)
    input_tokens = sum(item["metrics"]["input_tokens"] for item in results)
    output_tokens = sum(len(item["token_ids"]) for item in results)
    cache_counts = [item["metrics"].get("cached_input_tokens") for item in results]
    if any(
        count is not None and (type(count) is not int or not 0 <= count <= item["metrics"]["input_tokens"])
        for count, item in zip(cache_counts, results)
    ):
        raise ValueError("Invalid cached input count")
    cached_tokens = sum(cache_counts) if all(count is not None for count in cache_counts) else None
    events = sorted(
        [(item["timeline"]["started"], 1) for item in results] + [(item["timeline"]["ended"], -1) for item in results]
    )
    active = maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    gaps = [gap for item in results for gap in item["metrics"]["receiver_itl_s"]]
    return dict(
        requests=len(results),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_reported_requests=sum(count is not None for count in cache_counts),
        cached_input_tokens=cached_tokens,
        uncached_input_tokens=input_tokens - cached_tokens if cached_tokens is not None else None,
        elapsed_s=elapsed,
        requests_per_s=len(results) / elapsed,
        input_tokens_per_s=input_tokens / elapsed,
        output_tokens_per_s=output_tokens / elapsed,
        input_token_definition="requested prompt IDs; includes cached tokens",
        maximum_client_inflight=maximum,
        ttft_s=distribution([item["metrics"]["ttft_s"] for item in results]),
        tpot_s=distribution([item["metrics"]["tpot_s"] for item in results if item["metrics"]["tpot_s"] is not None]),
        e2e_s=distribution([item["metrics"]["e2e_s"] for item in results]),
        receiver_itl_s=distribution(gaps),
        receiver_zero_gap_fraction=sum(gap == 0 for gap in gaps) / len(gaps) if gaps else None,
        itl_definition="client arrival gaps; tokens delivered together share a timestamp",
        percentile_definition="linear interpolation between ordered observations",
    )


def run_cohort(workloads, concurrency, request, response_directory: Path):
    workloads = tuple(workloads)
    if type(concurrency) is not int or not 1 <= concurrency <= len(workloads):
        raise ValueError("Concurrency must be positive and no larger than the request count")
    if any(not isinstance(item, StreamingWorkload) for item in workloads):
        raise ValueError("Every request requires a qualified streaming workload")
    response_directory.mkdir(parents=True, exist_ok=True)
    pending = SimpleQueue()
    for index, workload in enumerate(workloads):
        pending.put((index, workload))
    records = [None] * len(workloads)
    ready = threading.Barrier(concurrency + 1)
    release = threading.Event()

    def worker():
        ready.wait(timeout=30)
        release.wait()
        while True:
            try:
                index, workload = pending.get_nowait()
            except Empty:
                return
            path = response_directory / f"{index:06d}.sse"
            record = dict(index=index, workload=workload.name, response=str(path), status="running")
            try:
                result = request(workload, path)
                record["result"] = result
                _validate_result(result)
                if result["metrics"]["input_tokens"] != len(workload.prompt_ids):
                    raise ValueError("Service input length differs from the qualified workload")
                if tuple(result["token_ids"]) != workload.expected_ids:
                    raise ValueError("Output IDs differ from the qualified reference")
                record["status"] = "passed"
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
            records[index] = record

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        try:
            ready.wait(timeout=30)
        finally:
            release.set()
        for future in futures:
            future.result()
    passed = all(item["status"] == "passed" for item in records)
    return dict(
        status="passed" if passed else "failed",
        concurrency=concurrency,
        records=records,
        metrics=summarize_cohort([item["result"] for item in records]) if passed else None,
    )
