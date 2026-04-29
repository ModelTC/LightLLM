"""
Test: does synchronous tokenizer.encode block the asyncio event loop?

A "heartbeat" coroutine ticks every 1ms. If encode blocks the loop,
heartbeat gaps will be >> 1ms. With run_in_executor, gaps stay small.
"""

import asyncio
import time
import statistics
from transformers import AutoTokenizer

MODEL_DIR = "/nvme/models/Qwen3.5-35B-A3B"
# Repeat a sentence to simulate long input
LONG_TEXT = "This is a test sentence for tokenizer performance benchmarking. " * 2000


async def heartbeat(interval_s: float, gaps: list, stop_event: asyncio.Event):
    """Record the actual gap between each tick."""
    last = time.perf_counter()
    while not stop_event.is_set():
        await asyncio.sleep(interval_s)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


async def test_sync_encode(tokenizer, text):
    """Tokenize synchronously — expected to block the event loop."""
    gaps = []
    stop = asyncio.Event()
    hb = asyncio.create_task(heartbeat(0.001, gaps, stop))

    t0 = time.perf_counter()
    # This runs on the event loop thread — blocks everything
    _ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - t0

    stop.set()
    await hb
    return elapsed, gaps, len(_ids)


async def test_executor_encode(tokenizer, text):
    """Tokenize via run_in_executor — should NOT block the loop."""
    gaps = []
    stop = asyncio.Event()
    hb = asyncio.create_task(heartbeat(0.001, gaps, stop))

    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    _ids = await loop.run_in_executor(None, tokenizer.encode, text)
    elapsed = time.perf_counter() - t0

    stop.set()
    await hb
    return elapsed, gaps, len(_ids)


def report(name, elapsed, gaps):
    if not gaps:
        print(f"  [{name}] encode took {elapsed * 1000:.1f}ms, no heartbeat ticks recorded")
        return
    max_gap = max(gaps) * 1000
    p99_gap = sorted(gaps)[int(len(gaps) * 0.99)] * 1000
    mean_gap = statistics.mean(gaps) * 1000
    ticks = len(gaps)
    print(
        f"  [{name}] encode: {elapsed * 1000:.1f}ms | "
        f"heartbeat ticks: {ticks} | "
        f"gap mean: {mean_gap:.1f}ms, p99: {p99_gap:.1f}ms, max: {max_gap:.1f}ms"
    )


async def main():
    print(f"Loading tokenizer from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    text_tokens_approx = len(LONG_TEXT.split())
    print(f"Text length: {len(LONG_TEXT)} chars, ~{text_tokens_approx} words\n")

    # Warmup
    tokenizer.encode("warmup")

    print("=== Sync encode (blocks event loop) ===")
    elapsed, gaps, n_tokens = await test_sync_encode(tokenizer, LONG_TEXT)
    report("sync", elapsed, gaps)
    print(f"  Token count: {n_tokens}")
    if gaps:
        blocked = max(gaps) * 1000
        print(f"  -> Event loop was blocked for up to {blocked:.0f}ms!")

    print()

    print("=== run_in_executor encode (non-blocking) ===")
    elapsed, gaps, n_tokens = await test_executor_encode(tokenizer, LONG_TEXT)
    report("executor", elapsed, gaps)
    print(f"  Token count: {n_tokens}")
    if gaps:
        max_gap = max(gaps) * 1000
        print(f"  -> Max event loop gap: {max_gap:.1f}ms (should be close to 1ms)")


if __name__ == "__main__":
    asyncio.run(main())
