# Example:
#   python test/benchmark/bench_agents.py \
#       --qps 5 \
#       --num-sessions 32 \
#       --turns-per-session 6 \
#       --isl 4096 \
#       --osl 256 \
#       --duration-sec 60 \
#       --warmup-sec 10

import argparse
import asyncio
import math
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

# Agent-style open-loop benchmark.
# - N sessions, each with a distinct long system prompt (so cache_aware spreads
#   them across P nodes); turns within a session share that prefix (KV reuse).
# - Fixed turn-emission rate (open loop): the scheduler fires one turn every
#   1/QPS seconds. If every session is still waiting on its previous turn, the
#   tick is counted as blocked -> the saturation signal.
# - Prefix grows each turn (full history re-sent), mirroring real multi-turn agents.


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return s[int(idx)]
    w = idx - lo
    return s[lo] * (1 - w) + s[hi] * w


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: 0.0 for k in ["mean", "p50", "p90", "p95", "p99", "max"]}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def filler(seed: int, target_chars: int) -> str:
    """Deterministic per-seed text of ~target_chars. Distinct seed -> distinct bytes."""
    out = []
    n = 0
    k = 0
    while n < target_chars:
        word = f"{seed:08d}-agentctx-{k} "
        out.append(word)
        n += len(word)
        k += 1
    return "".join(out)[:target_chars]


def make_system_prompt(seed: int, isl: int) -> str:
    # ~4 chars/token; distinct per session so cache_aware hashes the prefix to
    # different P nodes across sessions.
    return filler(seed, isl * 4)


def make_user_turn(session_seed: int, turn_idx: int) -> str:
    return f"[session {session_seed} turn {turn_idx}] " + filler(session_seed * 31 + turn_idx, 256)


class Session:
    __slots__ = ("sid", "system", "history", "pending", "turns_done", "birth")

    def __init__(self, sid: int, system: str):
        self.sid = sid
        self.system = system
        self.history: List[Dict[str, str]] = []
        self.pending = False
        self.turns_done = 0
        self.birth = 0.0

    def reset(self, system: str):
        self.system = system
        self.history = []
        self.pending = False
        self.turns_done = 0


async def do_turn(
    client: AsyncOpenAI,
    model: str,
    sess: Session,
    osl: int,
    metrics: List[Dict[str, Any]],
    warmup_until: float,
    rr_idx: List[int],
):
    user_msg = make_user_turn(sess.sid, sess.turns_done)
    messages = [{"role": "system", "content": sess.system}] + sess.history + [{"role": "user", "content": user_msg}]
    sess.pending = True
    t0 = time.perf_counter()
    first_t = second_t = last_t = None
    ntok = 0
    err = None
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=osl,
            temperature=0.7,
            stream=True,
        )
        async for chunk in resp:
            if not (chunk.choices and chunk.choices[0].delta):
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None) or getattr(delta, "reasoning_content", None)
            if text:
                ntok += 1
                now = time.perf_counter()
                if ntok == 1:
                    first_t = now
                elif ntok == 2:
                    second_t = now
                last_t = now
        end = time.perf_counter()
        if ntok < 1:
            err = "no tokens"
        else:
            # append assistant reply so the next turn's prefix includes it
            sess.history.append({"role": "assistant", "content": "(reply)"})
            if time.perf_counter() >= warmup_until:
                metrics.append(
                    {
                        "ttft": (first_t - t0) * 1000,
                        "itl": ((second_t - first_t) * 1000) if (first_t and second_t) else None,
                        "tpot": (((last_t - first_t) / (ntok - 1)) * 1000)
                        if (ntok > 1 and first_t and last_t)
                        else 0.0,
                        "lat": (end - t0) * 1000,
                        "turn_idx": sess.turns_done,
                        "sid": sess.sid,
                    }
                )
    except Exception as e:  # keep the run alive; record failure count only
        err = str(e)
    finally:
        sess.turns_done += 1
        sess.pending = False
        if err:
            metrics.append({"error": err, "turn_idx": sess.turns_done, "sid": sess.sid})
        rr_idx[0] += 1


async def run(args):
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    interval = 1.0 / args.qps if args.qps > 0 else 0.0

    # fresh set of sessions; distinct system prompt per session
    sessions = [Session(sid=sid, system=make_system_prompt(sid, args.isl)) for sid in range(args.num_sessions)]
    rr_idx = [0]  # round-robin cursor
    metrics: List[Dict[str, Any]] = []
    fired = 0
    blocked = 0
    warmup_until = time.perf_counter() + args.warmup_sec

    print(
        f"agent bench: qps={args.qps} sessions={args.num_sessions} turns/session={args.turns_per_session} "
        f"isl={args.isl} osl={args.osl} duration={args.duration_sec}s warmup={args.warmup_sec}s"
    )

    start = time.perf_counter()
    end_at = start + args.duration_sec
    next_tick = start

    async def fire_one():
        nonlocal fired, blocked
        # pick next non-pending session (round-robin over ready ones)
        n = len(sessions)
        chosen: Optional[Session] = None
        for i in range(n):
            s = sessions[(rr_idx[0] + i) % n]
            if not s.pending:
                chosen = s
                rr_idx[0] = (rr_idx[0] + i + 1) % n
                break
        if chosen is None:
            blocked += 1  # saturation: no session has finished its previous turn
            return
        fired += 1
        # retire a finished session into a fresh identity so the pool stays populated
        if chosen.turns_done >= args.turns_per_session:
            chosen.reset(system=make_system_prompt(seed=10_000_000 + rr_idx[0] + fired, isl=args.isl))
            chosen.birth = time.perf_counter()
        asyncio.create_task(do_turn(client, args.model, chosen, args.osl, metrics, warmup_until, rr_idx))

    # open-loop arrival loop: fire at fixed rate regardless of completions
    while True:
        now = time.perf_counter()
        if now >= end_at:
            break
        if now >= next_tick:
            await fire_one()
            next_tick += interval
            if interval == 0:
                next_tick = now
        else:
            await asyncio.sleep(min(next_tick - now, 0.01))

    # drain in-flight
    deadline = time.perf_counter() + args.drain_sec
    while any(s.pending for s in sessions) and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)

    total_time = time.perf_counter() - start
    summarize(metrics, fired, blocked, total_time, args)


def summarize(metrics, fired, blocked, total_time, args):
    ok = [m for m in metrics if "error" not in m]
    fail = [m for m in metrics if "error" in m]
    ttfts = [m["ttft"] for m in ok]
    tpots = [m["tpot"] for m in ok if m["tpot"] > 0]
    lats = [m["lat"] for m in ok]

    # break TTFT down by turn index to show prefix-cache benefit (turn 0 vs later)
    by_turn: Dict[int, List[float]] = {}
    for m in ok:
        by_turn.setdefault(m["turn_idx"], []).append(m["ttft"])

    def line(name, s):
        print(
            f"  {name:<30} mean={s['mean']:8.1f}  p50={s['p50']:8.1f}  "
            f"p90={s['p90']:8.1f}  p95={s['p95']:8.1f}  p99={s['p99']:8.1f}  max={s['max']:8.1f}  (ms)"
        )

    print("\n" + "=" * 96)
    print("AGENT OPEN-LOOP BENCHMARK RESULTS")
    print("=" * 96)
    print(f"  target QPS        : {args.qps}")
    print(f"  fired turns       : {fired}    blocked ticks (saturation): {blocked}")
    print(f"  completed (ok/fail): {len(ok)} / {len(fail)}")
    print(f"  achieved turn QPS : {len(ok) / total_time:.2f} /s   (wall {total_time:.1f}s)")
    print("-" * 96)
    line("TTFT", stats(ttfts))
    line("TPOT", stats(tpots))
    line("Total turn latency", stats(lats))
    print("-" * 96)
    print("  TTFT by turn index (prefix grows each turn; later turns should be faster if cache hits):")
    for idx in sorted(by_turn):
        s = stats(by_turn[idx])
        print(
            f"    turn {idx:<3} n={len(by_turn[idx]):<5} mean={s['mean']:8.1f}  "
            f"p50={s['p50']:8.1f}  p90={s['p90']:8.1f} (ms)"
        )
    print("=" * 96 + "\n")


def main():
    p = argparse.ArgumentParser(description="Agent multi-turn open-loop benchmark")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--base-url", default="http://127.0.0.1:16666/v1")
    p.add_argument("--model", default="qwen35")
    p.add_argument("--qps", type=float, default=5.0, help="turn emission rate (open loop)")
    p.add_argument("--num-sessions", type=int, default=32, help="concurrent sessions; distinct prefix each")
    p.add_argument("--turns-per-session", type=int, default=6, help="turns before a session is recycled")
    p.add_argument("--isl", type=int, default=2048, help="system-prompt length (tokens) per session")
    p.add_argument("--osl", type=int, default=256, help="max output tokens per turn")
    p.add_argument("--duration-sec", type=float, default=60.0, help="emission window")
    p.add_argument("--warmup-sec", type=float, default=10.0, help="discard metrics before this")
    p.add_argument("--drain-sec", type=float, default=120.0, help="max wait for in-flight turns after window")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
