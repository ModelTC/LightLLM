"""Agent-style multi-turn open-loop stress test.

What it does
------------
N concurrent agent sessions, each with a distinct long system prompt (so a
cache_aware router spreads them across prefill nodes); turns within a session
re-send the growing history (KV reuse). Turn emission is OPEN-LOOP: one turn is
fired every 1/QPS seconds regardless of whether previous turns finished. If every
session is still busy when a tick fires, the tick is counted as "blocked" -- the
saturation signal.

Two modes
---------
1) Single load point (default). Apply a fixed QPS for a fixed duration.

   exp -m "single agent stress load" python test/benchmark/bench_agents.py \\
       --base-url http://127.0.0.1:8000/v1 --model qwen35 \\
       --qps 5 --num-sessions 32 --turns-per-session 6 \\
       --isl 2048 --osl 256 --duration-sec 60 --warmup-sec 10

2) Stress test (--stress). Increase QPS step by step until the target is no
   longer sustained, then keep that pressure for a hold phase. Every phase is
   logged and a markdown report records where saturation started.

   exp -m "agent stress ramp" python test/benchmark/bench_agents.py --stress
   exp -m "quick agent stress ramp" python test/benchmark/bench_agents.py --stress --quick
   exp -m "custom agent stress ramp" python test/benchmark/bench_agents.py --stress \\
       --stress-start-qps 5 --stress-step-qps 5 --stress-max-qps 100 \\
       --stress-step-sec 60 --stress-hold-sec 300

Key flags
---------
  --qps             turn emission rate (open loop)
  --num-sessions    concurrent sessions; distinct prefix each (= concurrency cap)
  --turns-per-session  turns before a session is recycled into a fresh identity
  --isl / --osl     system-prompt length (tokens) / max output tokens per turn
  --duration-sec    emission window
  --warmup-sec      discard metrics before this in single-load mode
  --report-sec      live progress print interval (0 disables)
  --stress-*        ramp, hold, and saturation thresholds for stress mode

Reading the output
------------------
  - SATURATED: achieved QPS fell below the configured ratio, requests failed,
    or all sessions were busy often enough to block new turns.
  - blocked > 0: the session pool is full. Raise --num-sessions if the client,
    rather than the server, is limiting the offered load.
  - error breakdown: top error strings tell you whether failures are timeouts,
    connection resets, or empty streams.
"""

import argparse
import asyncio
import contextlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


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
    __slots__ = ("sid", "system", "history", "pending", "turns_done")

    def __init__(self, sid: int, system: str):
        self.sid = sid
        self.system = system
        self.history: List[Dict[str, str]] = []
        self.pending = False
        self.turns_done = 0

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
):
    user_msg = make_user_turn(sess.sid, sess.turns_done)
    messages = [{"role": "system", "content": sess.system}] + sess.history + [{"role": "user", "content": user_msg}]
    t0 = time.perf_counter()
    first_t = last_t = prev_t = None
    itl_sum = 0.0
    itl_n = 0
    ntok = 0
    reply_parts: List[str] = []
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
            extra = getattr(delta, "model_extra", None) or {}
            text = (
                getattr(delta, "content", None)
                or extra.get("reasoning")
                or extra.get("reasoning_content")
                or extra.get("text")
                or getattr(delta, "reasoning", None)
                or getattr(delta, "reasoning_content", None)
            )
            if text:
                reply_parts.append(text)
                now = time.perf_counter()
                ntok += 1
                if first_t is None:
                    first_t = now
                else:
                    itl_sum += now - prev_t
                    itl_n += 1
                prev_t = now
                last_t = now
        end = time.perf_counter()
        if ntok < 1:
            err = "no tokens"
        else:
            # append the real generated reply so the next turn's prefix grows
            # like a real multi-turn agent (full history re-sent)
            sess.history.append({"role": "assistant", "content": "".join(reply_parts)})
            metrics.append(
                {
                    "ttft": (first_t - t0) * 1000,
                    "itl": (itl_sum / itl_n * 1000) if itl_n else None,
                    "tpot": (((last_t - first_t) / (ntok - 1)) * 1000) if (ntok > 1 and first_t and last_t) else 0.0,
                    "lat": (end - t0) * 1000,
                    "turn_idx": sess.turns_done,
                    "sid": sess.sid,
                    "measured": time.perf_counter() >= warmup_until,
                }
            )
    except Exception as e:  # keep the run alive; record failure count only
        err = str(e)
    finally:
        sess.turns_done += 1
        sess.pending = False
        if err:
            metrics.append(
                {
                    "error": err,
                    "turn_idx": sess.turns_done,
                    "sid": sess.sid,
                    "measured": time.perf_counter() >= warmup_until,
                }
            )


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
        f"agent stress load: qps={args.qps} sessions={args.num_sessions} turns/session={args.turns_per_session} "
        f"isl={args.isl} osl={args.osl} duration={args.duration_sec}s warmup={args.warmup_sec}s"
    )

    start = time.perf_counter()
    end_at = start + args.duration_sec
    next_tick = start

    async def reporter():
        # periodic one-liner: progress + running TTFT/TPOT over metrics so far
        try:
            while True:
                await asyncio.sleep(args.report_sec)
                now = time.perf_counter()
                if now >= end_at:
                    return
                ok_now = [m for m in metrics if "error" not in m]
                errs = [m for m in metrics if "error" in m]
                elapsed = now - start
                done = len(ok_now)
                ttft_s = stats([m["ttft"] for m in ok_now])
                tpot_s = stats([m["tpot"] for m in ok_now if m["tpot"] > 0])
                in_flight = sum(1 for s in sessions if s.pending)
                last_err = f"  last_err={errs[-1]['error'][:120]!r}" if errs else ""
                print(
                    f"[{elapsed:5.1f}s] fired={fired:<5} blocked={blocked:<5} "
                    f"ok={done:<5} err={len(errs):<3} pend={in_flight:<3} "
                    f"achQPS={done / elapsed if elapsed > 0 else 0:.2f}  "
                    f"TTFT p50={ttft_s['p50']:.0f} p95={ttft_s['p95']:.0f} max={ttft_s['max']:.0f}  "
                    f"TPOT p50={tpot_s['p50']:.0f} p95={tpot_s['p95']:.0f}"
                    f"{last_err}"
                )
        except asyncio.CancelledError:
            pass

    reporter_task = asyncio.create_task(reporter()) if args.report_sec > 0 else None

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
        # claim before scheduling: fire_one doesn't await, so without this a
        # catch-up burst re-picks the same session before do_turn runs.
        chosen.pending = True
        asyncio.create_task(do_turn(client, args.model, chosen, args.osl, metrics, warmup_until))

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
    reporter_task and reporter_task.cancel()
    if reporter_task:
        await reporter_task

    total_time = time.perf_counter() - start
    return summarize(metrics, fired, blocked, total_time, args)


def summarize(metrics, fired, blocked, total_time, args):
    all_ok = [m for m in metrics if "error" not in m]
    all_fail = [m for m in metrics if "error" in m]
    ok = [m for m in all_ok if m["measured"]]
    ttfts = [m["ttft"] for m in ok]
    itls = [m["itl"] for m in ok if m["itl"] is not None]
    tpots = [m["tpot"] for m in ok if m["tpot"] > 0]
    lats = [m["lat"] for m in ok]
    completed = len(all_ok) + len(all_fail)
    unfinished = max(0, fired - completed)
    attempted = fired + blocked
    failure_rate = (len(all_fail) + unfinished) / fired if fired else 0.0
    blocked_rate = blocked / attempted if attempted else 0.0
    achieved_qps = len(all_ok) / args.duration_sec if args.duration_sec > 0 else 0.0

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
    print("AGENT OPEN-LOOP STRESS LOAD RESULTS")
    print("=" * 96)
    print(f"  target QPS        : {args.qps}")
    print(f"  fired turns       : {fired}    blocked ticks (saturation): {blocked}")
    print(f"  completed (ok/fail): {len(all_ok)} / {len(all_fail)}    unfinished after drain: {unfinished}")
    print(f"  failure / blocked rate: {failure_rate:.2%} / {blocked_rate:.2%}")
    if all_fail:
        from collections import Counter

        c = Counter(m["error"][:160] for m in all_fail)
        print("  error breakdown:")
        for msg, n in c.most_common(5):
            print(f"    [{n:>3}] {msg}")
    print(f"  achieved turn QPS : {achieved_qps:.2f} /s   (wall incl. drain {total_time:.1f}s)")
    print("-" * 96)
    line("TTFT", stats(ttfts))
    line("ITL", stats(itls))
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

    from collections import Counter

    top_err = Counter(m["error"][:120] for m in all_fail).most_common(1)
    return {
        "target_qps": args.qps,
        "fired": fired,
        "blocked": blocked,
        "blocked_rate": blocked_rate,
        "ok": len(all_ok),
        "fail": len(all_fail),
        "failure_rate": failure_rate,
        "unfinished": unfinished,
        "achieved_qps": achieved_qps,
        "wall": total_time,
        "ttft": stats(ttfts),
        "itl": stats(itls),
        "tpot": stats(tpots),
        "lat": stats(lats),
        "top_error": top_err[0][0] if top_err else None,
    }


class _Tee:
    """Write phase output to the terminal and its log at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _run_phase(name: str, out_dir: Path, base: dict, **kw) -> dict:
    args = make_args(**{**base, **kw})
    print(
        f"\n>>> {name}  qps={args.qps} sessions={args.num_sessions} "
        f"isl={args.isl} osl={args.osl} dur={args.duration_sec}s warmup={args.warmup_sec}s",
        flush=True,
    )
    t0 = time.perf_counter()
    with Path(out_dir, f"{name}.log").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
            res = asyncio.run(run(args))
    res["phase"] = name
    res["phase_wall"] = time.perf_counter() - t0
    ttft, tpot = res["ttft"], res["tpot"]
    err = f"  err={res['fail']}({res['top_error'][:40]})" if res["fail"] else ""
    print(
        f"    done in {res['phase_wall']:4.0f}s | ok={res['ok']:<4} fail={res['fail']:<3} "
        f"unfinished={res['unfinished']:<3} blocked={res['blocked']:<3} achQPS={res['achieved_qps']:.2f} "
        f"TTFT p50={ttft['p50']:.0f} p95={ttft['p95']:.0f}  "
        f"TPOT p50={tpot['p50']:.0f} p95={tpot['p95']:.0f}{err}",
        flush=True,
    )
    return res


def _md_table(rows: list) -> str:
    head = (
        "| phase | status | targetQPS | fired | ok | fail | unfinished | blocked% | achQPS | "
        "TTFT p50 | TTFT p95 | reason |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    body = []
    for r in rows:
        reason = "; ".join(r["saturation_reasons"]) or "-"
        reason = reason.replace("|", "\\|")
        body.append(
            f"| {r['phase']} | {r['status']} | {r['target_qps']:g} | {r['fired']} | {r['ok']} | "
            f"{r['fail']} | {r['unfinished']} | {r['blocked_rate']:.1%} | {r['achieved_qps']:.2f} | "
            f"{r['ttft']['p50']:.0f} | {r['ttft']['p95']:.0f} | {reason} |"
        )
    return head + "\n" + "\n".join(body)


def _saturation_reasons(result: dict, args) -> List[str]:
    reasons = []
    ratio = result["achieved_qps"] / result["target_qps"] if result["target_qps"] else 0.0
    if ratio < args.stress_min_achieved_ratio:
        reasons.append(f"achieved/target {ratio:.1%} < {args.stress_min_achieved_ratio:.1%}")
    if result["failure_rate"] > args.stress_max_error_rate:
        reasons.append(f"failure rate {result['failure_rate']:.1%} > {args.stress_max_error_rate:.1%}")
    if result["blocked_rate"] > args.stress_max_blocked_rate:
        reasons.append(f"blocked rate {result['blocked_rate']:.1%} > {args.stress_max_blocked_rate:.1%}")
    return reasons


def _qps_label(qps: float) -> str:
    return f"{qps:g}".replace(".", "_")


def _stress_qps_steps(start: float, step: float, maximum: float) -> List[float]:
    values = []
    current = start
    while current < maximum:
        values.append(current)
        current += step
    if not values or not math.isclose(values[-1], maximum):
        values.append(maximum)
    return values


def run_stress(args) -> None:
    scale = 0.25 if args.quick else 1.0
    step_duration = max(5.0, args.stress_step_sec * scale)
    hold_duration = max(10.0, args.stress_hold_sec * scale)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "num_sessions": args.num_sessions,
        "turns_per_session": args.turns_per_session,
        "isl": args.isl,
        "osl": args.osl,
        "warmup_sec": 0,
        "drain_sec": args.drain_sec,
        "report_sec": args.report_sec,
    }

    print(
        f"STRESS TEST -> {args.base_url} model={args.model} isl={args.isl} osl={args.osl} "
        f"qps={args.stress_start_qps:g}..{args.stress_max_qps:g} step={args.stress_step_qps:g} "
        f"step_sec={step_duration:g} hold_sec={hold_duration:g} sessions={args.num_sessions} "
        f"quick={args.quick} out={out_dir}",
        flush=True,
    )
    t_start = time.perf_counter()
    results = []
    hold_qps = args.stress_max_qps
    first_saturated_qps = None

    for qps in _stress_qps_steps(args.stress_start_qps, args.stress_step_qps, args.stress_max_qps):
        result = _run_phase(
            f"ramp_qps_{_qps_label(qps)}",
            out_dir,
            base,
            qps=qps,
            duration_sec=step_duration,
        )
        result["saturation_reasons"] = _saturation_reasons(result, args)
        result["status"] = "SATURATED" if result["saturation_reasons"] else "PASS"
        results.append(result)
        if result["saturation_reasons"]:
            first_saturated_qps = qps
            hold_qps = qps
            print(
                f"    saturation detected at {qps:g} QPS: {'; '.join(result['saturation_reasons'])}",
                flush=True,
            )
            break

    hold = _run_phase(
        f"hold_qps_{_qps_label(hold_qps)}",
        out_dir,
        base,
        qps=hold_qps,
        duration_sec=hold_duration,
    )
    hold["saturation_reasons"] = _saturation_reasons(hold, args)
    hold["status"] = "SATURATED" if hold["saturation_reasons"] else "PASS"
    results.append(hold)
    total = time.perf_counter() - t_start

    table = _md_table(results)
    print("\n" + "=" * 96)
    saturation = f"first saturation: {first_saturated_qps:g} QPS" if first_saturated_qps else "no ramp saturation"
    print(f"STRESS TEST DONE in {total:.0f}s   ({len(results)} phases, {saturation})")
    print("=" * 96)
    print(table)
    Path(out_dir, "report.md").write_text(
        f"# Agent stress test report\n\nTarget: `{args.base_url}` model=`{args.model}` "
        f"isl={args.isl} osl={args.osl} sessions={args.num_sessions} quick={args.quick}\n\n"
        f"Result: **{saturation}**. Total wall: {total:.0f}s.\n\n{table}\n",
        encoding="utf-8",
    )
    print(f"\nreport -> {Path(out_dir, 'report.md')}   (per-phase logs: {out_dir}/*.log)")


def build_parser():
    p = argparse.ArgumentParser(description="Agent multi-turn open-loop stress test")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen35")
    p.add_argument("--qps", type=float, default=5.0, help="turn emission rate (open loop)")
    p.add_argument(
        "--num-sessions",
        type=int,
        default=128,
        help="concurrent sessions; distinct prefix each",
    )
    p.add_argument(
        "--turns-per-session",
        type=int,
        default=6,
        help="turns before a session is recycled",
    )
    p.add_argument(
        "--isl",
        type=int,
        default=2048,
        help="system-prompt length (tokens) per session",
    )
    p.add_argument("--osl", type=int, default=256, help="max output tokens per turn")
    p.add_argument("--duration-sec", type=float, default=60.0, help="emission window")
    p.add_argument("--warmup-sec", type=float, default=10.0, help="discard metrics before this")
    p.add_argument(
        "--drain-sec",
        type=float,
        default=120.0,
        help="max wait for in-flight turns after window",
    )
    p.add_argument(
        "--report-sec",
        type=float,
        default=5.0,
        help="live progress print interval (0 disables)",
    )
    p.add_argument(
        "--stress",
        action="store_true",
        help="ramp QPS to saturation, then hold the pressure",
    )
    p.add_argument(
        "--stress-start-qps",
        type=float,
        default=5.0,
        help="first QPS in the stress ramp",
    )
    p.add_argument(
        "--stress-step-qps",
        type=float,
        default=5.0,
        help="QPS added after each passing ramp phase",
    )
    p.add_argument(
        "--stress-max-qps",
        type=float,
        default=40.0,
        help="highest QPS offered by the stress ramp",
    )
    p.add_argument(
        "--stress-step-sec",
        type=float,
        default=60.0,
        help="duration of each ramp phase",
    )
    p.add_argument(
        "--stress-hold-sec",
        type=float,
        default=300.0,
        help="duration of the final pressure hold",
    )
    p.add_argument(
        "--stress-max-error-rate",
        type=float,
        default=0.01,
        help="stop ramp above this failed/unfinished rate",
    )
    p.add_argument(
        "--stress-max-blocked-rate",
        type=float,
        default=0.01,
        help="stop ramp above this blocked-tick rate",
    )
    p.add_argument(
        "--stress-min-achieved-ratio",
        type=float,
        default=0.90,
        help="stop ramp when achieved QPS / target QPS falls below this ratio",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="(stress) run phases at one quarter duration",
    )
    p.add_argument(
        "--out",
        default="stress_test_out",
        help="(stress) dir for per-phase logs + report.md",
    )
    return p


def make_args(**overrides):
    args = build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def validate_args(parser, args) -> None:
    positive = {
        "qps": args.qps,
        "num-sessions": args.num_sessions,
        "turns-per-session": args.turns_per_session,
        "isl": args.isl,
        "osl": args.osl,
        "duration-sec": args.duration_sec,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name} must be greater than zero")
    if args.warmup_sec < 0 or args.drain_sec < 0 or args.report_sec < 0:
        parser.error("--warmup-sec, --drain-sec, and --report-sec cannot be negative")
    if args.stress:
        if args.stress_start_qps <= 0 or args.stress_step_qps <= 0 or args.stress_max_qps <= 0:
            parser.error("stress QPS values must be greater than zero")
        if args.stress_start_qps > args.stress_max_qps:
            parser.error("--stress-start-qps cannot exceed --stress-max-qps")
        if args.stress_step_sec <= 0 or args.stress_hold_sec <= 0:
            parser.error("stress phase durations must be greater than zero")
        for name in (
            "stress_max_error_rate",
            "stress_max_blocked_rate",
            "stress_min_achieved_ratio",
        ):
            if not 0 <= getattr(args, name) <= 1:
                parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.stress:
        run_stress(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
