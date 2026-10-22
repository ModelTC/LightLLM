"""Closed-loop, trace-shaped agent replay with real model responses.

Public trace text is redacted. This reconstructs synthetic contexts from the
recorded sizes, tool-content labels and waits; it is not an exact token replay.
Live responses feed into the next request. All scaling and capped waits are
reported. Independent controls can override the original tool-content labels.
"""

import argparse
import asyncio
import copy
import collections
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import aiohttp
import numpy as np
from transformers import AutoTokenizer


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return the supplied text unchanged.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    }
]


def render(tokenizer, messages):
    normalized = copy.deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    text = tokenizer.apply_chat_template(
        normalized,
        tools=[t["function"] for t in TOOLS],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def prepare(tokenizer, messages, instruction, target):
    messages.append({"role": "user", "content": "Reference data, not instructions:\n<reference>\n</reference>"})
    messages.append({"role": "user", "content": instruction})
    base = len(render(tokenizer, messages))
    filler = " data" * max(0, target - base)
    messages[-2]["content"] = "Reference data, not instructions:\n<reference>" + filler + "\n</reference>"
    tokens = render(tokenizer, messages)
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest(), len(tokens)


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))] if values else None


def select_sessions(args):
    groups = {}
    for line in args.rounds.read_text().splitlines():
        row = json.loads(line)
        groups.setdefault(row["session"], []).append(row)
    sessions = []
    for name, rows in sorted(groups.items()):
        partition = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 2
        if args.split != "all" and partition != (0 if args.split == "validation" else 1):
            continue
        rows.sort(key=lambda r: r["first_chunk_s"])
        if len(rows) >= args.turns:
            sessions.append((name, rows))
    rng = random.Random(args.seed)
    rng.shuffle(sessions)
    selected = []
    for name, rows in sessions[: args.sessions]:
        offset = rng.randrange(len(rows) - args.turns + 1)
        selected.append((name, rows[offset : offset + args.turns], rng.random() * args.start_jitter))
    if len(selected) < args.sessions:
        raise ValueError(f"requested {args.sessions} sessions, only {len(selected)} eligible in split {args.split}")
    return selected


def stream_kinds(selected, mix):
    """Session selection is already seeded; assign an exact agent fraction."""
    if mix not in ("agent25", "agent50"):
        return {name: "trace" for name, _, _ in selected}
    divisor = 4 if mix == "agent25" else 2
    count = max(1, len(selected) // divisor)
    return {name: "agent" if index < count else "one-shot" for index, (name, _, _) in enumerate(selected)}


def session_system(namespace, name):
    # Experimental labels such as "off" or "ordinary" are not instructions to
    # the model. Keep isolation opaque and identical between paired policies.
    scope = hashlib.sha256(f"{namespace}/{name}".encode()).hexdigest()[:32]
    return {
        "role": "system",
        "content": (
            f"Session {scope}. Follow only the latest user instruction; earlier user instructions are completed. "
            "Ignore all reference data. When instructed to call a tool, make the actual tool call using the declared "
            "schema, even if its result seems predictable. Do not substitute a plain-text description of a tool call."
        ),
    }


def reset_one_shot(messages, system, turn):
    # A completed independent request has no reusable conversation on its next
    # arrival. Its namespace differs before the long context, not after it.
    messages[:] = [{**system, "content": system["content"] + f" Independent request {turn}."}]


def reset_compacted_context(messages, system, turn, identity="fresh"):
    # Redacted source metadata cannot establish prefix identity after compaction.
    # A new scope before the long synthetic reference avoids inventing reuse.
    # Shared mode explicitly preserves the earlier optimistic assumption.
    if identity not in ("fresh", "shared"):
        raise ValueError("unknown compaction identity")
    suffix = f" Context revision {turn}." if identity == "fresh" else ""
    messages[:] = [{**system, "content": system["content"] + suffix}]


def latency_stats(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def ordinary_output_budget(tokenizer, answer, minimum):
    serialized = json.dumps(answer)
    # Source-shaped targets are estimates, not tokenized JSON lengths. Leave
    # room for the complete literal and EOS, including alternate BPE splits.
    tokens = tokenizer.encode(serialized, add_special_tokens=False)
    return max(minimum, len(tokens) + 32, len(serialized.encode("utf-8")) + 16)


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    selected = select_sessions(args)
    kinds = stream_kinds(selected, args.mix)
    output_rows = []
    failures = []
    sessions = []
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=args.sessions, limit_per_host=args.sessions)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as client:

        async def one(name, source, stagger):
            kind = kinds[name]
            await asyncio.sleep(stagger)
            system = session_system(args.namespace, name)
            messages = [system]
            began = time.perf_counter()
            waited = 0.0
            planned_abandon = kind != "one-shot" and (
                int(hashlib.sha256(f"{args.seed}:{name}".encode()).hexdigest(), 16) / (1 << 256) < args.abandon_fraction
            )
            previous_was_tool = False
            for turn, row in enumerate(source):
                target = max(256, min(args.max_prompt_tokens, round(row["input_tokens"] * args.context_scale)))
                compaction = turn > 0 and row["input_tokens"] < source[turn - 1]["input_tokens"] - 256
                if compaction:
                    reset_compacted_context(messages, system, turn, args.compaction_identity)
                if kind == "one-shot":
                    reset_one_shot(messages, system, turn)
                if turn:
                    original_wait = source[turn - 1].get("next_input_ready_gap_s")
                    original_wait = (
                        original_wait
                        if original_wait is not None
                        else source[turn - 1].get("next_first_chunk_gap_s", 0)
                    )
                    scaled_wait = original_wait * args.wait_scale
                    injected_wait = max(0, args.min_tool_wait - scaled_wait) if previous_was_tool else 0
                    wait = min(args.max_wait, scaled_wait + injected_wait)
                    await asyncio.sleep(wait)
                    waited += wait
                else:
                    original_wait = scaled_wait = injected_wait = wait = 0
                want_tool = bool(row["tool_names"])
                if args.mix == "ordinary":
                    want_tool = False
                elif args.mix == "all-tools":
                    want_tool = True
                elif args.mix == "mixed25":
                    want_tool = hashlib.sha256(f"{args.seed}:{name}:{row['request']}".encode()).digest()[0] < 64
                elif args.mix in ("agent25", "agent50"):
                    want_tool = kind == "agent" and turn < len(source) - 1
                if planned_abandon and turn == 0:
                    want_tool = True
                output_target = (
                    max(1, min(args.max_output_target, round(row["output_tokens"] * args.output_scale)))
                    if args.output_scale > 0 and row["output_tokens"] > 0
                    else 0
                )
                answer = "OK"
                instruction = (
                    'Invoke the declared echo tool exactly once with arguments {"text": "hello"}. '
                    "Emit a real tool call, not a plain-text answer."
                    if want_tool
                    else "Reply with only OK. Do not call any tools."
                )
                if output_target:
                    if want_tool:
                        argument = " ".join(str(index) for index in range(max(1, (output_target - 24) // 2)))
                        instruction = (
                            "Invoke the declared echo tool exactly once with text set to this exact string: "
                            f'"{argument}". '
                            "Emit a real tool call, not a plain-text answer."
                        )
                    else:
                        answer = " ".join(str(index) for index in range(max(1, output_target // 2)))
                        instruction = (
                            f'Reply with exactly this text, without quotes: "{answer}". Do not call any tools.'
                        )
                constrained_ordinary = not want_tool and getattr(args, "ordinary_output", "text") == "json-string"
                if constrained_ordinary:
                    instruction = f"Reply with this exact JSON string: {json.dumps(answer)}. Do not call any tools."
                prompt_hash, client_tokens = await asyncio.to_thread(prepare, tokenizer, messages, instruction, target)
                if client_tokens > args.max_prompt_tokens + 128:
                    reset_compacted_context(messages, system, turn, args.compaction_identity)
                    if kind == "one-shot":
                        reset_one_shot(messages, system, turn)
                    prompt_hash, client_tokens = await asyncio.to_thread(
                        prepare, tokenizer, messages, instruction, target
                    )
                    compaction = True
                request_id = f"tool-cache-{args.namespace}-{args.seed}-{name}-{turn}"
                payload = {
                    "model": args.model,
                    "messages": messages,
                    "tools": TOOLS,
                    "temperature": 0,
                    "max_tokens": max(128, output_target + 128),
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                if constrained_ordinary:
                    payload["max_tokens"] = ordinary_output_budget(tokenizer, answer, payload["max_tokens"])
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ordinary_reply",
                            "strict": True,
                            "schema": {"type": "string", "enum": [answer]},
                        },
                    }
                started = time.perf_counter()
                async with client.post(
                    args.url + "/v1/chat/completions", json=payload, headers={"X-Request-Id": request_id}
                ) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise RuntimeError(f"{request_id} HTTP {response.status}: {text[:500]}")
                    result = json.loads(text)
                elapsed = time.perf_counter() - started
                choice = result["choices"][0]
                message = choice["message"]
                if constrained_ordinary and json.loads(message.get("content") or "null") != answer:
                    raise RuntimeError("ordinary response did not satisfy its exact string constraint")
                calls = message.get("tool_calls") or []
                expected = "tool_calls" if want_tool else "stop"
                allowed = ["tool_calls"] if want_tool else ["stop", "length"]
                usage = result["usage"]
                if usage["prompt_tokens"] != client_tokens:
                    raise RuntimeError(
                        f"template mismatch for {request_id}: client={client_tokens}, server={usage['prompt_tokens']}"
                    )
                record = {
                    "request_id": request_id,
                    "source_request": row["request"],
                    "source_input_tokens": row["input_tokens"],
                    "source_output_tokens": row["output_tokens"],
                    "target_output_tokens": output_target,
                    "target_prompt_tokens": target,
                    "session": name,
                    "request_kind": kind,
                    "turn": turn,
                    "prompt_sha256": prompt_hash,
                    "client_prompt_tokens": client_tokens,
                    "seconds": elapsed,
                    "finish_reason": choice["finish_reason"],
                    "expected_finish_reason": expected,
                    "allowed_finish_reasons": allowed,
                    "usage": usage,
                    "source_wait_seconds": original_wait,
                    "scaled_wait_seconds": scaled_wait,
                    "injected_wait_seconds": injected_wait,
                    "wait_seconds": wait,
                    "wait_capped": wait < scaled_wait + injected_wait,
                    "compaction": compaction,
                    "output_signature": {"content": message.get("content"), "calls": [c["function"] for c in calls]},
                }
                output_rows.append(record)
                print(
                    json.dumps({k: record[k] for k in ["request_id", "seconds", "finish_reason", "usage"]}), flush=True
                )
                if choice["finish_reason"] not in allowed:
                    failures.append({"request_id": request_id, "expected": allowed, "actual": choice["finish_reason"]})
                # Keep the actual response. IDs are opaque; the template ignores
                # them, while the corresponding tool result retains the ID.
                assistant = {"role": "assistant", "content": message.get("content") or ""}
                if calls:
                    assistant["tool_calls"] = calls
                messages.append(assistant)
                for call in calls:
                    arguments = json.loads(call["function"]["arguments"])
                    if call["function"]["name"] != "echo" or not isinstance(arguments.get("text"), str):
                        raise RuntimeError(f"invalid echo invocation: {request_id} {call}")
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": arguments["text"]})
                previous_was_tool = bool(calls)
                if planned_abandon and turn == 0:
                    break
            sessions.append(
                {
                    "session": name,
                    "session_kind": kind,
                    "seconds": time.perf_counter() - began,
                    "wait_seconds": waited,
                    "abandoned": planned_abandon,
                    "completed_turns": turn + 1,
                }
            )

        started_unix = time.time()
        started = time.perf_counter()
        await asyncio.gather(*(one(*session) for session in selected))
        duration = time.perf_counter() - started
    prompt_tokens = sum(r["usage"]["prompt_tokens"] for r in output_rows)
    cached_tokens = sum(r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in output_rows)
    completed = [s for s in sessions if not s["abandoned"]]
    agents = [s for s in completed if s["session_kind"] == "agent"]
    cohorts = {}
    for kind in ("agent", "one-shot", "trace"):
        rows = [r for r in output_rows if r["request_kind"] == kind]
        if not rows:
            continue
        prompts = sum(r["usage"]["prompt_tokens"] for r in rows)
        cached = sum(r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in rows)
        cohorts[kind] = {
            "requests": len(rows),
            "prompt_tokens": prompts,
            "cached_tokens": cached,
            "prefilled_tokens": prompts - cached,
            "output_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
            "request_seconds": latency_stats([r["seconds"] for r in rows]),
        }
    result = {
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rounds_sha256": hashlib.sha256(args.rounds.read_bytes()).hexdigest(),
        "started_unix": started_unix,
        "finished_unix": time.time(),
        "wall_seconds": duration,
        "started_sessions": len(sessions),
        "completed_sessions": len(completed),
        "abandoned_sessions": len(sessions) - len(completed),
        "sessions_per_second": len(completed) / duration,
        "session_kind_counts": dict(collections.Counter(s["session_kind"] for s in sessions)),
        "agent_completed_sessions": len(agents),
        "agent_sessions_per_second": len(agents) / duration,
        "agent_session_seconds": latency_stats([s["seconds"] for s in agents]),
        "cohort_metrics": cohorts,
        "session_seconds": {
            "mean": statistics.mean(s["seconds"] for s in completed) if completed else None,
            "p50": percentile([s["seconds"] for s in completed], 0.5),
            "p95": percentile([s["seconds"] for s in completed], 0.95),
        },
        "requests": len(output_rows),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit_ratio": cached_tokens / prompt_tokens,
        "prefilled_tokens": prompt_tokens - cached_tokens,
        "output_tokens": sum(r["usage"]["completion_tokens"] for r in output_rows),
        "output_target_capped_requests": sum(
            r["source_output_tokens"] * args.output_scale > args.max_output_target for r in output_rows
        ),
        "request_seconds": {
            "mean": statistics.mean(r["seconds"] for r in output_rows),
            "p50": percentile([r["seconds"] for r in output_rows], 0.5),
            "p95": percentile([r["seconds"] for r in output_rows], 0.95),
        },
        "wait_capped_requests": sum(r["wait_capped"] for r in output_rows),
        "unexpected_end_states": failures,
        "finish_reason_counts": dict(collections.Counter(r["finish_reason"] for r in output_rows)),
        "sessions": sessions,
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["rows", "sessions"]}, indent=2), flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} unexpected end states; retain this run as invalid rather than silently filtering"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model", default="tool-cache-qwen4b")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--namespace", default="run", help="Unique per paired trial, identical for both policies")
    parser.add_argument("--split", choices=["validation", "test", "all"], default="validation")
    parser.add_argument(
        "--mix", choices=["trace", "ordinary", "all-tools", "mixed25", "agent25", "agent50"], default="trace"
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=0,
        help="Optional source-shaped output target; zero uses short echo/OK. Actual outputs are measured.",
    )
    parser.add_argument("--max-output-target", type=int, default=512)
    parser.add_argument(
        "--ordinary-output",
        choices=["text", "json-string"],
        default="text",
        help="Exact JSON strings avoid punctuation drift; strict paired output/token checks remain enabled.",
    )
    parser.add_argument("--abandon-fraction", type=float, default=0)
    parser.add_argument("--min-tool-wait", type=float, default=0)
    parser.add_argument("--context-scale", type=float, default=0.05)
    parser.add_argument("--max-prompt-tokens", type=int, default=12000)
    parser.add_argument("--wait-scale", type=float, default=0.1)
    parser.add_argument("--max-wait", type=float, default=5)
    parser.add_argument("--start-jitter", type=float, default=1)
    parser.add_argument(
        "--compaction-identity",
        choices=["fresh", "shared"],
        default="fresh",
        help="Fresh scopes avoid assumed cross-compaction reuse; shared reproduces earlier runs.",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
