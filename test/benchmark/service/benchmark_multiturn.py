"""
Multi-turn dialogue benchmark for LightLLM.

For each concurrency level in --concurrency_levels, launches N concurrent
"sessions". Each session starts from a prompt of ~start_input_len tokens
(with a per-session random prefix so different sessions don't share KV
cache) and keeps issuing streaming requests turn by turn. After every
turn the model's generated text plus a dynamically sampled number of new
tokens are appended to the prompt, simulating the user's next message.
A session stops when the next prompt would exceed max_input_len, or
after max_turns turns.

Metrics aggregated per concurrency level:
  - TTFT  (Time To First Token, ms): per-turn first-byte latency
  - TPOT  (Time Per Output Token, ms): mean inter-token gap after TTFT
  - QPS   (turns / wall_time)
  - TPM   ((prompt_tokens + completion_tokens) / wall_time * 60)
  - Cache hit ratio = sum(cached_tokens) / sum(prompt_tokens) across turns

The OpenAI v1/completions streaming endpoint is used because its final
`usage` chunk carries `prompt_tokens_details.cached_tokens`, which is
how prompt-cache hit length is exposed to clients.

Example:
  python benchmark_multiturn.py \\
      --url http://127.0.0.1:8000/v1/completions \\
      --tokenizer_path /path/to/tokenizer \\
      --model_name my-model \\
      --concurrency_levels 1,4,8,16 \\
      --start_input_len 1024 \\
      --max_input_len 16384 \\
      --turn_input_increment 256 \\
      --output_len 256
"""

import argparse
import asyncio
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


def seed_all(seed: int) -> None:
    if not seed:
        seed = int(time.time())
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_tokenizer(tokenizer_name: str) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def gen_random_token_ids(tokenizer, n: int, rng: random.Random) -> List[int]:
    vocab = tokenizer.vocab_size
    return [rng.randint(0, vocab - 1) for _ in range(n)]


def decode_ids(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def gen_session_initial_prompt(
    tokenizer,
    start_input_len: int,
    session_seed: int,
) -> Tuple[str, int]:
    """Build the initial prompt for a session. The prefix is unique per
    session so that prefix-cache hits across sessions are not counted."""
    rng = random.Random(session_seed)
    ids = gen_random_token_ids(tokenizer, start_input_len, rng)
    text = decode_ids(tokenizer, ids)
    # Re-encode so that the recorded token length matches what the server
    # will tokenize. Random ids -> decode -> re-encode is not lossless.
    real_ids = tokenizer.encode(text, add_special_tokens=False)
    return text, len(real_ids)


def append_turn_input(
    tokenizer,
    prompt: str,
    generated_text: str,
    turn_input_increment: int,
    rng: random.Random,
) -> Tuple[str, int]:
    """Append the model's generated text plus a fresh random user turn
    to the prompt. Returns (new_prompt, new_prompt_token_len)."""
    if turn_input_increment > 0:
        new_ids = gen_random_token_ids(tokenizer, turn_input_increment, rng)
        new_text = decode_ids(tokenizer, new_ids)
    else:
        new_text = ""
    new_prompt = prompt + generated_text + new_text
    new_len = len(tokenizer.encode(new_prompt, add_special_tokens=False))
    return new_prompt, new_len


async def stream_one_turn(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
) -> Optional[Dict]:
    """Send one streaming completion request, return per-turn stats:
      {
        "ttft": float seconds,
        "decode_times": [float seconds, ...],  # gaps between subsequent tokens
        "prompt_tokens": int,
        "completion_tokens": int,
        "cached_tokens": int,
        "generated_text": str,
      }
    Returns None on failure."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    decode_times: List[float] = []
    generated_text_parts: List[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0

    try:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                err = await response.text()
                print(f"\n[turn failed] status={response.status} body={err[:200]}")
                return None

            async for raw in response.content:
                line = raw.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                data_str = line[len(b"data:") :].strip()
                if data_str == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue

                # Final usage-only chunk: choices == [] and usage present
                usage = chunk.get("usage")
                choices = chunk.get("choices") or []
                if usage is not None and not choices:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = details.get("cached_tokens", cached_tokens)
                    continue

                # Token-bearing chunk
                if not choices:
                    continue
                text_piece = choices[0].get("text", "")
                if text_piece == "" and choices[0].get("finish_reason") is None:
                    continue

                now = time.time()
                if first_token_time is None:
                    first_token_time = now
                else:
                    decode_times.append(now - last_token_time)
                last_token_time = now
                if text_piece:
                    generated_text_parts.append(text_piece)
    except Exception as e:
        print(f"\n[turn exception] {e}")
        return None

    if first_token_time is None:
        return None

    return {
        "ttft": first_token_time - start_time,
        "decode_times": decode_times,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "generated_text": "".join(generated_text_parts),
    }


async def run_session(
    session_id: int,
    tokenizer,
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    start_input_len: int,
    max_input_len: int,
    min_turn_input_increment: int,
    turn_input_increment: int,
    min_output_len: int,
    output_len: int,
    max_turns: int,
    base_seed: int,
    progress_state: Dict,
) -> List[Dict]:
    """Run a single multi-turn dialogue session. Returns a list of per-turn
    stat dicts (same schema as stream_one_turn output)."""
    rng = random.Random(base_seed + session_id)
    prompt, prompt_len = gen_session_initial_prompt(tokenizer, start_input_len, base_seed + session_id)

    per_turn: List[Dict] = []
    turn_idx = 0
    while turn_idx < max_turns and prompt_len < max_input_len:
        turn_output_len = rng.randint(min_output_len, output_len)
        result = await stream_one_turn(session, url, model_name, prompt, turn_output_len)
        if result is None:
            break
        per_turn.append(result)
        progress_state["finished_turns"] += 1
        print(
            f"\rconc={progress_state['concurrency']} "
            f"finished_turns={progress_state['finished_turns']} "
            f"active_sessions={progress_state['active_sessions']}",
            end="",
        )
        turn_input_len = rng.randint(min_turn_input_increment, turn_input_increment)
        prompt, prompt_len = append_turn_input(
            tokenizer,
            prompt,
            result["generated_text"],
            turn_input_len,
            rng,
        )
        turn_idx += 1

    progress_state["active_sessions"] -= 1
    return per_turn


async def run_concurrency_level(
    concurrency: int,
    tokenizer,
    url: str,
    model_name: str,
    start_input_len: int,
    max_input_len: int,
    min_turn_input_increment: int,
    turn_input_increment: int,
    min_output_len: int,
    output_len: int,
    max_turns: int,
    base_seed: int,
    request_timeout_s: int,
) -> Dict:
    """Run one concurrency level. Returns the aggregated stats dict."""
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 32))
    progress_state = {
        "concurrency": concurrency,
        "finished_turns": 0,
        "active_sessions": concurrency,
    }

    wall_start = time.time()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                run_session(
                    sid,
                    tokenizer,
                    session,
                    url,
                    model_name,
                    start_input_len,
                    max_input_len,
                    min_turn_input_increment,
                    turn_input_increment,
                    min_output_len,
                    output_len,
                    max_turns,
                    base_seed,
                    progress_state,
                )
            )
            for sid in range(concurrency)
        ]
        session_results = await asyncio.gather(*tasks)
    wall_end = time.time()
    wall_time = max(wall_end - wall_start, 1e-9)
    print()  # newline after progress bar

    all_turns: List[Dict] = [t for s in session_results for t in s]
    return summarize(
        concurrency=concurrency,
        turns=all_turns,
        wall_time=wall_time,
        num_sessions=concurrency,
        max_turns=max_turns,
    )


def summarize(
    concurrency: int,
    turns: List[Dict],
    wall_time: float,
    num_sessions: int,
    max_turns: int,
) -> Dict:
    percentiles = [50, 75, 90, 95, 99]
    out: Dict = {
        "concurrency": concurrency,
        "num_sessions": num_sessions,
        "max_turns_per_session": max_turns,
        "total_turns": len(turns),
        "wall_time_s": round(wall_time, 4),
    }

    if not turns:
        out["error"] = "no successful turns"
        return out

    ttfts_ms = [t["ttft"] * 1000.0 for t in turns]
    # TPOT per turn = mean of decode_times (skip turns with <2 tokens)
    tpots_ms: List[float] = []
    for t in turns:
        if t["decode_times"]:
            tpots_ms.append(1000.0 * sum(t["decode_times"]) / len(t["decode_times"]))
    prompt_tokens = sum(t["prompt_tokens"] for t in turns)
    completion_tokens = sum(t["completion_tokens"] for t in turns)
    cached_tokens = sum(t["cached_tokens"] for t in turns)
    total_tokens = prompt_tokens + completion_tokens

    qps = len(turns) / wall_time
    tpm_total = total_tokens / wall_time * 60.0
    tpm_prompt = prompt_tokens / wall_time * 60.0
    tpm_completion = completion_tokens / wall_time * 60.0
    cache_hit_ratio = cached_tokens / prompt_tokens if prompt_tokens else 0.0

    out["QPS"] = round(qps, 4)
    out["TPM_total"] = round(tpm_total, 2)
    out["TPM_prompt"] = round(tpm_prompt, 2)
    out["TPM_completion"] = round(tpm_completion, 2)
    out["total_prompt_tokens"] = prompt_tokens
    out["total_completion_tokens"] = completion_tokens
    out["total_cached_prompt_tokens"] = cached_tokens
    out["cache_hit_ratio"] = round(cache_hit_ratio, 6)
    out["avg_prompt_tokens_per_turn"] = round(prompt_tokens / len(turns), 2)
    out["avg_completion_tokens_per_turn"] = round(completion_tokens / len(turns), 2)

    ttft_pcts = np.percentile(ttfts_ms, percentiles)
    out["TTFT_ms"] = {"mean": round(float(np.mean(ttfts_ms)), 3)}
    for p, v in zip(percentiles, ttft_pcts):
        out["TTFT_ms"][f"P{p}"] = round(float(v), 3)

    if tpots_ms:
        tpot_pcts = np.percentile(tpots_ms, percentiles)
        out["TPOT_ms"] = {"mean": round(float(np.mean(tpots_ms)), 3)}
        for p, v in zip(percentiles, tpot_pcts):
            out["TPOT_ms"][f"P{p}"] = round(float(v), 3)
    else:
        out["TPOT_ms"] = {"mean": None, "note": "all turns produced <2 tokens"}

    return out


def print_summary(summary: Dict) -> None:
    print("=" * 80)
    print(
        f"Concurrency = {summary['concurrency']}  sessions = {summary['num_sessions']}  "
        f"total_turns = {summary['total_turns']}  wall_time = {summary['wall_time_s']}s"
    )
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
        return
    print(f"  QPS                : {summary['QPS']}")
    print(f"  TPM (total)        : {summary['TPM_total']}")
    print(f"  TPM (prompt)       : {summary['TPM_prompt']}")
    print(f"  TPM (completion)   : {summary['TPM_completion']}")
    print(
        f"  Cache hit ratio    : {summary['cache_hit_ratio'] * 100:.2f}%  "
        f"({summary['total_cached_prompt_tokens']} / {summary['total_prompt_tokens']})"
    )
    print(f"  Avg prompt tokens  : {summary['avg_prompt_tokens_per_turn']}")
    print(f"  Avg output tokens  : {summary['avg_completion_tokens_per_turn']}")
    ttft = summary["TTFT_ms"]
    tpot = summary["TPOT_ms"]
    print(
        f"  TTFT ms  mean={ttft['mean']}  P50={ttft.get('P50')}  P90={ttft.get('P90')}  "
        f"P95={ttft.get('P95')}  P99={ttft.get('P99')}"
    )
    if tpot.get("mean") is None:
        print(f"  TPOT ms  (n/a: {tpot.get('note')})")
    else:
        print(
            f"  TPOT ms  mean={tpot['mean']}  P50={tpot.get('P50')}  P90={tpot.get('P90')}  "
            f"P95={tpot.get('P95')}  P99={tpot.get('P99')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8088/v1/completions",
        help="Streaming OpenAI completion endpoint. The benchmark relies on "
        "the final SSE `usage` chunk to obtain cached_tokens.",
    )
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name passed to the server. Defaults to --tokenizer_path.",
    )
    parser.add_argument(
        "--concurrency_levels",
        type=str,
        default="1,4,8,16,32,64,128,256",
        help="Comma-separated list of concurrency levels to sweep.",
    )
    parser.add_argument(
        "--start_input_len", type=int, default=32768, help="Initial prompt length in tokens per session."
    )
    parser.add_argument(
        "--max_input_len", type=int, default=163840, help="Stop a session when its prompt exceeds this length."
    )
    parser.add_argument(
        "--turn_input_increment",
        type=int,
        default=2048,
        help="Maximum new 'user' tokens sampled after each turn, on top " "of the model's generated text.",
    )
    parser.add_argument(
        "--min_turn_input_increment", type=int, default=512, help="Minimum new 'user' tokens sampled after each turn."
    )
    parser.add_argument("--output_len", type=int, default=512, help="Maximum max_new_tokens sampled per turn.")
    parser.add_argument("--min_output_len", type=int, default=128, help="Minimum max_new_tokens sampled per turn.")
    parser.add_argument(
        "--max_turns",
        type=int,
        default=64,
        help="Hard cap on turns per session. The session also stops once " "prompt length reaches --max_input_len.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request_timeout_s", type=int, default=3600)
    parser.add_argument(
        "--dump_file",
        type=str,
        default="",
        help="If set, append the per-concurrency summary dict to this JSON file. "
        "If the file already exists and is non-empty, it is read and printed.",
    )

    args = parser.parse_args()

    if args.min_output_len < 1:
        raise ValueError("--min_output_len must be >= 1")
    if args.min_output_len > args.output_len:
        raise ValueError("--min_output_len must be <= --output_len")
    if args.min_turn_input_increment < 0:
        raise ValueError("--min_turn_input_increment must be >= 0")
    if args.min_turn_input_increment > args.turn_input_increment:
        raise ValueError("--min_turn_input_increment must be <= --turn_input_increment")

    if args.dump_file and os.path.exists(args.dump_file) and os.path.getsize(args.dump_file) > 0:
        with open(args.dump_file, "r") as f:
            print(json.dumps(json.load(f), indent=4))
        return

    seed_all(args.seed)
    model_name = args.model_name or args.tokenizer_path
    tokenizer = get_tokenizer(args.tokenizer_path)
    concurrency_levels = [int(x) for x in args.concurrency_levels.split(",") if x.strip()]

    print(f"URL                : {args.url}")
    print(f"Model              : {model_name}")
    print(f"Concurrency levels : {concurrency_levels}")
    print(f"start_input_len    : {args.start_input_len}")
    print(f"max_input_len      : {args.max_input_len}")
    print(f"min_turn_input_increment: {args.min_turn_input_increment}")
    print(f"turn_input_increment: {args.turn_input_increment}")
    print(f"min_output_len     : {args.min_output_len}")
    print(f"output_len         : {args.output_len}")
    print(f"max_turns          : {args.max_turns}")

    all_summaries: List[Dict] = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        for concurrency in concurrency_levels:
            summary = loop.run_until_complete(
                run_concurrency_level(
                    concurrency=concurrency,
                    tokenizer=tokenizer,
                    url=args.url,
                    model_name=model_name,
                    start_input_len=args.start_input_len,
                    max_input_len=args.max_input_len,
                    min_turn_input_increment=args.min_turn_input_increment,
                    turn_input_increment=args.turn_input_increment,
                    min_output_len=args.min_output_len,
                    output_len=args.output_len,
                    max_turns=args.max_turns,
                    base_seed=args.seed,
                    request_timeout_s=args.request_timeout_s,
                )
            )
            print_summary(summary)
            all_summaries.append(summary)
    finally:
        loop.close()

    dump = {
        "config": {
            "url": args.url,
            "model_name": model_name,
            "tokenizer_path": args.tokenizer_path,
            "concurrency_levels": concurrency_levels,
            "start_input_len": args.start_input_len,
            "max_input_len": args.max_input_len,
            "min_turn_input_increment": args.min_turn_input_increment,
            "turn_input_increment": args.turn_input_increment,
            "min_output_len": args.min_output_len,
            "output_len": args.output_len,
            "max_turns": args.max_turns,
            "seed": args.seed,
        },
        "results": all_summaries,
    }
    print("\n" + "=" * 80)
    print(json.dumps(dump, indent=4, ensure_ascii=False))
    if args.dump_file:
        with open(args.dump_file, "w") as f:
            json.dump(dump, f, indent=4, ensure_ascii=False)
        print(f"\nResults dumped to {args.dump_file}")


if __name__ == "__main__":
    main()
