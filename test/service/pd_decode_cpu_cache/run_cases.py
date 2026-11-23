"""Exercise real PD services: 26 base scenarios, or 222 with --extended.

P CPU cache profiles add a GPU-eviction / CPU-restore scenario.

Checks actual HTTP inference, greedy token equality, CPU reuse, cancellation,
page boundaries and final shared-memory reference counts. No mock nodes are used.
Model logprob differences are recorded separately from the token equality checks.
"""
import argparse
import concurrent.futures
import json
import re
import struct
import sys
import time
import traceback
from pathlib import Path
import httpx
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lightllm.utils.kv_cache_utils import compute_token_list_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase")
parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pd_decode_cpu_cache_service")
parser.add_argument("--only", choices=["all", "boundary"], default="all")
parser.add_argument(
    "--extended", action="store_true", help="Add >190 boundary, prefix, output, stream and queue scenarios"
)
parser.add_argument("--tag-prefix", default="", help="Isolate prompt hashes on reruns")
parser.add_argument("--case-filter", help="Run only case names matching this regular expression")
parser.add_argument("--result-label", default="", help="Preserve earlier results using a filename prefix")
args = parser.parse_args()
OUT = args.output_dir.resolve()
RUN = OUT / args.phase
manifest = json.loads((RUN / "processes.json").read_text())
model = manifest[0]["command"][manifest[0]["command"].index("--model_dir") + 1]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
host = manifest[0]["command"][manifest[0]["command"].index("--host") + 1]
port = int(manifest[0]["command"][manifest[0]["command"].index("--port") + 1])
BASE = f"http://{host}:{port}"
REF = f"http://{host}:{port+3}"
client = httpx.Client(trust_env=False, timeout=httpx.Timeout(120, connect=5))
results = []
pf = args.phase.endswith("_pf")
hybrid = not args.phase.startswith("dense")
log = RUN / "decode.log"
common = dict(
    do_sample=False,
    top_k=1,
    top_p=1.0,
    temperature=1.0,
    max_new_tokens=8,
    ignore_eos=True,
    return_details=True,
    skip_special_tokens=False,
    disable_prompt_cache=False,
)


def prompt(n, tag):
    if n == 1:
        return "Hello"
    if n < 64:
        text = "x" + " x" * (n - 1)
        assert len(tok.encode(text, add_special_tokens=False)) == n
        return text
    header = f"Scenario {args.tag_prefix}{tag}. Read the notes carefully."
    suffix = "\nAlice has 3 apples, Bob has 2. Their total number of apples is"
    fixed = len(tok.encode(header + suffix, add_special_tokens=False))
    assert fixed <= n, (fixed, n)
    padding = n - fixed
    for _ in range(8):
        text = header + (" x" * padding) + suffix
        actual = len(tok.encode(text, add_special_tokens=False))
        if actual == n:
            break
        padding += n - actual
    assert len(tok.encode(text, add_special_tokens=False)) == n, (n, actual)
    return text


def length(text):
    return len(tok.encode(text, add_special_tokens=False))


def request(ids, endpoint=BASE, params=None):
    p = {**common, **(params or {})}
    if endpoint == REF:
        p["disable_prompt_cache"] = True
    t = time.monotonic()
    r = client.post(endpoint + "/generate", json={"inputs": ids, "parameters": p})
    assert r.status_code == 200, (r.status_code, r.text[:1200])
    d = r.json()
    assert "tokens" in d, d
    ids_out = [x["id"] for x in d["tokens"]]
    assert len(ids_out) == p["max_new_tokens"], (len(ids_out), p["max_new_tokens"], d)
    assert d["prompt_tokens"] == length(ids), (d["prompt_tokens"], length(ids))
    return dict(token_ids=ids_out, elapsed=time.monotonic() - t, response=d)


def offset():
    return log.stat().st_size


def evidence(start):
    time.sleep(0.12)
    with log.open() as f:
        f.seek(start)
        text = f.read()
    cpus = [int(x) for x in re.findall(r"cpu_prompt_cache_len:(\d+)", text)]
    gpu = [int(x) for x in re.findall(r"(?<!cpu_)(?<!disk_)prompt_cache_len:(\d+)", text)]
    requests = re.findall(r"group_request_id[:=]\s*(\d+)", text)
    return dict(
        cpu_cache_tokens=cpus,
        gpu_cache_tokens=gpu,
        group_request_ids=list(dict.fromkeys(requests)),
        log_excerpt=text[-16000:],
    )


def save():
    (RUN / f"{args.result_label}results.json").write_text(
        json.dumps(
            dict(
                phase=args.phase, model=model, case_filter=args.case_filter, tag_prefix=args.tag_prefix, results=results
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def case(name, fn):
    if args.case_filter and not re.search(args.case_filter, name):
        return
    t = time.monotonic()
    r = dict(name=name, status="RUNNING")
    try:
        r.update(fn() or {})
        r["status"] = "PASS"
    except Exception as e:
        r.update(status="FAIL", error=str(e), traceback=traceback.format_exc())
    r["elapsed"] = time.monotonic() - t
    results.append(r)
    save()
    print(
        json.dumps(
            {k: v for k, v in r.items() if k not in ["response", "log_excerpt", "traceback"]}, ensure_ascii=False
        ),
        flush=True,
    )
    return r


def compare(ids, params=None, expect_cpu=None):
    start = offset()
    actual = request(ids, params=params)
    reference = request(ids, endpoint=REF, params=params)
    ev = evidence(start)
    deltas = [
        abs(a["logprob"] - b["logprob"]) for a, b in zip(actual["response"]["tokens"], reference["response"]["tokens"])
    ]
    payload = dict(
        prompt_tokens=length(ids),
        actual_ids=actual["token_ids"],
        reference_ids=reference["token_ids"],
        max_logprob_delta=max(deltas, default=0),
        **ev,
    )
    assert actual["token_ids"] == reference["token_ids"], payload
    if expect_cpu is not None:
        assert max(ev["cpu_cache_tokens"], default=0) == expect_cpu, payload
    return payload


def wait_cached(text):
    token_ids = tok.encode(text, add_special_tokens=False)
    hashes = compute_token_list_hash(token_ids, 256)
    if hybrid and hashes and len(hashes) * 256 + 1 == len(token_ids):
        hashes = hashes[:-1]
    if not hashes:
        return
    name = re.findall(r"create lock shm (\w+)_cpu_kv_cache_page_items", log.read_text())[0]
    page_file = Path("/dev/shm") / (name + "_cpu_kv_cache_page_items")
    deadline = time.monotonic() + 10
    while True:
        rows = list(struct.iter_unpack("<iiiiiQQ", page_file.read_bytes()))[:-2]
        ready = {r[5] + (r[6] << 64) for r in rows if r[3] == 2 and r[4] == 0}
        if all(h in ready for h in hashes):
            return
        assert time.monotonic() < deadline, dict(
            error="CPU offload did not complete", needed=len(hashes), ready=len(ready)
        )
        time.sleep(0.02)


def warm(n, tag, params=None):
    ids = prompt(n, tag)
    # Prefill-first 单 token 请求在 P 上直接结束，不会选择 D；预热必须真正经过 D。
    request(ids)
    wait_cached(ids)
    expected = ((n - (2 if hybrid else 1)) // 256) * 256 if n >= 256 else 0
    prefill_only = pf and (params or {}).get("max_new_tokens") == 1
    return dict(
        decode_bypassed=prefill_only,
        **compare(ids, params=params, expect_cpu=0 if prefill_only else max(0, expected)),
    )


case(
    "01_health_and_node_registration",
    lambda: dict(
        health=client.get(BASE + "/health").raise_for_status().json(),
        baseline=client.get(REF + "/health").raise_for_status().json(),
    ),
)
case("02_one_token_prompt", lambda: compare(prompt(1, "single"), expect_cpu=0))
case("03_below_cpu_threshold_127", lambda: warm(127, "below127"))
case("04_page_minus_one_255", lambda: warm(255, "below255"))
case("05_exact_page_256", lambda: warm(256, "exact256"))
case("06_full_match_boundary_257", lambda: warm(257, "full257"))
case("07_full_match_plus_one_258", lambda: warm(258, "full258"))
case("08_two_pages_minus_one_511", lambda: warm(511, "minus511"))
case("09_exact_two_pages_512", lambda: warm(512, "exact512"))
case("10_two_page_full_match_513", lambda: warm(513, "full513"))
case("11_two_page_plus_one_514", lambda: warm(514, "full514"))
case("12_multi_pd_pages_1025", lambda: warm(1025, "multi1025"))


def partial():
    ids = prompt(1025, "branching")
    request(ids)
    wait_cached(ids)
    changed = tok.encode(ids, add_special_tokens=False)
    changed[600] = tok.encode("Different", add_special_tokens=False)[0]
    return compare(tok.decode(changed), expect_cpu=512)


case("13_shared_prefix_divergent_tail", partial)
case("14_distinct_prefix_no_false_hit", lambda: compare(prompt(769, "totally_new_14"), expect_cpu=0))
case(
    "15_single_generated_token", lambda: compare(prompt(769, "one-output"), params={"max_new_tokens": 1}, expect_cpu=0)
)
case(
    "16_long_decode_64_tokens", lambda: compare(prompt(514, "long-output"), params={"max_new_tokens": 64}, expect_cpu=0)
)

if args.only != "boundary":

    def run_concurrent(same):
        ids_list = [prompt(769, "concurrent_same" if same else f"concurrent_{i}") for i in range(4)]
        refs = [request(ids, endpoint=REF)["token_ids"] for ids in ids_list]
        start = offset()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outs = list(pool.map(request, ids_list))
        assert [x["token_ids"] for x in outs] == refs, outs
        return dict(count=len(outs), token_ids=[x["token_ids"] for x in outs], **evidence(start))

    case("17_concurrent_identical_prompts", lambda: run_concurrent(True))
    case("18_concurrent_distinct_prompts", lambda: run_concurrent(False))

    def cancel():
        ids = prompt(1025, "cancel-stream")
        with client.stream(
            "POST", BASE + "/generate_stream", json={"inputs": ids, "parameters": {**common, "max_new_tokens": 1024}}
        ) as r:
            assert r.status_code == 200, r.read()
            first = next(r.iter_lines())
            assert first, first
        time.sleep(1)
        return dict(first_chunk=first, health=client.get(BASE + "/health").raise_for_status().json())

    case("19_cancel_active_stream", cancel)
    case("20_recovery_after_cancellation", lambda: compare(prompt(514, "after-cancel")))

    def invalid(params, ids):
        r = client.post(BASE + "/generate", json={"inputs": ids, "parameters": {**common, **params}})
        assert r.status_code in (400, 413), (r.status_code, r.text[:1000])
        return dict(status_code=r.status_code, body=r.text[:800])

    case("21_invalid_zero_output_budget", lambda: invalid({"max_new_tokens": 0}, "Hello"))
    case("22_input_exceeds_context_limit", lambda: invalid({}, prompt(3100, "too_long")))

    def streaming():
        ids = prompt(514, "stream-complete")
        lines = []
        with client.stream("POST", BASE + "/generate_stream", json={"inputs": ids, "parameters": common}) as r:
            assert r.status_code == 200, r.read()
            for line in r.iter_lines():
                if line.startswith("data:"):
                    lines.append(json.loads(line[5:]))
        assert len(lines) == 8, lines
        return dict(chunks=len(lines), stream_token_ids=[x.get("token", {}).get("id") for x in lines])

    case("23_stream_finishes_without_duplicate_tokens", streaming)
    case("24_max_context_near_limit", lambda: compare(prompt(3000, "near-limit"), params={"max_new_tokens": 8}))

    if args.extended:
        boundary_lengths = [
            2,
            3,
            4,
            5,
            7,
            8,
            15,
            16,
            31,
            32,
            63,
            64,
            65,
            126,
            128,
            129,
            130,
            254,
            259,
            383,
            384,
            385,
            386,
            510,
            515,
            767,
            768,
            769,
            770,
            1023,
            1024,
            1026,
            1279,
            1280,
            1281,
            1282,
            1535,
            1536,
            1537,
            1538,
            2049,
            2561,
            3063,
        ]
        for n in boundary_lengths:
            case(f"E_warm_boundary_N{n}", lambda n=n: warm(n, f"extended-boundary-{n}"))

        for page in range(1, 12):
            for delta in (-1, 0, 1, 2):
                n = page * 256 + delta
                case(
                    f"E_cold_page{page}_offset{delta:+d}_N{n}",
                    lambda n=n: compare(prompt(n, f"extended-cold-{n}"), expect_cpu=0),
                )

        def changed_prefix(n, position):
            original = prompt(n, f"extended-prefix-{n}-{position}")
            request(original)
            wait_cached(original)
            original_ids = tok.encode(original, add_special_tokens=False)
            mutated_ids = list(original_ids)
            replacement = tok.encode(" y", add_special_tokens=False)
            assert len(replacement) == 1
            assert mutated_ids[position] != replacement[0]
            mutated_ids[position] = replacement[0]
            changed_text = tok.decode(mutated_ids)
            actual_ids = tok.encode(changed_text, add_special_tokens=False)
            common_prefix = 0
            for a, b in zip(original_ids, actual_ids):
                if a != b:
                    break
                common_prefix += 1
            expected = (common_prefix // 256) * 256
            return dict(
                mutation_position=position,
                common_prefix_tokens=common_prefix,
                **compare(changed_text, expect_cpu=expected),
            )

        for n in (1025, 1538, 2049, 2818):
            for position in (128, 255, 256, 257, 383, 511, 512, 513, 767, 768, 769):
                case(f"E_partial_N{n}_mutation{position}", lambda n=n, p=position: changed_prefix(n, p))

        for n in (257, 258, 513, 514, 1025):
            for output in (1, 2, 3, 4, 7, 16, 32, 63):
                case(
                    f"E_warm_output_N{n}_O{output}",
                    lambda n=n, output=output: warm(n, f"extended-output-{n}-{output}", {"max_new_tokens": output}),
                )

        def stream_compare(n, output):
            text = prompt(n, f"extended-stream-{n}-{output}")
            request(text)
            wait_cached(text)
            chunks = []
            with client.stream(
                "POST",
                BASE + "/generate_stream",
                json={"inputs": text, "parameters": {**common, "max_new_tokens": output}},
            ) as response:
                assert response.status_code == 200, response.read()
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        chunks.append(json.loads(line[5:]))
            ids = [chunk["token"]["id"] for chunk in chunks]
            ref_ids = request(text, endpoint=REF, params={"max_new_tokens": output})["token_ids"]
            assert ids == ref_ids, dict(actual_ids=ids, reference_ids=ref_ids, chunks=chunks)
            assert len(ids) == output
            return dict(prompt_tokens=n, output_tokens=output, actual_ids=ids, reference_ids=ref_ids)

        for n in (257, 258, 513, 514):
            for output in (1, 2, 16):
                case(f"E_stream_N{n}_O{output}", lambda n=n, output=output: stream_compare(n, output))

        def queue_pressure(count, same):
            lengths = (257, 258, 513, 514, 1025, 1538, 2049, 2818)
            prompts = [
                prompt(1025 if same else lengths[i], f"extended-queue-{count}-" + ("same" if same else str(i)))
                for i in range(count)
            ]
            refs = [request(text, endpoint=REF)["token_ids"] for text in prompts]
            for text in dict.fromkeys(prompts):
                request(text)
                wait_cached(text)
            start = offset()
            with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                outputs = list(pool.map(request, prompts))
            assert [out["token_ids"] for out in outputs] == refs
            return dict(concurrent_requests=count, same_prompt=same, **evidence(start))

        for count in (1, 2, 4, 8):
            for same in (False, True):
                case(f"E_queue_concurrency{count}_same{same}", lambda c=count, s=same: queue_pressure(c, s))

        for name, parameters in (
            ("negative_output", {"max_new_tokens": -1}),
            ("negative_temperature", {"temperature": -1}),
            ("top_p_above_one", {"top_p": 1.1}),
            ("negative_top_k", {"top_k": -2}),
        ):
            case(f"E_invalid_{name}", lambda parameters=parameters: invalid({**parameters, "do_sample": True}, "Hello"))

        def clipped_output_budget():
            text = prompt(3063, "extended-budget-clipped")
            response = client.post(
                BASE + "/generate", json={"inputs": text, "parameters": {**common, "max_new_tokens": 4096}}
            )
            assert response.status_code == 200, response.text[:1200]
            data = response.json()
            actual_ids = [token["id"] for token in data["tokens"]]
            expected_output = 3072 - length(text)
            assert len(actual_ids) == expected_output, (len(actual_ids), expected_output)
            ref_ids = request(text, endpoint=REF, params={"max_new_tokens": expected_output})["token_ids"]
            assert actual_ids == ref_ids, dict(actual_ids=actual_ids, reference_ids=ref_ids)
            return dict(
                prompt_tokens=length(text),
                requested_output=4096,
                actual_output=len(actual_ids),
                actual_ids=actual_ids,
                reference_ids=ref_ids,
            )

        case("E_output_budget_clipped_to_context", clipped_output_budget)

    case("25_health_after_cache_pressure", lambda: dict(health=client.get(BASE + "/health").raise_for_status().json()))

    def audit_node_references(node_log):
        text = node_log.read_text()
        matches = re.findall(r"create lock shm (\w+)_cpu_kv_cache_page_items", text)
        assert matches, "cannot locate cache service namespace"
        name = matches[0]
        page_file = Path("/dev/shm") / (name + "_cpu_kv_cache_page_items")
        deadline = time.monotonic() + 10
        while True:
            data = page_file.read_bytes()
            rows = list(struct.iter_unpack("<iiiiiQQ", data))[:-2]
            refs = [row[4] for row in rows]
            statuses = [row[3] for row in rows]
            if not any(refs):
                break
            assert time.monotonic() < deadline, dict(refs=refs, statuses=statuses)
            time.sleep(0.1)
        assert sum(x == 2 for x in statuses) > 0, statuses
        counters = {
            p.name: struct.unpack("<q", p.read_bytes())[0]
            for p in Path("/dev/shm").glob(name + "_kv_available_tokens_*")
        }
        return dict(
            namespace=name,
            cpu_pages=len(rows),
            ready_pages=sum(x == 2 for x in statuses),
            all_cpu_refs_zero=True,
            gpu_free_tokens=counters,
        )

    def audit_references():
        audit = audit_node_references(log)
        if "pcpu" in args.phase:
            audit["prefill_cache"] = audit_node_references(RUN / "prefill.log")
        return audit

    if "pcpu" in args.phase:

        def prefill_cpu_round_trip():
            text = prompt(1025, "prefill-cpu-round-trip")
            request(text)
            wait_cached(text)
            # Hybrid 依赖 GPU radix 管理状态页，不能禁用它。用不同前缀耗尽大页池，淘汰源前缀。
            for index in range(20):
                request(prompt(513, f"prefill-cpu-eviction-{index}"), params={"max_new_tokens": 2})
            p_log = RUN / "prefill.log"
            p_offset = p_log.stat().st_size
            result = compare(text)
            time.sleep(0.2)
            with p_log.open() as handle:
                handle.seek(p_offset)
                tail = handle.read()
            cpu = [int(x) for x in re.findall(r"cpu_prompt_cache_len:(\d+)", tail)]
            gpu = [int(x) for x in re.findall(r"(?<!cpu_)(?<!disk_)prompt_cache_len:(\d+)", tail)]
            assert max(cpu, default=0) == 1024, dict(cpu=cpu, gpu=gpu, log_excerpt=tail[-6000:])
            assert max(gpu, default=0) == 0, dict(cpu=cpu, gpu=gpu)
            return dict(prefill_cpu_tokens=cpu, prefill_gpu_tokens=gpu, **result)

        case("27_prefill_cpu_restore_after_gpu_eviction", prefill_cpu_round_trip)

    case("26_cpu_page_references_released", audit_references)

summary = dict(
    phase=args.phase,
    passed=sum(r["status"] == "PASS" for r in results),
    failed=sum(r["status"] == "FAIL" for r in results),
    total=len(results),
)
(RUN / f"{args.result_label}summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
sys_exit = 1 if summary["failed"] else 0
raise SystemExit(sys_exit)
