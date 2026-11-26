"""Independent PR review: three PD variants, varied correctness cases and paired benchmarks.

Artifacts, including exact commands and all failures, are written beneath --output-dir.
This runner never patches production code and never stops unrelated processes.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ("current_on", "current_off", "origin_off")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def serve(args):
    out = args.output_dir / args.profile
    out.mkdir(parents=True, exist_ok=True)
    if (out / "processes.json").exists():
        raise RuntimeError("Use a fresh output directory; previous evidence must be retained")
    host = socket.gethostbyname(socket.gethostname())
    preferred = args.port
    # Clients read endpoints.json; relocate the whole HTTP/NCCL port set together.
    # Existing services are never stopped to obtain a port.
    for candidate in [preferred, *range(19000, 29000, 37)]:
        ports = [candidate + 100 * i + j for i in range(3) for j in (0, 1, 2, 6001, 6002)]
        ports += [candidate + 300, candidate + 6300]
        try:
            for port in ports:
                with socket.socket() as sock:
                    sock.bind(("", port))
        except OSError:
            continue
        args.port = candidate
        break
    else:
        raise RuntimeError("No unoccupied HTTP/NCCL port set found")
    if args.port != preferred:
        print(f"Port set {preferred} occupied; using {args.port}", flush=True)
    model = args.model_dir
    common = [
        "--model_dir",
        model,
        "--model_name",
        "deep-review",
        "--host",
        host,
        "--max_req_total_len",
        "8192",
        "--max_total_token_num",
        "32768",
        "--running_max_req_size",
        "16",
        "--batch_max_tokens",
        "2048",
        "--chunked_prefill_size",
        "512",
        "--graph_max_batch_size",
        "16",
        "--disable_vision",
        "--disable_audio",
        "--output_constraint_mode",
        "none",
        "--httpserver_workers",
        "1",
        "--page_size",
        "4",
        "--linear_att_hash_page_size",
        "128",
        "--linear_att_page_block_num",
        "4",
        "--linear_att_cache_size",
        "64",
        "--disable_cudagraph",
        "--att_backend",
        "gqa.prefill=triton",
        "gdn.prefill=triton",
        "gqa.decode=triton",
        "gdn.decode=triton",
    ]
    if "graph" in args.profile or "mtp" in args.profile:
        common.remove("--disable_cudagraph")
    if "mtp" in args.profile:
        common += ["--mtp_mode", "vanilla_with_att", "--mtp_step", "1"]
    if "working_set" in args.profile:
        common[common.index("--max_total_token_num") + 1] = "8192"
        common[common.index("--max_req_total_len") + 1] = "6144"
    devices = [int(x) for x in args.gpu_order.split(",")]
    assert sorted(devices) == list(range(7)), "GPU order must be a permutation of 0..6"
    specs = []
    for i, variant in enumerate(VARIANTS):
        repo = args.origin_root if variant == "origin_off" else ROOT
        base = args.port + 100 * i
        master = [
            "--model_dir",
            model,
            "--model_name",
            "deep-review",
            "--host",
            host,
            "--max_req_total_len",
            "8192",
            "--running_max_req_size",
            "16",
            "--disable_vision",
            "--disable_audio",
            "--output_constraint_mode",
            "none",
            "--httpserver_workers",
            "1",
            "--run_mode",
            "pd_master",
            "--port",
            str(base),
            "--pd_master_mode",
            "1p1d",
            "--select_p_d_node_strategy",
            "round_robin",
        ]
        if args.profile.endswith("pf"):
            master += ["--pd_prefill_first"]
        if "working_set" in args.profile:
            master[master.index("--max_req_total_len") + 1] = "6144"
        specs.append((variant, "master", "", repo, master))
        for role, device, port in (("prefill", i * 2, base + 1), ("decode", i * 2 + 1, base + 2)):
            cmd = common + [
                "--run_mode",
                role,
                "--port",
                str(port),
                "--nccl_port",
                str(port + 6000),
                "--pd_trans_mode",
                "nccl",
                "--pd_kv_page_size",
                "512",
                "--pd_master_ip",
                host,
                "--pd_master_port",
                str(base),
            ]
            if variant == "current_on" and role == "decode":
                cmd += [
                    "--enable_cpu_cache",
                    "--cpu_cache_storage_size",
                    "4" if "working_set" in args.profile else "2",
                    "--cpu_cache_token_page_size",
                    "256",
                    "--cache_placement_strategy",
                    "legacy",
                ]
            if (
                args.profile.startswith("dense")
                and role == "decode"
                and not ("gpu_reference" in args.profile and variant == "origin_off")
                and "working_set" not in args.profile
            ):
                cmd += ["--disable_dynamic_prompt_cache"]
            visible_devices = str(devices[device])
            if "dp2" in args.profile and variant == "current_on" and role == "decode":
                cmd += ["--tp", "2", "--dp", "2"]
                visible_devices += "," + str(devices[6])
            specs.append((variant, role, visible_devices, repo, cmd))
    specs.append(
        (
            "reference",
            "normal",
            str(devices[6]),
            ROOT,
            common + ["--run_mode", "normal", "--port", str(args.port + 300), "--nccl_port", str(args.port + 6300)],
        )
    )
    manifest, processes = [], []
    endpoints = {v: f"http://{host}:{args.port + 100 * i}" for i, v in enumerate(VARIANTS)}
    endpoints["reference"] = f"http://{host}:{args.port + 300}"
    write_json(out / "endpoints.json", endpoints)
    # Check both HTTP and rendezvous ports before creating any process.
    for _, _, _, _, cmd in specs:
        for opt in ("--port", "--nccl_port"):
            if opt in cmd:
                with socket.socket() as s:
                    s.bind(("", int(cmd[cmd.index(opt) + 1])))
    try:
        for variant, role, gpus, repo, cmd in specs:
            log = out / f"{variant}_{role}.log"
            handle = log.open("w")
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpus,
                PYTHONPATH=str(repo),
                PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="2",
                LOADWORKER="4",
                LIGHTLLM_LOG_LEVEL="debug",
                NO_PROXY=f"{host},localhost,127.0.0.1",
                no_proxy=f"{host},localhost,127.0.0.1",
            )
            command = [sys.executable, "-m", "lightllm.server.api_server", *cmd]
            p = subprocess.Popen(
                command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append((p, handle))
            manifest.append(
                dict(variant=variant, role=role, pid=p.pid, gpus=gpus, root=str(repo), command=command, log=str(log))
            )
            write_json(out / "processes.json", manifest)
        client = httpx.Client(trust_env=False, timeout=2)
        pending = {f"{v}_{r}": f"http://{host}:{cmd[cmd.index('--port') + 1]}" for v, r, _, _, cmd in specs}
        deadline = time.monotonic() + 900
        while pending:
            for p, _ in processes:
                if p.poll() is not None:
                    raise RuntimeError(f"service pid={p.pid} exited with {p.returncode}")
            for name, url in list(pending.items()):
                try:
                    if client.get(url + "/readiness").status_code == 200:
                        del pending[name]
                        print("READY", name, flush=True)
                except httpx.HTTPError:
                    pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"Startup timed out: {list(pending)}")
            if pending:
                time.sleep(2)
        write_json(out / "ready.json", dict(time=time.time(), endpoints=endpoints))
        print("ALL_READY", flush=True)
        while True:
            if any(p.poll() is not None for p, _ in processes):
                raise RuntimeError("A service exited")
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        for p, _ in processes:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 25
        for p, handle in processes:
            try:
                p.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(timeout=10)
            handle.close()
        write_json(out / "stopped.json", dict(time=time.time(), pids=[p.pid for p, _ in processes]))


TASKS = [
    "Compute 17 * 23. Give the integer result.",
    "把这句话翻译成英文：今天北京下雨，我们决定在家读书。",
    "Return JSON with keys name, age, active for Ada, 37, true.",
    "Write a Python function that reverses a singly linked list.",
    "Sort these integers ascending: 19, -3, 7, 0, 19, 2.",
    "Summarize: A red fox crossed the river at dawn. It returned with food for its cubs.",
    "Extract the email address from: Contact Ada at ada@example.org after Tuesday.",
    "Continue this SQL query: SELECT department, COUNT(*) FROM employees GROUP BY",
    "A train leaves at 09:15 and travels for 125 minutes. What is the arrival time?",
    "Explain why a binary search needs a sorted array in two sentences.",
    "Complete the sequence with the next three numbers: 2, 3, 5, 8, 13,",
    "Count the letter a in banana. Answer with a single number.",
    "Convert hexadecimal 2F into decimal and show the calculation.",
    "给小学生解释为什么月亮会有阴晴圆缺，用两句话回答。",
    "Translate into French: The library closes at six o'clock.",
    "Return a Markdown table for apples=3, pears=5, oranges=2.",
    "Fix the bug: def add(a, b): return a - b",
    "Decode these escaped characters and explain: \\n, \\t, \\u4e2d.",
    "List three differences between TCP and UDP.",
    "Given A implies B and B implies C, what follows if A is true?",
]


class Runner:
    def __init__(self, args):
        from transformers import AutoTokenizer

        self.args = args
        self.out = args.output_dir / args.profile
        self.urls = json.loads((self.out / "endpoints.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        self.client = httpx.Client(trust_env=False, timeout=httpx.Timeout(180, connect=5))
        self.hybrid = args.profile.startswith("hybrid")
        self.pf = args.profile.endswith("pf")
        self.results = []
        self.common = dict(
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

    def text(self, n, tag, task):
        if n < 32:
            text = "x" + " x" * (n - 1)
            assert len(self.tok.encode(text, add_special_tokens=False)) == n
            return text
        header = f"Review {self.args.profile} {tag}. Background records:\n"
        suffix = "\nIgnore the background records. Task: " + task + "\nAnswer:"
        pad = n - len(self.tok.encode(header + suffix, add_special_tokens=False))
        assert pad >= 0, (n, tag)
        for _ in range(12):
            text = header + " x" * pad + suffix
            actual = len(self.tok.encode(text, add_special_tokens=False))
            if actual == n:
                return text
            pad += n - actual
        raise AssertionError((n, actual, tag))

    def request(self, variant, text, params=None, stream=False):
        p = {**self.common, **(params or {})}
        if variant == "reference":
            p["disable_prompt_cache"] = True
        start = time.perf_counter()
        if not stream:
            response = self.client.post(self.urls[variant] + "/generate", json=dict(inputs=text, parameters=p))
            assert response.status_code == 200, (variant, response.status_code, response.text[:4000])
            data = response.json()
            tokens = data["tokens"]
            return dict(
                token_ids=[x["id"] for x in tokens],
                logprobs=[x["logprob"] for x in tokens],
                cache_tokens=max((x.get("prompt_cache_len", 0) for x in tokens), default=0),
                elapsed=time.perf_counter() - start,
                response=data,
            )
        tokens, chunks, ttft = [], [], None
        with self.client.stream(
            "POST", self.urls[variant] + "/generate_stream", json=dict(inputs=text, parameters=p)
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                if "token" not in chunk:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - start
                tokens.append(chunk["token"])
                chunks.append(chunk)
        elapsed = time.perf_counter() - start
        assert tokens, chunks
        return dict(
            token_ids=[x["id"] for x in tokens],
            logprobs=[x["logprob"] for x in tokens],
            cache_tokens=max((x.get("prompt_cache_len", 0) for x in tokens), default=0),
            ttft=ttft,
            elapsed=elapsed,
            tpot=(elapsed - ttft) / max(1, len(tokens) - 1),
            chunks=chunks,
        )

    def page_rows(self):
        log = (self.out / "current_on_decode.log").read_text()
        matches = re.findall(r"create lock shm (\w+)_cpu_kv_cache_page_items", log)
        if not matches:
            # The lock creation message is DEBUG in some builds. The CPU metadata log is INFO.
            matches = re.findall(r"(\w+)_cpu_kv_cache_page_items", log)
        if not matches:
            manifest = json.loads((self.out / "processes.json").read_text())
            p = next(p for p in manifest if p["variant"] == "current_on" and p["role"] == "decode")
            namespaces = re.findall(r"unique_server_name[:= ]+([\w]+)", log)
            candidates = [
                x
                for x in Path("/dev/shm").glob("*_cpu_kv_cache_page_items")
                if x.stat().st_mtime >= (self.out / "processes.json").stat().st_mtime - 900
            ]
            if len(candidates) != 1:
                raise RuntimeError(f"Cannot identify CPU metadata for pid {p['pid']}: {candidates}, {namespaces}")
            page_file = candidates[0]
            name = page_file.name.removesuffix("_cpu_kv_cache_page_items")
        else:
            name = matches[0]
            page_file = Path("/dev/shm") / (name + "_cpu_kv_cache_page_items")
        return name, list(struct.iter_unpack("<iiiiiQQ", page_file.read_bytes()))[:-2]

    def expected(self, text):
        n = len(self.tok.encode(text, add_special_tokens=False))
        return max(0, ((n - (2 if self.hybrid else 1)) // 256) * 256)

    def wait_cached(self, text):
        import xxhash
        import numpy as np

        ids = self.tok.encode(text, add_special_tokens=False)
        count = self.expected(text) // 256
        h = xxhash.xxh3_128()
        hashes = []
        for i in range(count):
            h.update(np.asarray(ids[i * 256 : (i + 1) * 256], dtype=np.uint64).tobytes())
            hashes.append(h.intdigest())
        if not hashes:
            return
        deadline = time.monotonic() + 20
        while True:
            _, rows = self.page_rows()
            ready = {r[5] + (r[6] << 64) for r in rows if r[3] == 2 and r[4] == 0}
            if all(h in ready for h in hashes):
                return
            if time.monotonic() > deadline:
                raise AssertionError(dict(error="offload not ready", hashes=hashes, ready_count=len(ready)))
            time.sleep(0.02)

    def warm(self, text):
        # Always go through D, including PF scenarios subsequently requesting one token.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda v: self.request(v, text, {"max_new_tokens": 4}), VARIANTS))
        self.wait_cached(text)

    def compare(self, text, params, expected, stream=False):
        log = self.out / "current_on_decode.log"
        log_offset = log.stat().st_size
        variants = (*VARIANTS, "reference")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            samples = dict(zip(variants, pool.map(lambda v: self.request(v, text, params, stream), variants)))
        reference = samples["reference"]["token_ids"]
        time.sleep(0.04)
        with log.open() as handle:
            handle.seek(log_offset)
            evidence = handle.read()
        cpu = [int(x) for x in re.findall(r"cpu_prompt_cache_len:(\d+)", evidence)]
        gpu = [int(x) for x in re.findall(r"gpu_prompt_cache_len:(\d+)", evidence)]
        actual_cache = max(cpu, default=0)
        gpu_cache = max(gpu, default=0)
        n = len(self.tok.encode(text, add_special_tokens=False))
        if self.pf and params.get("max_new_tokens", 8) == 1:
            expected = 0
        data = dict(
            prompt_tokens=n,
            expected_cpu_tokens=expected,
            actual_cache_tokens=actual_cache,
            actual_gpu_tokens=gpu_cache,
            cpu_log_values=cpu,
            gpu_log_values=gpu,
            parameters=params,
            stream=stream,
            token_ids={v: s["token_ids"] for v, s in samples.items()},
            max_logprob_delta={
                v: max((abs(a - b) for a, b in zip(s["logprobs"], samples["reference"]["logprobs"])), default=0)
                for v, s in samples.items()
            },
            elapsed={v: s["elapsed"] for v, s in samples.items()},
        )
        assert all(s["token_ids"] == reference for s in samples.values()), data
        assert len(reference) == params.get("max_new_tokens", 8), data
        # Disk is off. Verify actual CPU/GPU lengths separately; count a CPU hit
        # only when the additional interval meets the real H2D admission threshold.
        if not stream:
            if expected > 0:
                assert cpu, data
            assert actual_cache == max(0, expected - gpu_cache), data
            assert expected <= gpu_cache or expected - gpu_cache >= 128, data
            assert samples["reference"]["cache_tokens"] == 0, data
        return data

    def case(self, name, family, fn, definition=None):
        r = dict(name=name, family=family, status="RUNNING", definition=definition or {})
        start = time.monotonic()
        try:
            r.update(fn() or {})
            r["status"] = "PASS"
        except Exception as exc:
            if isinstance(exc, AssertionError) and exc.args and isinstance(exc.args[0], dict):
                r.update(exc.args[0])
            r.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        r["seconds"] = time.monotonic() - start
        self.results.append(r)
        write_json(self.out / f"{self.args.label}_correctness_results.json", self.results)
        print(name, r["status"], round(r["seconds"], 3), flush=True)

    def correctness(self):
        if self.args.profile == "dense_normal":
            self.penalty_cache_invariance_probe()
        definitions = []
        for task_id, task in enumerate(TASKS[: self.args.task_count]):
            params = {"max_new_tokens": (8, 16, 32, 2, 1)[task_id % 5]}
            if task_id % 4 == 1:
                params["repetition_penalty"] = 1.05
            elif task_id % 4 == 2:
                params["presence_penalty"] = 0.1
            elif task_id % 4 == 3:
                params["frequency_penalty"] = 0.1
            for state in ("cold", "warm", "partial"):
                values = {
                    "cold": [127, 256, 257, 258, 1025, 2818],
                    "warm": [257, 258, 513, 514, 1025, 2049],
                    "partial": [0, 127, 255, 256, 511, 768],
                }[state]
                for value in values:
                    definitions.append(dict(task_id=task_id, task=task, state=state, value=value, params=params))
        if self.args.limit:
            definitions = definitions[: self.args.limit]
        write_json(self.out / f"{self.args.label}_scenario_definitions.json", definitions)
        for d in definitions:
            tag = f"{self.args.label}-T{d['task_id']:02d}-{d['state']}-{d['value']}"
            n = 1538 if d["state"] == "partial" else d["value"]

            def execute(d=d, tag=tag, n=n):
                text = self.text(n, tag, d["task"])
                expected = 0
                if d["state"] in ("warm", "partial"):
                    self.warm(text)
                    expected = self.expected(text)
                if d["state"] == "partial":
                    ids = self.tok.encode(text, add_special_tokens=False)
                    changed = ids.copy()
                    replacement = self.tok.encode(" y", add_special_tokens=False)[0]
                    changed[d["value"]] = replacement if ids[d["value"]] != replacement else replacement + 1
                    text = self.tok.decode(changed)
                    reencoded = self.tok.encode(text, add_special_tokens=False)
                    prefix = next(
                        (i for i, (a, b) in enumerate(zip(ids, reencoded)) if a != b), min(len(ids), len(reencoded))
                    )
                    expected = min(prefix // 256 * 256, self.expected(text))
                return self.compare(text, d["params"], expected)

            self.case(tag, d["state"], execute, d)
        if not self.args.limit:
            for i in range(24):
                n = [258, 513, 514, 1025, 2049, 4098][i % 6]
                output = [1, 2, 8, 32][i // 6]
                tag = f"{self.args.label}-stream-{n}-{output}"

                def streaming(i=i, n=n, output=output, tag=tag):
                    text = self.text(n, tag, TASKS[i % len(TASKS)])
                    self.warm(text)
                    return self.compare(text, {"max_new_tokens": output}, self.expected(text), stream=True)

                self.case(tag, "stream", streaming)
            for count in (2, 4, 8):
                for state in ("cold_distinct", "warm_distinct", "warm_shared"):
                    tag = f"{self.args.label}-concurrent-{count}-{state}"

                    def concurrent_case(count=count, state=state, tag=tag):
                        texts = [self.text([258, 514, 1025, 1538][i % 4], tag + str(i), TASKS[i]) for i in range(count)]
                        if state == "warm_shared":
                            texts = [texts[0]] * count
                        if state.startswith("warm"):
                            for text in dict.fromkeys(texts):
                                self.warm(text)
                        references = [self.request("reference", t)["token_ids"] for t in texts]
                        results = {}
                        for v in VARIANTS:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                                actual = list(pool.map(lambda t: self.request(v, t), texts))
                            results[v] = [a["token_ids"] for a in actual]
                        assert all(ids == references for ids in results.values()), dict(
                            token_ids=results, reference_ids=references
                        )
                        return dict(count=count, state=state, token_ids=results, reference_ids=references)

                    self.case(tag, "concurrency", concurrent_case)
            for n in (1, 2, 3, 4):
                self.case(
                    f"{self.args.label}-tiny-{n}",
                    "tiny_prompt",
                    lambda n=n: self.compare(self.text(n, "tiny", ""), {"max_new_tokens": 8}, 0),
                )
            invalid_parameters = [
                {"max_new_tokens": 0},
                {"max_new_tokens": -1},
                {"do_sample": True, "temperature": -1},
                {"do_sample": True, "top_p": 1.1},
                {"do_sample": True, "top_k": -2},
                {"repetition_penalty": 0},
                {"presence_penalty": -1},
                {"frequency_penalty": -1},
            ]
            for i, parameters in enumerate(invalid_parameters):

                def invalid(parameters=parameters):
                    responses = {}
                    for v in (*VARIANTS, "reference"):
                        response = self.client.post(
                            self.urls[v] + "/generate",
                            json=dict(inputs="Hello", parameters={**self.common, **parameters}),
                        )
                        responses[v] = dict(status=response.status_code, body=response.text[:1000])
                    assert all(400 <= r["status"] < 500 for r in responses.values()), responses
                    return dict(parameters=parameters, responses=responses)

                self.case(f"{self.args.label}-invalid-{i}", "invalid_parameters", invalid)
            for after_first in (False, True):

                def cancellation(after_first=after_first):
                    text = self.text(2050, f"{self.args.label}-cancel-{after_first}", TASKS[0])
                    self.warm(text)
                    for v in VARIANTS:
                        with self.client.stream(
                            "POST",
                            self.urls[v] + "/generate_stream",
                            json=dict(inputs=text, parameters={**self.common, "max_new_tokens": 1024}),
                        ) as response:
                            response.raise_for_status()
                            if after_first:
                                next(line for line in response.iter_lines() if line.startswith("data:"))
                    time.sleep(0.8)
                    recovery = self.text(514, f"{self.args.label}-recovery-{after_first}", TASKS[0])
                    return dict(after_first=after_first, recovery=self.compare(recovery, {"max_new_tokens": 8}, 0))

                self.case(f"{self.args.label}-cancel-{after_first}", "cancellation", cancellation)
            self.case("final_cpu_reference_audit", "resource_audit", self.audit)
        write_json(
            self.out / f"{self.args.label}_correctness_summary.json",
            dict(
                total=len(self.results),
                passed=sum(r["status"] == "PASS" for r in self.results),
                failed=sum(r["status"] == "FAIL" for r in self.results),
                positive_hits=sum(r.get("actual_cache_tokens", 0) > 0 for r in self.results),
                zero_hits=sum(r.get("actual_cache_tokens") == 0 for r in self.results),
            ),
        )

    def audit(self):
        deadline = time.monotonic() + 20
        while True:
            namespace, rows = self.page_rows()
            refs = [r[4] for r in rows]
            if not any(refs):
                break
            assert time.monotonic() < deadline, refs
            time.sleep(0.1)
        counters = {
            p.name: struct.unpack("<q", p.read_bytes())[0]
            for p in Path("/dev/shm").glob(namespace + "_kv_available_tokens_*")
        }
        return dict(
            namespace=namespace,
            cpu_pages=len(rows),
            ready_pages=sum(r[3] == 2 for r in rows),
            all_cpu_refs_zero=True,
            gpu_free_tokens=counters,
        )

    def performance(self):
        if self.args.profile == "hybrid_normal":
            self.concurrency_probe()
        if self.args.profile == "dense_normal":
            self.strong_penalty_probe()
        # Every batch is measured with exactly one variant active. All variants use
        # dedicated H200s, identical flags, HTTP/stream parsing and prompt contents.
        # Rotate variant order so warm machine state and drift do not favor one arm.
        groups = [
            ("cold_short", "cold", 258, 16, 1),
            ("cold_long", "cold", 4098, 16, 1),
            ("hot_short", "hot", 514, 16, 1),
            ("hot_long", "hot", 4098, 16, 1),
            ("hot_full_match_long", "hot", 4097, 16, 1),
            ("hot_full_match_concurrent4", "hot", 2049, 16, 4),
            ("hot_long_decode", "hot", 4098, 128, 1),
            ("partial_long", "partial", 4098, 16, 1),
            ("hot_concurrent4", "hot", 4098, 32, 4),
            ("hot_concurrent8", "hot", 2050, 32, 8),
            ("cold_concurrent4", "cold", 4098, 16, 4),
            ("boundary257", "hot", 257, 16, 1),
            ("boundary513", "hot", 513, 16, 1),
        ]
        all_results, summary = [], []
        rng = random.Random(1760)
        for name, state, n, output, concurrency in groups:
            if self.args.bench_filter and not re.search(self.args.bench_filter, name):
                continue
            hot_text = self.text(n, f"{self.args.label}-perf-{name}", TASKS[0])
            if state == "hot":
                for variant in VARIANTS:
                    for _ in range(3):
                        self.request(variant, hot_text, {"max_new_tokens": output}, stream=True)
                self.wait_cached(hot_text)
            rounds = {variant: [] for variant in VARIANTS}
            # Two discarded warmup batches followed by nine measured batches per arm.
            for iteration in range(self.args.bench_rounds + 2):
                text = hot_text
                if state in ("cold", "partial"):
                    text = self.text(n, f"{self.args.label}-perf-{name}-R{iteration}", TASKS[iteration % len(TASKS)])
                if state == "partial":
                    for variant in VARIANTS:
                        self.request(variant, text)
                    self.wait_cached(text)
                    ids = self.tok.encode(text, add_special_tokens=False)
                    ids[n // 2] = self.tok.encode(" y", add_special_tokens=False)[0]
                    text = self.tok.decode(ids)
                order = list(VARIANTS)
                rng.shuffle(order)
                for variant in order:
                    logs = {role: self.out / f"{variant}_{role}.log" for role in ("prefill", "decode")}
                    offsets = {role: path.stat().st_size for role, path in logs.items()}
                    start = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        samples = list(
                            pool.map(
                                lambda _: self.request(variant, text, {"max_new_tokens": output}, stream=True),
                                range(concurrency),
                            )
                        )
                    wall = time.perf_counter() - start
                    assert all(len(s["token_ids"]) == output for s in samples), (name, variant, samples)
                    time.sleep(0.03)
                    evidence = {}
                    for role, path in logs.items():
                        with path.open() as handle:
                            handle.seek(offsets[role])
                            tail = handle.read()
                        evidence[role] = {
                            tier: [int(x) for x in re.findall(tier + r"_prompt_cache_len:(\d+)", tail)]
                            for tier in ("cpu", "gpu", "disk")
                        }
                    record = dict(
                        group=name,
                        variant=variant,
                        iteration=iteration,
                        measured=iteration >= 2,
                        state=state,
                        prompt_tokens=len(self.tok.encode(text, add_special_tokens=False)),
                        output_tokens=output,
                        concurrency=concurrency,
                        wall_seconds=wall,
                        output_tokens_per_second=concurrency * output / wall,
                        evidence=evidence,
                        samples=[{k: v for k, v in s.items() if k != "chunks"} for s in samples],
                    )
                    all_results.append(record)
                    write_json(self.out / f"{self.args.label}_performance_raw.json", all_results)
                    if iteration >= 2:
                        rounds[variant].append(record)
                    # Let final page publication and the request accounting settle.
                    time.sleep(0.03)
                batch_records = all_results[-len(VARIANTS) :]
                origin_ids = next(r for r in batch_records if r["variant"] == "origin_off")["samples"]
                origin_ids = [s["token_ids"] for s in origin_ids]
                for record in batch_records:
                    record["accuracy_vs_origin"] = [s["token_ids"] for s in record["samples"]] == origin_ids
                write_json(self.out / f"{self.args.label}_performance_raw.json", all_results)
                assert all(r["accuracy_vs_origin"] for r in batch_records), batch_records
                print("BENCH", name, iteration, flush=True)
            group_summary = dict(
                group=name, state=state, prompt_tokens=n, output_tokens=output, concurrency=concurrency, variants={}
            )
            for variant, records in rounds.items():
                samples = [s for r in records for s in r["samples"]]

                def percentile(values, q):
                    values = sorted(values)
                    index = (len(values) - 1) * q
                    lo = int(index)
                    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)

                group_summary["variants"][variant] = dict(
                    rounds=len(records),
                    requests=len(samples),
                    ttft_p50_ms=statistics.median(s["ttft"] for s in samples) * 1000,
                    ttft_p95_ms=percentile([s["ttft"] for s in samples], 0.95) * 1000,
                    tpot_p50_ms=statistics.median(s["tpot"] for s in samples) * 1000,
                    e2e_p50_ms=statistics.median(s["elapsed"] for s in samples) * 1000,
                    output_tps_p50=statistics.median(r["output_tokens_per_second"] for r in records),
                )
            baseline = group_summary["variants"]["origin_off"]
            for metrics in group_summary["variants"].values():
                metrics["ttft_speedup_vs_origin"] = baseline["ttft_p50_ms"] / metrics["ttft_p50_ms"]
                metrics["throughput_speedup_vs_origin"] = metrics["output_tps_p50"] / baseline["output_tps_p50"]
            summary.append(group_summary)
            write_json(self.out / f"{self.args.label}_performance_summary.json", summary)
        write_json(self.out / f"{self.args.label}_post_performance_audit.json", self.audit())

    def concurrency_probe(self):
        """Retest the exact mixed-batch prompts against sequential and batched Normal."""
        records = []
        for count in (4, 8):
            tag = f"wide-concurrent-{count}-warm_distinct"
            texts = [self.text([258, 514, 1025, 1538][i % 4], tag + str(i), TASKS[i]) for i in range(count)]
            for text in texts:
                self.warm(text)
            sequential = [self.request("reference", text)["token_ids"] for text in texts]
            for trial in range(3):
                outputs = {}
                for variant in (*VARIANTS, "reference"):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                        outputs[variant] = [r["token_ids"] for r in pool.map(lambda t: self.request(variant, t), texts)]
                record = dict(
                    count=count,
                    trial=trial,
                    sequential_normal=sequential,
                    outputs=outputs,
                    on_equals_origin=outputs["current_on"] == outputs["origin_off"],
                    off_equals_origin=outputs["current_off"] == outputs["origin_off"],
                    normal_batch_equals_sequential=outputs["reference"] == sequential,
                )
                records.append(record)
                write_json(self.out / "concurrency_probe.json", records)

    def penalty_cache_invariance_probe(self):
        """Check whether CPU-full-match changes output with a fixed repetition penalty."""
        records = []
        for n in (257, 513, 4097, 4098):
            text = self.text(n, f"cache-invariance-{n}", TASKS[1])
            params = {"max_new_tokens": 32, "repetition_penalty": 1.05}
            trials = {}
            for state in ("cold", "cpu_warm"):
                if state == "cpu_warm":
                    self.wait_cached(text)
                trials[state] = {v: self.request(v, text, params) for v in (*VARIANTS, "reference")}
            ids = {state: {v: sample["token_ids"] for v, sample in arm.items()} for state, arm in trials.items()}
            records.append(
                dict(
                    prompt_tokens=n,
                    parameters=params,
                    token_ids=ids,
                    on_cache_invariant=ids["cold"]["current_on"] == ids["cpu_warm"]["current_on"],
                    off_cache_invariant=ids["cold"]["current_off"] == ids["cpu_warm"]["current_off"],
                    origin_cache_invariant=ids["cold"]["origin_off"] == ids["cpu_warm"]["origin_off"],
                    hot_on_equals_origin=ids["cpu_warm"]["current_on"] == ids["cpu_warm"]["origin_off"],
                    hot_on_equals_normal=ids["cpu_warm"]["current_on"] == ids["cpu_warm"]["reference"],
                    raw=trials,
                )
            )
            write_json(self.out / "penalty_cache_invariance_probe.json", records)

    def strong_penalty_probe(self):
        records = []
        tasks = [
            "Print the digit 7 twenty times, separated by spaces. No explanation.",
            "只输出十六个‘哈’字，不要标点，不要解释。",
            "Continue with twelve more digits and spaces: 1 1 1 1 1 1",
            "Print a JSON array containing the number 0 twelve times. No other text.",
        ]
        for i, task in enumerate(tasks):
            for penalty in (dict(repetition_penalty=1.8), dict(frequency_penalty=2.0)):
                text = self.text(513, f"strong-penalty-{i}-{list(penalty)[0]}", task)
                params = dict(max_new_tokens=24, input_penalty=False, **penalty)
                trials = {}
                for state in ("cold", "cpu_warm"):
                    if state == "cpu_warm":
                        self.wait_cached(text)
                    variants = (*VARIANTS, "reference")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                        trials[state] = dict(zip(variants, pool.map(lambda v: self.request(v, text, params), variants)))
                ids = {state: {v: s["token_ids"] for v, s in arm.items()} for state, arm in trials.items()}
                records.append(
                    dict(
                        task=task,
                        parameters=params,
                        token_ids=ids,
                        hot_on_equals_origin=ids["cpu_warm"]["current_on"] == ids["cpu_warm"]["origin_off"],
                        hot_off_equals_origin=ids["cpu_warm"]["current_off"] == ids["cpu_warm"]["origin_off"],
                        hot_on_equals_normal=ids["cpu_warm"]["current_on"] == ids["cpu_warm"]["reference"],
                        raw=trials,
                    )
                )
                write_json(self.out / "strong_penalty_probe.json", records)

    def working_set_performance(self):
        """Cycle a corpus larger than GPU KV capacity but smaller than CPU capacity."""
        raw, summary = [], []
        rng = random.Random(1760)
        task = "Compute 17 * 23. Reply only with the integer, without explanation."
        for name, state, n in (
            ("evicted_gpu_cpu_full", "full", 4097),
            ("evicted_gpu_cpu_partial", "partial", 4098),
            ("evicted_gpu_cold", "cold", 4097),
        ):
            corpus = [self.text(n, f"working-set-{name}-{i}", task) for i in range(4)]
            if state != "cold":
                for text in corpus:
                    for variant in VARIANTS:
                        self.request(variant, text, {"max_new_tokens": 8})
                    self.wait_cached(text)
            group_records = {v: [] for v in VARIANTS}
            for iteration in range(14):
                text = corpus[iteration % 4] if state != "cold" else self.text(n, f"working-set-cold-{iteration}", task)
                order = list(VARIANTS)
                rng.shuffle(order)
                batch = []
                for variant in order:
                    logs = {role: self.out / f"{variant}_{role}.log" for role in ("prefill", "decode")}
                    offsets = {role: path.stat().st_size for role, path in logs.items()}
                    sample = self.request(variant, text, {"max_new_tokens": 4}, stream=True)
                    assert len(sample["token_ids"]) == 4, sample
                    if variant == "current_on":
                        # Outside the measured response interval: finish D2H before
                        # the next arm uses the host/PCIe fabric.
                        self.wait_cached(text)
                    time.sleep(0.03)
                    evidence = {}
                    for role, path in logs.items():
                        with path.open() as handle:
                            handle.seek(offsets[role])
                            tail = handle.read()
                        evidence[role] = {
                            tier: [int(x) for x in re.findall(tier + r"_prompt_cache_len:(\d+)", tail)]
                            for tier in ("cpu", "gpu")
                        }
                    record = dict(
                        group=name,
                        variant=variant,
                        iteration=iteration,
                        measured=iteration >= 2,
                        state=state,
                        prompt_tokens=n,
                        output_tokens=4,
                        concurrency=1,
                        corpus_prompts=4,
                        gpu_capacity_tokens=8192,
                        cpu_capacity_gib=4,
                        prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        wall_seconds=sample["elapsed"],
                        output_tokens_per_second=4 / sample["elapsed"],
                        samples=[{k: v for k, v in sample.items() if k != "chunks"}],
                        evidence=evidence,
                    )
                    batch.append(record)
                    raw.append(record)
                    if iteration >= 2:
                        group_records[variant].append(record)
                origin = next(r for r in batch if r["variant"] == "origin_off")["samples"][0]["token_ids"]
                for record in batch:
                    record["accuracy_vs_origin"] = record["samples"][0]["token_ids"] == origin
                write_json(self.out / f"{self.args.label}_performance_raw.json", raw)
                assert all(r["accuracy_vs_origin"] for r in batch), batch
                on = next(r for r in batch if r["variant"] == "current_on")
                if iteration >= 2:
                    cpu = max(on["evidence"]["decode"]["cpu"], default=0)
                    gpu = max(on["evidence"]["decode"]["gpu"], default=0)
                    assert gpu < 128 and ((cpu >= 3968) if state != "cold" else cpu == 0), on
                print("WORKING_SET", name, iteration, flush=True)
            group = dict(group=name, state=state, prompt_tokens=n, output_tokens=4, concurrency=1, variants={})
            for variant, records in group_records.items():
                samples = [r["samples"][0] for r in records]
                group["variants"][variant] = dict(
                    rounds=len(records),
                    requests=len(samples),
                    ttft_p50_ms=statistics.median(s["ttft"] for s in samples) * 1000,
                    ttft_p95_ms=statistics.quantiles([s["ttft"] for s in samples], n=100, method="inclusive")[94]
                    * 1000,
                    tpot_p50_ms=statistics.median(s["tpot"] for s in samples) * 1000,
                    e2e_p50_ms=statistics.median(s["elapsed"] for s in samples) * 1000,
                    output_tps_p50=statistics.median(r["output_tokens_per_second"] for r in records),
                )
            baseline = group["variants"]["origin_off"]
            for values in group["variants"].values():
                values["ttft_speedup_vs_origin"] = baseline["ttft_p50_ms"] / values["ttft_p50_ms"]
                values["throughput_speedup_vs_origin"] = values["output_tps_p50"] / baseline["output_tps_p50"]
            summary.append(group)
            write_json(self.out / f"{self.args.label}_performance_summary.json", summary)
        write_json(self.out / f"{self.args.label}_post_performance_audit.json", self.audit())


def run(args):
    runner = Runner(args)
    if args.action == "correctness":
        runner.correctness()
        raise SystemExit(int(any(r["status"] == "FAIL" for r in runner.results)))
    elif args.action == "working_set":
        runner.working_set_performance()
    else:
        runner.performance()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["serve", "correctness", "performance", "working_set"])
    parser.add_argument("--profile", default="hybrid_normal")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--origin-root", type=Path)
    parser.add_argument("--model-dir", default="/mtc/models/Qwen3.5-0.8B")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="full")
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--bench-rounds", type=int, default=9)
    parser.add_argument("--bench-filter")
    parser.add_argument("--gpu-order", default="0,1,2,3,4,5,6")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.action == "serve":
        serve(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
