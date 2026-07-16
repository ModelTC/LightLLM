#!/usr/bin/env python3
"""Run, retain, parse, and render the GLM-5.2 static benchmark matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CONTEXTS = [16384, 32768, 65536, 131072, 204800, 256000]
HITS = [0.0, 0.5, 0.7, 0.8, 0.9]
LIMITS = [20.0, 33.0, 50.0]
LABELS = {16384: "16k", 32768: "32k", 65536: "64k", 131072: "128k", 204800: "200k", 256000: "250k"}
NODE_HOSTS = ["10.119.17.190", "10.119.17.202", "10.119.17.179", "10.119.17.169"]
GPUS_PER_NODE = 8


def csv(values):
    return ",".join(str(v) for v in values)


def base_command(
    model_dir: str,
    mem_fraction: float,
    nccl_host: str,
    nccl_port: int,
    batch_max_tokens: int,
    mtp: bool,
    node_rank: int,
    nnodes: int,
    gpus_per_node: int,
    expert_dtype: str,
    graph_max_batch_size: int,
):
    world_size = nnodes * gpus_per_node
    cmd = [
        sys.executable, "test/benchmark/static_inference/test_model.py",
        "--model_dir", os.path.expanduser(model_dir), "--tp", str(world_size),
        "--llm_kv_type", "fp8kv_dsa", "--batch_max_tokens", str(batch_max_tokens),
        "--output_lens", "600", "--decode_batch_size_mode", "profile",
        "--quant_type", "deepgemm-fp8w8a8-b128", "--dp", str(world_size),
        "--enable_ep_moe", "--expert_dtype", expert_dtype,
        "--graph_max_batch_size", str(graph_max_batch_size),
        "--mem_fraction", str(mem_fraction),
        "--nnodes", str(nnodes), "--node_rank", str(node_rank),
        "--nccl_host", nccl_host, "--nccl_port", str(nccl_port),
    ]
    if mtp:
        cmd += [
            "--mtp_mode", "eagle_with_att",
            "--mtp_step", "2",
            "--mtp_draft_model_dir", os.path.expanduser("~/models/GLM-5.2/"),
            "--mtp_accept_rate", "0.85",
        ]
    return cmd


def launch_nodes(root: Path, node_hosts, commands, log_paths, env):
    processes = []
    handles = []
    remote_env = {
        "PYTHONPATH": env["PYTHONPATH"],
        "LIGHTLLM_TRITON_AUTOTUNE_LEVEL": env["LIGHTLLM_TRITON_AUTOTUNE_LEVEL"],
        "LOADWORKER": env["LOADWORKER"],
    }
    launch_order = list(range(1, len(node_hosts))) + [0]
    try:
        for rank in launch_order:
            log_handle = log_paths[rank].open("w", encoding="utf-8")
            handles.append(log_handle)
            if rank == 0:
                proc = subprocess.Popen(
                    commands[rank], cwd=root, env=env,
                    text=True, stdout=log_handle, stderr=subprocess.STDOUT,
                )
            else:
                exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in remote_env.items())
                remote_command = (
                    f"cd {shlex.quote(str(root))} && exec env {exports} "
                    f"{shlex.join(commands[rank])}"
                )
                proc = subprocess.Popen(
                    [
                        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        node_hosts[rank], f"bash -lc {shlex.quote(remote_command)}",
                    ],
                    text=True, stdout=log_handle, stderr=subprocess.STDOUT,
                )
            processes.append((rank, proc))

        while any(proc.poll() is None for _, proc in processes):
            if any(proc.poll() not in (None, 0) for _, proc in processes):
                for _, proc in processes:
                    if proc.poll() is None:
                        proc.terminate()
                break
            time.sleep(1)
        returncodes = [None] * len(node_hosts)
        for rank, proc in processes:
            try:
                returncodes[rank] = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncodes[rank] = proc.wait()
        return returncodes
    finally:
        for handle in handles:
            handle.close()


def parse_table(text: str, stage: str):
    marker = f"[{stage}]"
    pos = text.rfind(marker)
    if pos < 0:
        return []
    lines = [line.strip() for line in text[pos + len(marker):].splitlines() if line.strip()]
    if len(lines) < 3:
        return []
    headers = re.split(r"\s{2,}", lines[0])
    rows = []
    for line in lines[2:]:
        fields = re.split(r"\s{2,}", line)
        if len(fields) != len(headers):
            break
        rows.append(dict(zip(headers, fields)))
    return rows


def run_case(args, stage: str, contexts, hits, max_batch_size=None):
    root = Path(__file__).resolve().parents[3]
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    node_hosts = [host for host in re.split(r"[, ]+", args.node_hosts) if host]
    if not node_hosts:
        raise ValueError("--node-hosts must contain at least one host")
    nccl_host = args.nccl_host or node_hosts[0]
    mf = str(args.mem_fraction).replace(".", "p")
    mode_suffix = "_mtp" if args.mtp else ""
    config_suffix = f"_n{len(node_hosts)}_btok{args.batch_max_tokens}_mf{mf}{mode_suffix}"
    if stage == "prefill":
        config_suffix += f"_chunk{args.prefill_chunk_size}_overlap"
    hit_suffix = "-".join(str(v).replace(".", "p") for v in hits)
    suffix = (f"prefill_ctx{'-'.join(map(str, contexts))}_hit{hit_suffix}{config_suffix}" if stage == "prefill" else
              f"decode_cap{max_batch_size}_ctx{'-'.join(map(str, contexts))}{config_suffix}")
    stem = out_dir / suffix
    parsed_path = stem.with_suffix(".json")
    log_paths = [
        stem.with_suffix(".log") if rank == 0 else Path(f"{stem}.rank{rank}.log")
        for rank in range(len(node_hosts))
    ]
    manifest_path = stem.with_suffix(".manifest.json")
    if parsed_path.exists() and not args.force:
        print(f"reuse {parsed_path}")
        return 0

    commands = []
    for node_rank in range(len(node_hosts)):
        cmd = base_command(
            args.model_dir, args.mem_fraction, nccl_host, args.nccl_port,
            args.batch_max_tokens, args.mtp, node_rank, len(node_hosts),
            args.gpus_per_node, args.expert_dtype,
            args.graph_max_batch_size,
        )
        cmd += ["--benchmark", stage]
        if stage == "prefill":
            cmd += [
                "--input_lens", csv(contexts),
                "--prefill_cache_hit_rates", csv(hits),
                "--chunked_prefill_sizes", str(args.prefill_chunk_size),
                "--enable_prefill_microbatch_overlap",
                "--enable_tpsp_mix_mode",
            ]
        else:
            cmd += ["--context_lens", csv(contexts), "--max_batch_size", str(max_batch_size)]
        commands.append(cmd)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["LIGHTLLM_TRITON_AUTOTUNE_LEVEL"] = "1"
    env["LOADWORKER"] = "18"
    started = datetime.now(timezone.utc).isoformat()
    for rank, (host, cmd) in enumerate(zip(node_hosts, commands)):
        print(f"running node {rank} ({host}):", shlex.join(cmd), flush=True)
    returncodes = launch_nodes(root, node_hosts, commands, log_paths, env)
    rank0_output = log_paths[0].read_text(encoding="utf-8", errors="replace")
    rows = parse_table(rank0_output, stage)
    manifest = {"started_utc": started, "command": commands[0], "mem_fraction": args.mem_fraction,
                "batch_max_tokens": args.batch_max_tokens, "mtp": args.mtp,
                "prefill_chunk_size": args.prefill_chunk_size if stage == "prefill" else None,
                "prefill_microbatch_overlap": stage == "prefill",
                "returncode": next((code for code in returncodes if code), 0),
                "log": str(log_paths[0]), "row_count": len(rows),
                "nodes": [
                    {"host": host, "node_rank": rank, "command": commands[rank],
                     "log": str(log_paths[rank]), "returncode": returncodes[rank]}
                    for rank, host in enumerate(node_hosts)
                ]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if any(returncodes) or not rows:
        for rank, log_path in enumerate(log_paths):
            output = log_path.read_text(encoding="utf-8", errors="replace")
            print(f"node {rank} log tail ({log_path}):\n" + "\n".join(output.splitlines()[-40:]), file=sys.stderr)
        return next((code for code in returncodes if code), 2)
    parsed_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(rank0_output[rank0_output.rfind(f"[{stage}]"):], flush=True)
    return 0


def load_rows(out_dir: Path, pattern: str):
    rows = []
    for path in sorted(out_dir.glob(pattern)):
        if path.name.endswith(".manifest.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for row in payload:
            row = dict(row)
            row["source"] = path.name
            rows.append(row)
    return rows


def render(args):
    root = Path(__file__).resolve().parents[3]
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefill = load_rows(out_dir, "prefill*.json")
    decode = load_rows(out_dir, "decode_cap*.json")
    pmap = {(int(r["ctx"]), float(r["hit"])): r for r in prefill}
    lines = ["| ctx | hit | bs | max_total_token_num | qps | tok/s | logical_tok/s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for ctx in CONTEXTS:
        for hit in HITS:
            r = pmap.get((ctx, hit), {})
            lines.append(f'| {LABELS[ctx]} | {hit:g} | {r.get("bs", "")} | {r.get("max_total_token_num", "")} | {r.get("qps", "")} | {r.get("tok/s", "")} | {r.get("logical_tok/s", "")} |')
    if args.mtp:
        lines += ["", "| ctx | bs | accept | max_total_token_num | qps | tok/s | itl_ms | itl_ms约束 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
    else:
        lines += ["", "| ctx | bs | max_total_token_num | qps | tok/s | itl_ms | itl_ms约束 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
    for ctx in CONTEXTS:
        ctx_rows = [r for r in decode if int(r["ctx"]) == ctx]
        for limit in LIMITS:
            eligible = [r for r in ctx_rows if float(r["itl_ms"]) <= limit]
            r = max(eligible, key=lambda x: int(x["bs"]), default={})
            if args.mtp:
                lines.append(f'| {LABELS[ctx]} | {r.get("bs", "")} | {r.get("accept", "")} | {r.get("max_total_token_num", "")} | {r.get("qps", "")} | {r.get("tok/s", "")} | {r.get("itl_ms", "")} | <= {limit:g} |')
            else:
                lines.append(f'| {LABELS[ctx]} | {r.get("bs", "")} | {r.get("max_total_token_num", "")} | {r.get("qps", "")} | {r.get("tok/s", "")} | {r.get("itl_ms", "")} | <= {limit:g} |')
    output = "\n".join(lines) + "\n"
    table_path = out_dir / "tables.md"
    table_path.write_text(output, encoding="utf-8")
    print(output)


def cost_or_blank(rows, ctx: int, hit: float, machine_cost_per_second: float) -> str:
    row = rows.get((ctx, hit), {})
    tps = row.get("logical_tok/s")
    if not tps:
        return ""
    return f"{machine_cost_per_second / float(tps) * 1_000_000:.3f}"


def decode_cost_or_blank(rows, ctx: int, limit: float, machine_cost_per_second: float) -> str:
    ctx_rows = [r for r in rows if int(r["ctx"]) == ctx and float(r["itl_ms"]) <= limit]
    row = max(ctx_rows, key=lambda x: int(x["bs"]), default={})
    tps = row.get("tok/s")
    if not tps:
        return ""
    return f"{machine_cost_per_second / float(tps) * 1_000_000:.3f}"


def render_cost(args):
    root = Path(__file__).resolve().parents[3]
    out_dir = root / args.output_dir
    prefill = load_rows(out_dir, "prefill*.json")
    decode = load_rows(out_dir, "decode_cap*.json")
    pmap = {(int(r["ctx"]), float(r["hit"])): r for r in prefill}
    machine_cost_per_second = float(args.monthly_rent) / (30 * 24 * 60 * 60)

    lines = [
        "# GLM5.2 MTP 静态 PD 成本测算" if args.mtp else "# GLM5.2 静态 PD 成本测算",
        "",
        f"- 机器整机月租：{args.monthly_rent:g} 元/月",
        "- 每月按 30 天计算",
        f"- 机器整机每秒成本：`{args.monthly_rent:g} / 30 / 24 / 3600 = {machine_cost_per_second:.6f} 元/s`",
        "- Prefill 使用 prefill 表中的 `logical_tok/s` 计算",
        "- Decode 使用 decode 表中对应 `itl_ms约束` 档位的 `tok/s` 计算",
        "",
        "## 计算公式",
        "",
        "```text",
        "机器整机每秒成本 = 机器整机月租 / 30 / 24 / 3600",
        "输入成本 元/M tokens = 机器整机每秒成本 / prefill_logical_tok_s * 10^6",
        "输出成本 元/M tokens = 机器整机每秒成本 / decode_tok_s * 10^6",
        "```",
    ]

    header = (
        "| ctx | 输入cache miss成本 元/M tokens | 输出成本 元/M tokens | "
        "输入@50% hit平均成本 元/M logical tokens | "
        "输入@70% hit平均成本 元/M logical tokens | "
        "输入@80% hit平均成本 元/M logical tokens | "
        "输入@90% hit平均成本 元/M logical tokens |"
    )
    for limit in LIMITS:
        lines += [
            "",
            f"## Decode <= {limit:g}ms 成本表",
            "",
            header,
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for ctx in CONTEXTS:
            values = [
                cost_or_blank(pmap, ctx, 0.0, machine_cost_per_second),
                decode_cost_or_blank(decode, ctx, limit, machine_cost_per_second),
                cost_or_blank(pmap, ctx, 0.5, machine_cost_per_second),
                cost_or_blank(pmap, ctx, 0.7, machine_cost_per_second),
                cost_or_blank(pmap, ctx, 0.8, machine_cost_per_second),
                cost_or_blank(pmap, ctx, 0.9, machine_cost_per_second),
            ]
            lines.append(f"| {LABELS[ctx].upper()} | " + " | ".join(values) + " |")

    output = "\n".join(lines) + "\n"
    cost_path = out_dir / args.cost_output
    cost_path.write_text(output, encoding="utf-8")
    sys.stdout.write(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prefill", "decode-probe", "render", "cost"])
    parser.add_argument("--model-dir", default="~/models/GLM-5.2/")
    parser.add_argument("--output-dir", default="benchmark_results/glm52-static-matrix-4n")
    parser.add_argument("--cost-output", default="cost_analysis.md")
    parser.add_argument("--monthly-rent", type=float, default=280000.0)
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    parser.add_argument("--batch-max-tokens", type=int, default=4096)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--node-hosts", default=csv(NODE_HOSTS))
    parser.add_argument("--gpus-per-node", type=int, default=GPUS_PER_NODE)
    parser.add_argument("--nccl-host")
    parser.add_argument("--nccl-port", type=int, default=28765)
    parser.add_argument("--expert-dtype", choices=["fp8", "fp4"], default="fp8")
    parser.add_argument("--graph-max-batch-size", type=int, default=200)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--contexts", default=csv(CONTEXTS))
    parser.add_argument("--hits", default=csv(HITS))
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mtp and args.output_dir == "benchmark_results/glm52-static-matrix-4n":
        args.output_dir = "benchmark_results/glm52-static-matrix-4n-mtp"
    contexts = [int(x) for x in re.split(r"[, ]+", args.contexts) if x]
    hits = [float(x) for x in re.split(r"[, ]+", args.hits) if x]
    if args.action == "render":
        render(args)
        return
    if args.action == "cost":
        render_cost(args)
        return
    if args.action == "decode-probe" and not args.max_batch_size:
        parser.error("decode-probe requires --max-batch-size > 0")
    raise SystemExit(run_case(args, "prefill" if args.action == "prefill" else "decode", contexts, hits, args.max_batch_size))


if __name__ == "__main__":
    main()
