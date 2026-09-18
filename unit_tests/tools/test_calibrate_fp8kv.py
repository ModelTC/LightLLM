import importlib.util
import json
from pathlib import Path
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_fp8kv", Path(__file__).parents[2] / "tools" / "calibrate_fp8kv.py"
)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _snap(rank, per_head=True):
    return {
        "rank": rank,
        "layer_num": 2,
        "per_head": per_head,
        "shape": [2, 4 if per_head else 2],
        "head_num": 2,
        "num_target_layers": 2,
        "num_draft_layers": 0,
        "counts": [1, 2],
        "observed_token_rows": [4, 8],
        "qmin": -448.0,
        "qmax": 448.0,
        "architecture": "Fake",
        "abs_max": [[1 + rank, 2 + rank, 3 + rank, 4 + rank], [5 + rank, 6 + rank, 7 + rank, 8 + rank]]
        if per_head
        else [[1 + rank, 3 + rank], [5 + rank, 7 + rank]],
    }


def test_merge_per_head_orders_k_then_v_by_rank():
    out = tool.merge_rank_snapshots([_snap(1), _snap(0)], expected_ranks=2)
    assert out["quant_type"] == "per_head"
    assert out["scales_shape"] == [2, 8]
    assert out["scales"][0] == [1 / 448, 2 / 448, 2 / 448, 3 / 448, 3 / 448, 4 / 448, 4 / 448, 5 / 448]


def test_merge_per_tensor_takes_each_kv_maximum():
    out = tool.merge_rank_snapshots([_snap(0, False), _snap(1, False)], expected_ranks=2)
    assert out["quant_type"] == "per_tensor"
    assert out["scales"] == [[2 / 448, 4 / 448], [6 / 448, 8 / 448]]


def test_merge_q_marks_prefill_and_decode_and_embeds_without_changing_kv():
    q_rows = []
    for rank in range(2):
        row = _snap(rank)
        q_rows.append(
            {
                key: row[key]
                for key in (
                    "rank",
                    "layer_num",
                    "head_num",
                    "counts",
                    "observed_token_rows",
                    "qmin",
                    "qmax",
                    "architecture",
                    "num_target_layers",
                    "num_draft_layers",
                )
            }
            | {
                "tensor": "q",
                "per_head": True,
                "global_head_num": 2,
                "shape": [2, 2],
                "abs_max": [[1 + rank, 2 + rank], [5 + rank, 6 + rank]],
            }
        )
    q_cfg = tool.merge_q_rank_snapshots(q_rows, expected_ranks=2)
    assert q_cfg["calibration_stage"] == "prefill_and_decode"
    kv_cfg = tool.merge_rank_snapshots([_snap(0), _snap(1)], expected_ranks=2)
    combined = tool._embed_q_calibration(kv_cfg, q_cfg)
    assert combined["q_calibration"] == q_cfg
    assert {key: combined[key] for key in kv_cfg} == kv_cfg
    assert "q_calibration" not in kv_cfg


@pytest.mark.parametrize("change", ["missing", "nan"])
def test_merge_rejects_missing_or_nonfinite_rank_data(change):
    rows = [_snap(0), _snap(1)]
    if change == "missing":
        rows[1]["counts"] = [0, 1]
    else:
        rows[1]["abs_max"][0][0] = float("nan")
    with pytest.raises(ValueError):
        tool.merge_rank_snapshots(rows, expected_ranks=2)


def test_random_tokens_are_seeded_valid_and_not_special():
    class T:
        vocab_size = 8
        all_special_ids = [0, 7]

        def get_vocab(self):
            return {"zero": 0, "one": 1, "three": 3, "last": 7, "out": 99}

    assert tool.random_token_samples(
        T(), num_samples=3, max_input_tokens=4, seed=42, model_vocab=8
    ) == tool.random_token_samples(T(), num_samples=3, max_input_tokens=4, seed=42, model_vocab=8)
    assert all(
        0 not in x and 7 not in x
        for x in tool.random_token_samples(T(), num_samples=3, max_input_tokens=4, seed=42, model_vocab=8)
    )


def test_jsonl_requires_enough_distinct_rows(tmp_path):
    class T:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "x"

    p = tmp_path / "d.jsonl"
    p.write_text('{"prompt":"a"}\n')
    with pytest.raises(ValueError, match="not repeated"):
        tool.jsonl_token_samples(p, T(), max_input_tokens=8, required=2)


def test_jsonl_streams_unicode_separator_and_stops_after_required(tmp_path):
    seen = []

    class T:
        def encode(self, text, add_special_tokens=False):
            seen.append(text)
            return [1]

    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"prompt": "a\u2028b"}, ensure_ascii=False) + "\n{broken}\n")
    assert tool.jsonl_token_samples(path, T(), max_input_tokens=8, required=1) == [[1]]
    assert seen == ["a\u2028b"]
    for body in ("[]", "null"):
        path.write_text(body + "\n")
        with pytest.raises(ValueError, match="must be an object"):
            tool.jsonl_token_samples(path, T(), max_input_tokens=8, required=1)


def test_public_server_cli_rejects_internal_export_flag():
    import argparse
    from lightllm.server.api_cli import add_cli_args

    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--export_fp8kv_calibration"])


def _own(output):
    from types import SimpleNamespace

    return SimpleNamespace(
        dataset=None,
        num_samples=2,
        max_input_tokens=4,
        max_new_tokens=2,
        concurrency=1,
        seed=42,
        output=output,
        overwrite=False,
        startup_timeout=1,
        request_timeout=1,
        drain_timeout=1,
    )


def _fake_args():
    from types import SimpleNamespace

    return SimpleNamespace(
        model_dir="unused",
        port=29999,
        tp=1,
        dp=1,
        mtp_mode=None,
        llm_prefill_att_backend=["fa3"],
    )


@pytest.mark.parametrize("fail", [False, True])
def test_main_uses_real_http_concurrency_and_cleans_child(tmp_path, monkeypatch, fail):
    """Exercise the tool's HTTP client and concurrency rather than mocking I/O."""
    import subprocess
    import sys
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        active = maximum = 0
        lock = threading.Lock()

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            if self.path.endswith("/generate"):
                with self.lock:
                    type(self).active += 1
                    type(self).maximum = max(type(self).maximum, type(self).active)
                time.sleep(0.05)
                with self.lock:
                    type(self).active -= 1
                if fail:
                    self.send_error(500, "calibration failed")
                    return
                result = {"generated_text": "x", "count_output_tokens": 1, "finish_reason": "length"}
            elif self.path.endswith("/snapshot"):
                result = {"success": True, "http_pending": 0, "ranks": [_snap(0)]}
            elif self.path.endswith("/status"):
                result = {"success": True, "http_pending": 0, "ranks": [{"idle": True}]}
            else:
                result = {"success": True, "http_pending": 0, "ranks": [{"idle": True}]}
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "real-http.json"
    own = _own(output)
    own.num_samples = 4
    own.concurrency = 2
    args = _fake_args()
    args.port = server.server_port
    monkeypatch.setattr(tool, "_parse", lambda: (own, []))
    monkeypatch.setattr(tool, "_service_args", lambda argv, job, calibration_target="kv": args)
    monkeypatch.setattr(
        tool,
        "_load_tokenizer",
        lambda _: (
            type("T", (), {"all_special_ids": [0], "get_vocab": lambda self: {"a": 1, "b": 2, "c": 3}})(),
            4,
            "Fake",
        ),
    )
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    monkeypatch.setattr(tool, "_spawn_service", lambda args, state, log: proc)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="HTTP 500"):
                tool.main()
            assert not output.exists() and not output.with_name(output.name + ".report.json").exists()
        else:
            assert tool.main() == 0
            assert Handler.maximum >= 2
            assert output.exists() and output.with_name(output.name + ".report.json").exists()
            assert json.loads(output.with_name(output.name + ".report.json").read_text())["output_tokens"] == 4
        assert proc.poll() is not None
    finally:
        server.shutdown()
        server.server_close()
        tool._terminate(proc)


def test_service_args_preserves_valid_intent_then_uses_normal_kv(monkeypatch):
    args = tool._service_args(
        [
            "--model_dir",
            "fake-model",
            "--llm_prefill_att_backend",
            "fa3",
            "--llm_kv_type",
            "fp8kv_sph",
            "--port=28991",
        ],
        "job",
    )
    assert args.export_fp8kv_calibration and args.calibration_job_id == "job"
    assert args.llm_kv_type == "None" and args.enable_rl is False
    assert args.health_monitor is False and args.httpserver_workers == 1
    with pytest.raises(ValueError, match="conflicts"):
        tool._service_args(
            ["--model_dir", "fake-model", "--llm_prefill_att_backend", "fa3", "--llm_kv_type", "fp8kv_spt"], "job"
        )


def test_terminate_kills_term_ignoring_owned_group():
    import subprocess
    import sys

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)",
        ],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "ready"
    tool._terminate(proc, timeout=0.05)
    assert proc.poll() is not None


def test_terminate_kills_child_when_parent_exits(tmp_path):
    import os
    import subprocess
    import sys

    child_pid = tmp_path / "child.pid"
    code = (
        "import os,signal,time,sys; p=os.fork(); "
        "(signal.signal(signal.SIGTERM, signal.SIG_IGN), open(sys.argv[1],'w').write(str(os.getpid())), time.sleep(60)) "
        "if p==0 else os._exit(0)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(child_pid)], start_new_session=True)
    proc.wait(timeout=2)
    for _ in range(40):
        if child_pid.exists():
            break
        __import__("time").sleep(0.01)
    pid = int(child_pid.read_text())
    tool._terminate(proc, timeout=0.05)
    state = open(f"/proc/{pid}/stat").read().split()[2] if os.path.exists(f"/proc/{pid}/stat") else "gone"
    assert state in {"Z", "gone"}


def test_private_route_rejects_ordinary_service_and_wrong_job(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import lightllm.server.api_http as api_http
    import lightllm.server.api_http_calibration as route

    class Request:
        def __init__(self, body):
            self.body = body

        async def json(self):
            return self.body

    class Metric:
        def counter_inc(self, *args):
            pass

    monkeypatch.setattr(
        api_http,
        "g_objs",
        SimpleNamespace(
            args=SimpleNamespace(export_fp8kv_calibration=False, calibration_job_id="job"), metric_client=Metric()
        ),
    )
    assert asyncio.run(route.calibration_operation("status", Request({"job_id": "job"}))).status_code == 404
    api_http.g_objs.args.export_fp8kv_calibration = True
    assert asyncio.run(route.calibration_operation("status", Request({"job_id": "wrong"}))).status_code == 403
