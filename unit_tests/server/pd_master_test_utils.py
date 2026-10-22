"""Helpers for real CPU-only PD master integration tests."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

import psutil
import pytest

from lightllm.utils.start_utils import kill_recursive


def get_test_haproxy_binary():
    try:
        from ray_haproxy import get_haproxy_binary
    except ImportError:
        pytest.skip("Install ray-haproxy to run native proxy integration tests")
    return get_haproxy_binary()


def create_tiny_tokenizer(model_dir: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]")
    fast_tokenizer.save_pretrained(model_dir)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 128,
                "vocab_size": 4,
                "eos_token_id": 1,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "generation_config.json").write_text('{"eos_token_id":1}', encoding="utf-8")
    return model_dir


@dataclass
class RunningPDMaster:
    process: subprocess.Popen
    public_url: str
    workers: list[dict]
    opener: object
    log_path: Path

    def get_json(self, path, base_url=None):
        with self.opener.open((base_url or self.public_url) + path, timeout=3) as response:
            return json.load(response)

    def descendants(self):
        try:
            return psutil.Process(self.process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            return []

    def haproxy_runtime_dir(self):
        match = re.search(r"PD master HAProxy configuration file (.+)/haproxy.cfg", self.log_path.read_text())
        assert match is not None, self.log_path.read_text()
        return Path(match.group(1))


def assert_processes_exit(processes, timeout=8):
    def is_alive(process):
        try:
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    deadline = time.monotonic() + timeout
    while True:
        alive = [process for process in processes if is_alive(process)]
        if not alive:
            return
        assert time.monotonic() < deadline, f"Processes survived shutdown: {[process.pid for process in alive]}"
        time.sleep(0.1)


def discover_test_workers(opener, public_url, timeout=10):
    """Build local worker identities from the ports published by the launcher."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with opener.open(public_url + "/get_server_info", timeout=3) as response:
                public_info = json.load(response)
            break
        except (HTTPError, OSError, URLError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)
    host = public_url.split("://", 1)[1].rsplit(":", 1)[0]
    single_worker = public_info["httpserver_workers"] == 1
    workers = []
    for worker_port in public_info["pd_master_multi_ports"]:
        host_ip_port = f"{host}:{worker_port}"
        workers.append(
            {
                "node_id": public_info["pd_node_id"] if single_worker else worker_port,
                "host_ip_port": host_ip_port,
                "port": worker_port,
            }
        )
    return workers


@contextmanager
def running_pd_master(runtime_dir: Path, *, extra_args=(), wait_for_startup=True, launcher_script=None):
    """Run real metric/scheduler/API workers and HAProxy without model weights."""
    get_test_haproxy_binary()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    model_dir = create_tiny_tokenizer(runtime_dir / "tiny-tokenizer")
    with socket.socket() as listener:
        listener.bind(("", 0))
        public_port = listener.getsockname()[1]
    command = [sys.executable, "-m", "lightllm.server.api_server"]
    if launcher_script is not None:
        command = [sys.executable, str(launcher_script)]
    command.extend(
        [
            "--run_mode",
            "pd_master",
            "--host",
            "0.0.0.0",
            "--port",
            str(public_port),
            "--httpserver_workers",
            "2",
            "--model_dir",
            str(model_dir),
            "--max_req_total_len",
            "128",
        ]
    )
    command.extend(extra_args)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    opener = build_opener(ProxyHandler({}))
    public_url = f"http://127.0.0.1:{public_port}"
    log_path = runtime_dir / "launcher.log"
    process = None
    with log_path.open("wb") as log:
        try:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if not wait_for_startup:
                yield RunningPDMaster(process, public_url, [], opener, log_path)
                return
            deadline = time.monotonic() + 90
            while True:
                assert process.poll() is None, log_path.read_text()
                try:
                    with opener.open(public_url + "/liveness", timeout=0.5) as response:
                        assert response.status == 200
                    break
                except (OSError, URLError):
                    if time.monotonic() >= deadline:
                        pytest.fail("PD master launcher did not become live:\n" + log_path.read_text())
                    time.sleep(0.1)
            workers = discover_test_workers(opener, public_url)
            yield RunningPDMaster(process, public_url, workers, opener, log_path)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        kill_recursive(process)
                process.wait(timeout=5)
