"""Exercise the four independent-master launcher with CPU protocol peers only."""

import asyncio
import base64
import os
from pathlib import Path
import pickle
import signal
import shlex
import argparse
import socket
import subprocess
import sys
import time

import httpx
import psutil
import pytest

from lightllm.server.core.objs.sampling_params import JSON_SCHEMA_MAX_LENGTH
from lightllm.utils.net_utils import get_hostname_ip

from unit_tests.server.pd_master_test_utils import (
    assert_processes_exit,
    create_tiny_tokenizer,
)
from unit_tests.server.test_pd_master_multiprocess_e2e import FakePDPeers


def _free_ports():
    """Reserve a free contiguous backend block and two separate frontend ports."""
    while True:
        listeners = []
        try:
            for _ in range(3):
                listener = socket.socket()
                listeners.append(listener)
                listener.bind(("0.0.0.0", 0))
            front, config, base = [listener.getsockname()[1] for listener in listeners]
            if base > 65531 or any(base <= port <= base + 3 for port in (front, config)):
                continue
            for port in range(base + 1, base + 4):
                listener = socket.socket()
                listeners.append(listener)
                listener.bind(("0.0.0.0", port))
            return front, config, base
        except OSError:
            continue
        finally:
            for listener in listeners:
                listener.close()


@pytest.mark.skipif(JSON_SCHEMA_MAX_LENGTH != 65536, reason="Run with LIGHTLLM_JSON_SCHEMA_MAX_LENGTH=65536")
def test_four_independent_masters_register_forward_and_shut_down(tmp_path):
    root = Path(__file__).resolve().parents[2]
    model = create_tiny_tokenizer(tmp_path / "model")
    front, config, base = _free_ports()
    master_host = get_hostname_ip()
    env = os.environ.copy()
    env.update(
        CPU_SOURCE=str(root),
        CPU_MODEL_DIR=str(model),
        CPU_EXTRA_PYTHONPATH=env.get("PYTHONPATH", ""),
        PYTHON_BIN=sys.executable,
        RUN_DIR=str(tmp_path / "run"),
        CONFIG_SERVER_HOST="127.0.0.1",
        CONFIG_SERVER_PORT=str(config),
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    processes = []
    descendants = []

    async def run():
        async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
            deadline = time.monotonic() + 90
            workers = []
            while time.monotonic() < deadline:
                assert all(process.poll() is None for process in processes), "Independent control plane exited during startup"
                try:
                    response = await client.get(f"http://127.0.0.1:{config}/registered_objects")
                    response.raise_for_status()
                    # This config server is created by this test on its private port.
                    registered = pickle.loads(base64.b64decode(response.json()["data"]))
                    if len(registered) == 4:
                        workers = [
                            {"node_id": node.node_id, "host_ip_port": f"{master_host}:{port}"}
                            for port, node in sorted(
                                (int(node.host_ip_port.rsplit(":", 1)[1]), node) for node in registered.values()
                            )
                        ]
                        break
                except (httpx.HTTPError, KeyError):
                    pass
                await asyncio.sleep(0.1)
            assert len(workers) == 4, "Four independent masters did not register"
            assert len({worker["node_id"] for worker in workers}) == 4
            assert {int(worker["host_ip_port"].rsplit(":", 1)[1]) for worker in workers} == set(range(base, base + 4))
            for worker in workers:
                response = await client.get("http://" + worker["host_ip_port"] + "/get_server_info")
                info = response.json()
                assert info["httpserver_workers"] == 1
                assert info["select_p_d_node_strategy"] == "cache_aware"
                assert info["config_server_port"] == config

            async with FakePDPeers(workers) as peers:

                async def generate(endpoint):
                    response = await client.post(
                        endpoint + "/generate",
                        json={"inputs": "hello world", "parameters": {"max_new_tokens": 2}},
                        timeout=15,
                    )
                    assert response.status_code == 200
                    assert response.json()["generated_text"] == ["hello world"]

                # All four public masters must work independently; each request ID
                # remains unique despite separate request-id generator processes.
                await asyncio.gather(*(generate("http://" + worker["host_ip_port"]) for worker in workers))
                assert len(peers.requests) == 4
                assert {owner for owner, _ in peers.requests} == {worker["node_id"] for worker in workers}
                assert len({request_id for _, request_id in peers.requests}) == 4

    with (tmp_path / "control.log").open("wb") as log:
        try:
            launcher = str(root / "test/start_scripts/glm53/glm53_pd_2p2d.sh")
            for role in ["config", "master0", "master1", "master2", "master3"]:
                child_env = dict(env)
                if role.startswith("master"):
                    child_env["MASTER_PORT"] = str(base + int(role[-1]))
                processes.append(subprocess.Popen(
                    ["bash", launcher, role], env=child_env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                ))
            asyncio.run(run())
        finally:
            for process in processes:
                if process.poll() is None:
                    descendants += psutil.Process(process.pid).children(recursive=True)
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            assert_processes_exit(descendants)


def test_control_launcher_refuses_an_occupied_port_without_writing_runtime(tmp_path):
    root = Path(__file__).resolve().parents[2]
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        env = os.environ.copy()
        env.update(
            PYTHON_BIN=sys.executable, MASTER_PORT=str(listener.getsockname()[1]), RUN_DIR=str(tmp_path / "run"),
            CONFIG_SERVER_HOST="127.0.0.1", CPU_SOURCE=str(root)
        )
        result = subprocess.run(
            ["bash", str(root / "test/start_scripts/glm53/glm53_pd_2p2d.sh"), "master"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "no service was started" in result.stderr
        assert not (tmp_path / "run").exists()
        listener.listen()


@pytest.mark.parametrize(
    "role", ["config", "master0", "master1", "master2", "master3", "master", "p0", "p1", "d0", "d1"]
)
def test_launch_dry_run_matches_current_branch_cli_and_creates_no_files(tmp_path, role):
    from lightllm.server.api_cli import add_cli_args

    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        CPU_SOURCE=str(tmp_path / "cpu source"), GPU_SOURCE=str(tmp_path / "gpu source"), RUN_DIR=str(tmp_path / "run"),
        CONFIG_SERVER_HOST="config-service.test", MODEL_DIR=str(tmp_path / "custom model")
    )
    env.pop("CPU_MODEL_DIR", None)
    result = subprocess.run(
        ["bash", str(root / "test/start_scripts/glm53/glm53_pd_2p2d.sh"), role, "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    command = next(line[8:] for line in result.stdout.splitlines() if line.startswith("command="))
    tokens = shlex.split(command)
    parser = add_cli_args(argparse.ArgumentParser())
    args = parser.parse_args(tokens[tokens.index("-m") + 2 :])
    if role.startswith("master"):
        assert args.run_mode == "pd_master" and args.httpserver_workers == 1
        assert args.select_p_d_node_strategy == "cache_aware"
        assert args.port == 60011
        assert args.model_dir == str(tmp_path / "custom model")
        assert args.config_server_host and args.config_server_port == 60088
    elif role != "config":
        assert args.node_rank == int(role[-1]) and args.nnodes == 2
        assert args.use_config_server_to_init_nccl and args.nccl_host == "config-service.test"
        assert args.tp == args.dp == 16
        assert args.model_dir == str(tmp_path / "custom model")
        assert args.mtp_draft_model_dir == [str(tmp_path / "custom model")]
        assert args.llm_kv_type == "fp8kv_dsa" and args.max_req_total_len == 1048576
        assert args.ssm_state_mode == "replay" and args.ssm_replay_config["cache_len"] == 8
        assert args.config_server_host and args.config_server_port == 60088
    else:
        assert args.run_mode == "config_server"
    assert not (tmp_path / "run").exists()
