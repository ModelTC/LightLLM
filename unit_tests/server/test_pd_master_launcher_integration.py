"""Exercise real PD master startup, supervision, and cleanup."""

import json
from pathlib import Path
import time
from urllib.error import HTTPError

import pytest

from unit_tests.server.pd_master_test_utils import assert_processes_exit, running_pd_master


@pytest.mark.parametrize(
    "failed_process",
    [
        "api_server",
        "pd_node_stats_server",
        "prompt_cache_tree_server",
        "req_id_generator_server",
        "metric_manager",
        "haproxy",
    ],
)
def test_real_launcher_discovers_workers_before_pd_readiness_and_fails_as_a_group(tmp_path, failed_process):
    with running_pd_master(tmp_path) as master:
        assert len({worker["node_id"] for worker in master.workers}) == 2
        assert len({worker["port"] for worker in master.workers}) == 2
        public_port = int(master.public_url.rsplit(":", 1)[1])
        for endpoint in master.workers:
            info = master.get_json("/get_server_info", "http://" + endpoint["host_ip_port"])
            assert info["port"] == public_port
            assert info["pd_node_id"] == endpoint["node_id"]
            assert info["httpserver_workers"] == len(master.workers)
            assert set(info["pd_master_multi_ports"]) == {
                worker["port"] for worker in master.workers
            }
            assert info["config_server_host"] is None
            assert info["config_server_port"] is None
            assert "pd_master_ipc_name" not in info
            assert "metric_port" not in info

        deadline = time.monotonic() + 15
        while True:
            try:
                master.get_json("/readiness")
            except HTTPError as readiness:
                assert readiness.code == 503
                assert json.load(readiness) == {"status": "not ready"}
                break
            assert time.monotonic() < deadline, "Public readiness checks did not converge"
            time.sleep(0.1)
        with master.opener.open(master.public_url + "/metrics", timeout=3) as response:
            assert b"lightllm_request_count" in response.read()

        runtime = master.haproxy_runtime_dir()
        assert runtime.is_dir()
        descendants = master.descendants()
        if failed_process == "haproxy":
            candidates = [
                process
                for process in descendants
                if "haproxy" in process.name() and process.ppid() == master.process.pid
            ]
        else:
            candidates = [process for process in descendants if "::" + failed_process in process.name()]
        assert candidates, [(process.pid, process.name()) for process in descendants]
        candidates[0].kill()
        assert master.process.wait(timeout=20) != 0
        assert_processes_exit(descendants)
        assert not runtime.exists()


@pytest.mark.parametrize(
    "failure_stage",
    ["backend_start", "proxy_start"],
)
def test_partial_startup_failure_stops_every_started_process_and_removes_runtime(tmp_path, failure_stage):
    launcher_script = Path(__file__).with_name("pd_master_fault_launcher.py")
    observations = tmp_path / "cleanup.json"
    with running_pd_master(
        tmp_path,
        launcher_script=launcher_script,
        wait_for_startup=False,
        extra_args=("--failure-stage", failure_stage, "--cleanup-observations", str(observations)),
    ) as master:
        descendants = {}
        deadline = time.monotonic() + 60
        while master.process.poll() is None:
            descendants.update({process.pid: process for process in master.descendants()})
            assert time.monotonic() < deadline, master.log_path.read_text()
            time.sleep(0.05)
        assert master.process.returncode != 0
        assert "injected" in master.log_path.read_text() or failure_stage == "backend_start"
        assert_processes_exit(descendants.values())
        cleanup = json.loads(observations.read_text())
        assert all(not Path(runtime).exists() for runtime in cleanup["runtime_paths"])
        assert cleanup["runtime_before_stop"] == [True]
        assert cleanup["runtime_after_stop"] == [True]
        assert cleanup["backends_stopped"] is True
