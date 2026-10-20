import argparse
import json
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.httpserver_for_pd_master import multiprocess
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.net_utils import get_pd_master_unix_socket_address


def _args(**kwargs):
    options = dict(
        run_mode="pd_master",
        host="10.0.0.1",
        port=8000,
        httpserver_workers=2,
    )
    options.update(kwargs)
    return StartArgs(**options)


def _worker_ports():
    return [8100, 8101]


def _ports(**kwargs):
    values = dict(
        port=8000,
        metric_port=8104,
        pd_master_multi_ports=[8100, 8101],
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_workers_use_the_shared_service_ipc_address(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "shared-service")
    get_unique_server_name.cache_clear()
    launcher_address = get_pd_master_unix_socket_address("pd_node_stats")

    worker_address = get_pd_master_unix_socket_address("pd_node_stats")

    assert worker_address == launcher_address
    get_unique_server_name.cache_clear()


@pytest.mark.parametrize("exited_worker", [None, 1])
def test_backend_launch_tracks_processes_and_passes_worker_index(monkeypatch, exited_worker):
    children = [Mock(pid=2101), Mock(pid=2102)]
    managed = [object(), object()]
    manager = SimpleNamespace(processes=[], process_names={})
    process_factory = Mock(side_effect=children)
    process_context = Mock(Process=process_factory)
    monkeypatch.setattr(
        multiprocess.multiprocessing,
        "get_context",
        Mock(return_value=process_context),
    )
    monkeypatch.setattr(multiprocess, "process_manager", manager)

    def lookup(pid):
        index = [child.pid for child in children].index(pid)
        if index == exited_worker:
            raise multiprocess.psutil.NoSuchProcess(pid)
        return managed[index]

    monkeypatch.setattr(multiprocess.psutil, "Process", lookup)
    if exited_worker is None:
        multiprocess._start_backends(2)
    else:
        with pytest.raises(multiprocess.psutil.NoSuchProcess):
            multiprocess._start_backends(2)

    registered_count = len(children) if exited_worker is None else exited_worker
    assert manager.processes == managed[:registered_count]
    assert manager.process_names == {
        process: f"pd_master_worker_{index}" for index, process in enumerate(managed[:registered_count])
    }
    multiprocess.multiprocessing.get_context.assert_called_once_with("spawn")
    for index, call in enumerate(process_factory.call_args_list):
        assert call.kwargs == {
            "target": multiprocess._run_pd_master_worker,
            "args": (index,),
        }
        children[index].start.assert_called_once_with()


def test_worker_bootstrap_uses_its_assigned_port(monkeypatch):
    import hypercorn.__main__

    args = _args(port=8000, httpserver_workers=2)
    ports = SimpleNamespace(pd_master_multi_ports=[8100, 8101])
    create_ports = Mock(return_value=ports)
    monkeypatch.setenv("LIGHTLLM_PD_MASTER_WORKER_INDEX", "1")
    monkeypatch.setattr(multiprocess, "get_env_start_args", lambda: args)
    monkeypatch.setattr(multiprocess, "get_shm_port_args", create_ports)

    def start_hypercorn(argv):
        assert args.port == 8000
        assert args.pd_node_id == 8101
        assert argv[argv.index("--workers") + 1] == "0"
        assert argv[argv.index("--bind") + 1] == "10.0.0.1:8101"
        assert argv[-1] == "lightllm.server.api_http:app"

    monkeypatch.setattr(hypercorn.__main__, "main", start_hypercorn)
    multiprocess._start_pd_master_worker()
    create_ports.assert_called_once_with()


def test_worker_port_table_preserves_dynamically_allocated_internal_ports(monkeypatch):
    from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
    from lightllm.utils.shm_port_args import ShmPortArgs

    args = _args(port=8000, pd_master_port=8001, config_server_port=8103)
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_pd_worker_ports_" + uuid.uuid4().hex)
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(vars(args)))
    get_unique_server_name.cache_clear()
    get_env_start_args.cache_clear()
    ports = ShmPortArgs(create=True)
    linked = None
    try:
        allocator = Mock(side_effect=[8120, 8104])
        monkeypatch.setattr(ports, "_alloc_free_port", allocator)
        assert ports.router_port == 8120
        assert ports.metric_port == 8104
        assert ports.port == 8000
        assert ports.config_server_port == 8103
        linked = ShmPortArgs()
        assert linked.router_port == 8120
        assert linked.metric_port == 8104
        assert linked.port == 8000
        assert linked.config_server_port == 8103
        assert allocator.call_count == 2
    finally:
        if linked is not None:
            linked.close()
        ports.shm.unlink()
        ports.close()
        get_unique_server_name.cache_clear()
        get_env_start_args.cache_clear()


def test_haproxy_config_streams_without_request_replay():
    config = multiprocess._build_haproxy_config(_args(), _worker_ports())

    assert "balance leastconn" in config
    assert "nbthread 2" in config
    assert "server worker_0 10.0.0.1:8100" in config
    assert "server worker_1 10.0.0.1:8101" in config
    assert "bind 10.0.0.1:8000" in config
    assert "no option http-buffer-request" in config
    assert "option http-server-close" in config
    assert "retries 0" in config
    assert "retry-on none" in config
    assert "timeout client 3600s" in config
    assert "timeout server 3600s" in config
    assert "acl private_pd_channel path -i /pd_register /kv_move_status" in config
    assert "http-request return status 426 if private_pd_channel" in config
    assert "option forwardfor" in config
    assert "registered_objects" not in config
    assert "allocate_global_unique_multimodal_id_range" not in config
    assert "pd_master_config_server" not in config


@pytest.mark.parametrize(
    ("host", "expected_address"),
    [("0.0.0.0", "0.0.0.0:8100"), ("::", "[::]:8100")],
)
def test_haproxy_worker_addresses_follow_the_configured_listen_host(
    host, expected_address
):
    config = multiprocess._build_haproxy_config(_args(host=host), _worker_ports())

    assert f"server worker_0 {expected_address}" in config


@pytest.mark.parametrize("config", [{"config_server_host": "10.0.0.2"}, {"config_server_port": 9000}])
def test_multiworker_rejects_config_server_before_haproxy_lookup(monkeypatch, config):
    lookup = Mock()
    monkeypatch.setitem(
        sys.modules,
        "ray_haproxy",
        SimpleNamespace(get_haproxy_binary=lookup),
    )

    with pytest.raises(ValueError, match="does not yet support"):
        multiprocess.validate_multiprocess_pd_master_args(_args(**config))
    lookup.assert_not_called()


@pytest.mark.parametrize("port", [None, 0, 65536])
def test_multiworker_requires_a_valid_public_port_before_haproxy_lookup(monkeypatch, port):
    lookup = Mock()
    monkeypatch.setitem(
        sys.modules,
        "ray_haproxy",
        SimpleNamespace(get_haproxy_binary=lookup),
    )

    with pytest.raises(ValueError, match="between 1 and 65535"):
        multiprocess.validate_multiprocess_pd_master_args(_args(port=port))
    lookup.assert_not_called()


def test_cli_rejects_the_removed_worker_ports_option(capsys):
    from lightllm.server.api_cli import add_cli_args

    parser = add_cli_args(argparse.ArgumentParser())
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--pd_master_worker_ports", "8100", "8101"])
    assert error.value.code == 2
    assert "unrecognized arguments: --pd_master_worker_ports" in capsys.readouterr().err


def test_cli_rejects_the_removed_multiprocess_ports_option(capsys):
    from lightllm.server.api_cli import add_cli_args

    parser = add_cli_args(argparse.ArgumentParser())
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--pd_master_multi_ports", "8100", "8101"])
    assert error.value.code == 2
    assert "unrecognized arguments: --pd_master_multi_ports" in capsys.readouterr().err


def test_missing_haproxy_package_reports_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray_haproxy", None)
    with pytest.raises(RuntimeError, match=r"pip install ray-haproxy==2\.8\.25"):
        multiprocess.validate_multiprocess_pd_master_args(_args())


def test_haproxy_uses_the_native_package(monkeypatch):
    package = SimpleNamespace(get_haproxy_binary=Mock(return_value="/tmp/native-wheel/haproxy"))
    monkeypatch.setitem(sys.modules, "ray_haproxy", package)
    args = _args()
    assert multiprocess.validate_multiprocess_pd_master_args(args) == "/tmp/native-wheel/haproxy"
    package.get_haproxy_binary.assert_called_once_with()


def test_package_resolution_failure_remains_an_error(monkeypatch):
    error = FileNotFoundError("bundled executable unavailable")
    package = SimpleNamespace(get_haproxy_binary=Mock(side_effect=error))
    monkeypatch.setitem(sys.modules, "ray_haproxy", package)
    with pytest.raises(RuntimeError, match="Cannot locate the packaged HAProxy executable"):
        multiprocess.validate_multiprocess_pd_master_args(_args())


def test_worker_ports_are_allocated_as_one_launcher_owned_list(monkeypatch):
    from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
    from lightllm.utils.shm_port_args import ShmPortArgs

    args = _args()
    monkeypatch.setenv(
        "LIGHTLLM_UNIQUE_SERVICE_NAME_ID",
        "test_pd_master_multi_ports_" + uuid.uuid4().hex,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(vars(args)))
    get_unique_server_name.cache_clear()
    get_env_start_args.cache_clear()
    ports = ShmPortArgs(create=True)
    linked = None
    try:
        allocator = Mock(side_effect=[18100, 18101])
        monkeypatch.setattr(ports, "_alloc_free_port", allocator)

        assert ports.pd_master_multi_ports == [18100, 18101]
        linked = ShmPortArgs()
        assert linked.pd_master_multi_ports == [18100, 18101]
        assert allocator.call_count == 2
    finally:
        if linked is not None:
            linked.close()
        ports.shm.unlink()
        ports.close()
        get_unique_server_name.cache_clear()
        get_env_start_args.cache_clear()


@pytest.mark.parametrize("run_mode", ["pd_master", "prefill", "config_server"])
def test_config_server_port_is_never_allocated_dynamically(monkeypatch, run_mode):
    from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
    from lightllm.utils.shm_port_args import ShmPortArgs

    args = _args(run_mode=run_mode, config_server_port=None)
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_pd_config_port_" + uuid.uuid4().hex)
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(vars(args)))
    get_unique_server_name.cache_clear()
    get_env_start_args.cache_clear()
    ports = ShmPortArgs(create=True)
    try:
        with pytest.raises(RuntimeError, match="config_server_port.*None"):
            _ = ports.config_server_port
    finally:
        ports.shm.unlink()
        ports.close()
        get_unique_server_name.cache_clear()
        get_env_start_args.cache_clear()


@pytest.mark.parametrize("failure_stage", [None, "backend", "proxy"])
def test_launcher_uses_process_manager_before_releasing_runtime_and_ports(monkeypatch, tmp_path, failure_stage):
    args, ports = _args(), _ports(close=Mock())
    proxy = Mock(pid=3103)
    managed_proxy = object()
    runtime_paths, events = [], []
    create_runtime = multiprocess.tempfile.TemporaryDirectory

    def runtime_directory(**kwargs):
        runtime = create_runtime(dir=tmp_path, **kwargs)
        runtime_paths.append(multiprocess.Path(runtime.name))
        return runtime

    def start_backends(worker_count):
        assert worker_count == 2
        if failure_stage == "backend":
            raise RuntimeError("injected backend startup failure")

    def terminate():
        assert runtime_paths[0].is_dir()
        ports.close.assert_not_called()
        events.append("stop")

    manager = SimpleNamespace(
        processes=[],
        process_names={},
        terminate_all_processes=Mock(side_effect=terminate),
        setup_signal_handlers=Mock(),
        supervise_processes=Mock(),
    )
    ports.close.side_effect = lambda: events.append("close ports")
    monkeypatch.setattr(multiprocess, "process_manager", manager)
    monkeypatch.setattr(multiprocess, "get_unique_server_name", lambda: "test-cleanup-order")
    monkeypatch.setattr(multiprocess.tempfile, "TemporaryDirectory", runtime_directory)
    monkeypatch.setattr(multiprocess, "_start_backends", start_backends)
    config_check = Mock()
    monkeypatch.setattr(multiprocess.subprocess, "run", config_check)
    monkeypatch.setattr(multiprocess.psutil, "Process", Mock(return_value=managed_proxy))
    launch_proxy = Mock(return_value=proxy)
    if failure_stage == "proxy":
        launch_proxy.side_effect = OSError("injected proxy startup failure")
    monkeypatch.setattr(multiprocess.subprocess, "Popen", launch_proxy)

    if failure_stage is None:
        multiprocess.start_multiprocess_pd_master(
            args,
            ports=ports,
            haproxy_binary="/bundled/haproxy",
        )
        manager.supervise_processes.assert_called_once_with(proxy)
        assert manager.processes == [managed_proxy]
        assert manager.process_names == {managed_proxy: "pd_master_haproxy"}
    else:
        with pytest.raises((RuntimeError, OSError), match=f"injected {failure_stage} startup failure"):
            multiprocess.start_multiprocess_pd_master(
                args,
                ports=ports,
                haproxy_binary="/bundled/haproxy",
            )
        manager.supervise_processes.assert_not_called()

    manager.terminate_all_processes.assert_called_once_with()
    assert events == ["stop", "close ports"]
    assert not runtime_paths[0].exists()
    ports.close.assert_called_once_with()
    if failure_stage != "backend":
        assert config_check.call_args.args[0][0:3] == ["/bundled/haproxy", "-c", "-f"]


