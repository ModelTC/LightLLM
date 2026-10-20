"""Launch independent PD masters behind a streaming HTTP reverse proxy.

P/D nodes discover and connect to every worker directly. HAProxy only handles
public HTTP requests; generated tokens never cross a Python proxy process.
"""

from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import tempfile

import psutil

from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import (
    get_env_start_args,
    get_unique_server_name,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import format_host_port
from lightllm.utils.shm_port_args import ShmPortArgs, get_shm_port_args
from lightllm.utils.start_utils import process_manager

logger = init_logger(__name__)

MAX_PROXY_THREADS = 64


def validate_multiprocess_pd_master_args(args: StartArgs) -> str:
    """Validate prerequisites before allocating SHM or launching subprocesses."""
    if args.config_server_host is not None or args.config_server_port is not None:
        raise ValueError("Multiprocess PD master does not yet support --config_server_host/--config_server_port")
    if args.port is None or not 1 <= args.port <= 65535:
        raise ValueError("PD master public port must be between 1 and 65535")

    try:
        from ray_haproxy import get_haproxy_binary

        return get_haproxy_binary()
    except ImportError as exc:
        raise RuntimeError(
            "Multiprocess PD master requires the native HAProxy wheel. "
            "Install requirements.txt or run pip install ray-haproxy==2.8.25."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot locate the packaged HAProxy executable: {exc}") from exc


def start_multiprocess_pd_master(
    args: StartArgs,
    ports: ShmPortArgs,
    haproxy_binary: str,
) -> None:
    with ExitStack() as resources:
        resources.callback(ports.close)
        runtime_dir = resources.enter_context(
            tempfile.TemporaryDirectory(prefix=f"lightllm-pd-master-{get_unique_server_name()}-")
        )
        # 启动中途失败时也通过统一进程管理器清理已经注册的子进程。
        resources.callback(process_manager.terminate_all_processes)
        worker_ports = ports.pd_master_multi_ports
        _start_backends(len(worker_ports))

        config_path = Path(runtime_dir) / "haproxy.cfg"
        config_path.write_text(_build_haproxy_config(args, worker_ports), encoding="utf-8")
        logger.info("PD master HAProxy configuration file %s", config_path)
        subprocess.run([haproxy_binary, "-c", "-f", str(config_path)], check=True, timeout=10)
        # -db 使 HAProxy 保持前台运行，便于统一进程管理器直接监督。
        proxy_process = subprocess.Popen(
            [haproxy_binary, "-db", "-f", str(config_path)],
            start_new_session=True,
        )
        managed_proxy = psutil.Process(proxy_process.pid)
        process_manager.processes.append(managed_proxy)
        process_manager.process_names[managed_proxy] = "pd_master_haproxy"
        # launcher 收到 SIGTERM/SIGHUP 后，先让 HAProxy 停止接收新连接并等待
        # 已建立的 SSE 请求结束。HAProxy 退出后，统一进程管理器再清理后端 worker。
        process_manager.setup_signal_handlers(proxy_process, graceful_shutdown_signal=signal.SIGUSR1)
        if args.health_monitor:
            from lightllm.server.health_monitor.manager import start_health_check_process

            process_manager.start_submodule_processes(
                start_funcs=[start_health_check_process],
                start_args=[(args,)],
            )
        process_manager.supervise_processes(proxy_process)


def _build_haproxy_config(args: StartArgs, worker_ports: list[int]) -> str:
    """生成 PD master 多 worker 的 HAProxy 配置。"""
    worker_servers = [
        f"    server worker_{worker_index} {format_host_port(args.host, worker_port)}"
        for worker_index, worker_port in enumerate(worker_ports)
    ]
    worker_count = len(worker_ports)
    listen_address = format_host_port(args.host, args.port)
    proxy_threads = min(args.httpserver_workers, MAX_PROXY_THREADS)

    # 进程级配置和所有 HTTP frontend/backend 共用的连接策略。
    config = [
        "global",
        "    log stdout format raw local0",
        # 让 HAProxy 根据进程的 FD 硬上限自动计算 maxconn，同时限制其最多使用的 FD 数量。
        "    fd-hard-limit 131072",
        f"    nbthread {proxy_threads}",
        "",
        "defaults",
        "    mode http",
        "    log global",
        "    option httplog",
        "    option http-server-close",
        "    option abortonclose",
        "    option forwardfor",
        "    no option http-buffer-request",
        "    retries 0",
        "    retry-on none",
        "    timeout connect 10s",
        "    timeout client 3600s",
        "    timeout server 3600s",
        "    timeout http-request 3600s",
        "    timeout http-keep-alive 10s",
        "",
        # 对外入口：拒绝内部 P/D 通道，健康接口进入聚合 backend。
        "frontend pd_master_public",
        f"    bind {listen_address}",
        "    acl private_pd_channel path -i /pd_register /kv_move_status",
        "    http-request return status 426 if private_pd_channel",
        "    http-request set-header X-Forwarded-Proto http",
        "    use_backend pd_master_liveness if { path /liveness } { method GET POST }",
        "    use_backend pd_master_readiness if { path /readiness } { method GET POST }",
        "    use_backend pd_master_health if { path /health } { method GET HEAD }",
        "    use_backend pd_master_health if { path /healthz } { method GET }",
        "    default_backend pd_master_workers",
        "",
        # 普通请求继续使用全部 worker，不受健康聚合 backend 状态影响。
        "backend pd_master_workers",
        "    balance leastconn",
        *worker_servers,
    ]

    # 每类健康接口使用独立 backend。nbsrv 只有在全部 worker 正常时才满足条件。
    for name, interval, timeout, success_body, failure_body in (
        ("liveness", "6s", "6s", {"status": "ok"}, {"status": "not alive"}),
        ("readiness", "6s", "6s", {"status": "ok"}, {"status": "not ready"}),
        ("health", "10s", "12s", {"message": "Ok"}, {"message": "Error"}),
    ):
        backend_name = f"pd_master_{name}"
        success_json = json.dumps(success_body, separators=(",", ":"))
        failure_json = json.dumps(failure_body, separators=(",", ":"))
        config.extend(
            [
                "",
                f"backend {backend_name}",
                f"    option httpchk GET /{name}",
                "    http-check expect status 200",
                f"    timeout check {timeout}",
                f"    default-server check inter {interval} fall 1 rise 1",
                (
                    "    http-request return status 503 content-type application/json "
                    f"string '{failure_json}' unless {{ nbsrv({backend_name}) eq {worker_count} }}"
                ),
                ("    http-request return status 200 content-type application/json " f"string '{success_json}'"),
                *worker_servers,
            ]
        )

    return "\n".join(config) + "\n"


def _start_pd_master_worker() -> None:
    """Start one worker on the HTTP port assigned by the launcher."""
    from hypercorn.__main__ import main
    from lightllm.server.api_start import _hypercorn_config_args

    args = get_env_start_args()
    worker_index = int(os.environ["LIGHTLLM_PD_MASTER_WORKER_INDEX"])
    ports = get_shm_port_args()
    worker_port = ports.pd_master_multi_ports[worker_index]
    # 所有 worker 共享同一份启动参数，仅节点 ID 需要区分。请求携带该 ID，
    # P/D 节点据此将状态回传给发起请求的 PD master worker。
    args.pd_node_id = worker_port
    # Hypercorn workers=0 runs one ASGI worker in this process. This makes every
    # backend directly supervised and prevents a hidden worker pool respawn.
    main(
        [
            *_hypercorn_config_args(args),
            "--workers",
            "0",
            "--bind",
            format_host_port(args.host, worker_port),
            "--log-level",
            "info",
            "--access-logfile",
            "-",
            "--error-logfile",
            "-",
            "lightllm.server.api_http:app",
        ]
    )


def _run_pd_master_worker(worker_index: int) -> None:
    os.environ["LIGHTLLM_PD_MASTER_WORKER_INDEX"] = str(worker_index)
    _start_pd_master_worker()


def _start_backends(worker_count: int) -> None:
    process_context = multiprocessing.get_context("spawn")
    for worker_index in range(worker_count):
        process = process_context.Process(
            target=_run_pd_master_worker,
            args=(worker_index,),
        )
        process.start()
        managed = psutil.Process(process.pid)
        process_manager.processes.append(managed)
        process_manager.process_names[managed] = f"pd_master_worker_{worker_index}"
