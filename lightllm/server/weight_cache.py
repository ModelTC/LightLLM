"""Preload immutable weights, then retain them across serving-worker restarts.

Uses the serving CLI unchanged: select this module in the checked-in launcher.
It builds weights through the registered model classes, without KV, communication
groups, tokenizer state or graphs. Each GPU has one independent owning process.
"""

import argparse
import json
import os
import signal
import socket
import struct
import threading
import time
import uuid
from dataclasses import asdict
from multiprocessing.connection import Client, Listener
from pathlib import Path

import psutil
import torch
import torch.multiprocessing as mp
from filelock import FileLock
from transformers.configuration_utils import PretrainedConfig

from lightllm.common.weight_cache import (
    cache_authkey,
    cache_socket_path,
    export_weight_state,
    preload_weights,
    weight_cache_key,
)
from lightllm.server.api_cli import add_cli_args
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.dist_utils import set_distributed_rank_env, set_current_device_id
from lightllm.utils.envs_utils import set_env_start_args, get_unique_server_name
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import install_fatal_thread_excepthook, is_process_active
from lightllm.utils.service_shm_cleanup import ServiceShmCleanup

logger = init_logger(__name__)


def load_weight_models(args):
    from lightllm.models import get_model, get_draft_model_class

    kvargs = dict(
        weight_dir=args.model_dir,
        run_mode=args.run_mode,
        max_total_token_num=1,
        load_way=args.load_way,
        quant_type=args.quant_type,
        quant_cfg=args.quant_cfg,
        expert_dtype=args.expert_dtype,
        disable_cudagraph=True,
    )
    config, _ = PretrainedConfig.get_config_dict(args.model_dir)
    with preload_weights():
        target, _ = get_model(config, dict(kvargs))
        models = {"target": target}
        if args.mtp_mode is not None:
            if args.mtp_mode != "dspark" or not args.mtp_draft_model_dir or len(args.mtp_draft_model_dir) != 1:
                raise ValueError("Resident weights require one DSpark checkpoint when MTP is enabled")
            draft_dir = args.mtp_draft_model_dir[0]
            draft_config, _ = PretrainedConfig.get_config_dict(draft_dir)
            draft_class = get_draft_model_class(draft_config, args.mtp_mode)
            models["draft"] = draft_class(
                dict(
                    **{**kvargs, "weight_dir": draft_dir, "run_mode": "normal"},
                    main_model=target,
                    mtp_previous_draft_models=[],
                )
            )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return models


class WeightOwner:
    def __init__(self, args, models, namespace):
        self.models = models
        self.namespace = namespace
        self.device = torch.cuda.current_device()
        self.key = weight_cache_key(args)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.clients = {}
        self.host_tables = {
            layer.engram_embed_._host_shm.name: layer.engram_embed_._host_shm.size
            for layer in models["target"].trans_layers_weight
            if layer.has_engram
        }

    def request_stop(self):
        with self.lock:
            if self.clients:
                return False
            self.stop.set()
            return True

    def handle(self, connection):
        torch.cuda.set_device(self.device)
        acquired = False
        try:
            request = connection.recv()
            if request["op"] == "status":
                with self.lock:
                    connection.send(
                        dict(
                            status="ok",
                            pid=os.getpid(),
                            clients=list(self.clients.values()),
                            memory_bytes=torch.cuda.memory_allocated(),
                            host_namespace=self.namespace,
                            host_tables=self.host_tables,
                            key=self.key,
                        )
                    )
            elif request["op"] == "stop":
                stopped = self.request_stop()
                connection.send(dict(status="ok" if stopped else "busy"))
            elif request["op"] == "acquire":
                peer_socket = socket.socket(fileno=os.dup(connection.fileno()))
                with peer_socket:
                    pid, uid, _ = struct.unpack("3i", peer_socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if uid != os.getuid() or pid <= 0:
                    raise PermissionError("Weight leases require the same user and a visible peer PID")
                peer = psutil.Process(pid)
                with self.lock:
                    if request["key"] != self.key or self.stop.is_set():
                        connection.send(
                            dict(
                                status="error",
                                error=(
                                    "checkpoint, GPU, rank or weight-layout fingerprint differs: "
                                    f"owner={self.key}, requested={request['key']}"
                                ),
                            )
                        )
                        return
                    self.clients[id(connection)] = pid
                    acquired = True
                connection.send(
                    dict(
                        status="ok",
                        pid=os.getpid(),
                        host_namespace=self.namespace,
                        memory_bytes=torch.cuda.memory_allocated(),
                        state=export_weight_state(self.models),
                    )
                )
                try:
                    connection.recv()
                except (EOFError, OSError):
                    pass
            else:
                raise ValueError(f"Unknown weight cache operation: {request['op']}")
        except (EOFError, OSError, psutil.NoSuchProcess):
            pass
        except Exception:
            logger.exception("Weight cache request failed")
            raise
        finally:
            connection.close()
            if acquired:
                # Even a partial send can leave CUDA IPC views in a client. EOF
                # alone does not prove its CUDA context has completed cleanup.
                try:
                    while peer.is_running() and is_process_active(peer.pid):
                        time.sleep(0.1)
                except psutil.NoSuchProcess:
                    pass
                with self.lock:
                    del self.clients[id(connection)]

    def serve(self, path, authkey):
        with Listener(path, family="AF_UNIX", authkey=authkey) as listener:
            listener._listener._socket.settimeout(1)
            os.chmod(path, 0o600)
            logger.info(
                f"resident_weights_ready pid={os.getpid()} socket={path} GPU_bytes={torch.cuda.memory_allocated()}"
            )
            while not self.stop.is_set():
                try:
                    connection = listener.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self.handle, args=(connection,), daemon=True).start()


def _run_owner(args, local_rank, namespace, authkey):
    import setproctitle

    install_fatal_thread_excepthook()

    rank = args.node_rank * (args.tp // args.nnodes) + local_rank
    set_distributed_rank_env(dict(world_size=args.tp, rank_id=rank, dp_size=args.dp, args=args))
    set_current_device_id(local_rank)
    torch.cuda.set_device(local_rank)
    set_env_start_args(asdict(args))
    os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = namespace
    get_unique_server_name.cache_clear()
    setproctitle.setproctitle(f"lightllm::weight_owner:RANK{rank}")
    models = load_weight_models(args)
    owner = WeightOwner(args, models, namespace)

    def stop(_signal, _frame):
        if not owner.request_stop():
            logger.warning("Weight owner remains active: serving workers still hold leases")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, stop)
    owner.serve(cache_socket_path(args.weight_cache_dir), authkey)


def control_owners(directory, operation):
    authkey = cache_authkey(directory)
    paths = sorted(Path(directory).glob("*.sock"))
    if not paths:
        raise RuntimeError(f"No resident weight owners found in {directory}")

    def request(path, op):
        with Client(str(path), family="AF_UNIX", authkey=authkey) as connection:
            connection.send(dict(op=op))
            return connection.recv()

    with FileLock(str(Path(directory) / "control.lock")):
        statuses = {str(path): request(path, "status") for path in paths}
        if operation == "stop":
            if any(status["clients"] for status in statuses.values()):
                raise RuntimeError(f"Stop serving workers before releasing resident weights: {statuses}")
            for path in paths:
                if request(path, "stop")["status"] != "ok":
                    raise RuntimeError(f"Weight owner still has active consumers: {path}")
            processes = []
            for status in statuses.values():
                try:
                    processes.append(psutil.Process(status["pid"]))
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(processes, timeout=180)
            remaining = [
                process.pid for process in processes if process.is_running() and is_process_active(process.pid)
            ]
            if remaining:
                raise RuntimeError(f"Weight owners are still releasing resources: {remaining}")
            for namespace in {status["host_namespace"] for status in statuses.values()}:
                ServiceShmCleanup.cleanup_posix_shm(namespace)
            for status in statuses.values():
                status["owner_exited"] = True
    return statuses


def main():
    parser = add_cli_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--weight_cache_action", choices=("serve", "status", "stop"), default="serve")
    options = vars(parser.parse_args())
    action = options.pop("weight_cache_action")
    args = StartArgs(**options)
    if not args.weight_cache_dir:
        parser.error("--weight_cache_dir is required")
    if action != "serve":
        print(json.dumps(control_owners(args.weight_cache_dir, action), indent=2))
        return
    if not args.model_dir:
        parser.error("--model_dir is required for serve")
    if args.data_type is None:
        from lightllm.utils.config_utils import get_dtype

        args.data_type = get_dtype(args.model_dir)
    authkey = cache_authkey(args.weight_cache_dir, create=True)
    if args.nnodes < 1 or args.tp < 1 or args.dp < 1 or args.tp % args.dp or not 0 <= args.node_rank < args.nnodes:
        raise ValueError("Weight owner requires a valid TP/DP/node topology")
    count = args.tp // args.nnodes
    if args.tp % args.nnodes or count > torch.cuda.device_count():
        raise ValueError("Weight owner topology does not fit the visible GPUs")
    with FileLock(str(Path(args.weight_cache_dir) / "owner.lock"), timeout=0):
        if list(Path(args.weight_cache_dir).glob("*.sock")):
            raise RuntimeError("Existing owner sockets found; inspect status/stop before starting another owner")
        namespace = f"resident_{uuid.uuid4().hex[:16]}_{args.node_rank}"
        logger.info(f"resident_weights_loading namespace={namespace}")
        context = mp.get_context("spawn")
        processes = [context.Process(target=_run_owner, args=(args, rank, namespace, authkey)) for rank in range(count)]
        for process in processes:
            process.start()
        (Path(args.weight_cache_dir) / "owner.json").write_text(
            json.dumps(
                dict(
                    namespace=namespace,
                    pid=os.getpid(),
                    workers=[process.pid for process in processes],
                )
            )
        )
        stop_requested = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
        signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
        signal.signal(signal.SIGHUP, lambda *_: stop_requested.set())
        failed = False
        while any(process.is_alive() for process in processes):
            failed = failed or any(process.exitcode not in (None, 0) for process in processes)
            if stop_requested.is_set() or failed:
                with FileLock(str(Path(args.weight_cache_dir) / "control.lock")):
                    live_clients = False
                    for path in Path(args.weight_cache_dir).glob("*.sock"):
                        try:
                            with Client(str(path), family="AF_UNIX", authkey=authkey) as connection:
                                connection.send(dict(op="status"))
                                live_clients |= bool(connection.recv()["clients"])
                        except (ConnectionError, FileNotFoundError, EOFError):
                            # Failed children can leave socket names. Keep their
                            # CPU namespace for explicit inspection after failure.
                            pass
                    if live_clients:
                        logger.warning("Owner stop deferred until serving workers exit")
                        stop_requested.clear()
                    else:
                        for process in processes:
                            if process.is_alive():
                                process.terminate()
            for process in processes:
                process.join(timeout=0.2)
            if failed:
                time.sleep(1)
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("A weight owner failed; inspect its logs and retained resources before restarting")
        ServiceShmCleanup.cleanup_posix_shm(namespace)
        logger.info(f"resident_weights_released namespace={namespace}")


if __name__ == "__main__":
    main()
