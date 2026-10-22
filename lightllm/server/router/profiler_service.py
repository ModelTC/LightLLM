import threading

import rpyc

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class RouterProfilerCmdQueue:
    def __init__(self):
        self.cmds = []
        self.lock = threading.Lock()

    def append(self, cmd: str):
        with self.lock:
            self.cmds.append(cmd)
        return

    def pop(self):
        with self.lock:
            if not self.cmds:
                return None
            return self.cmds.pop(0)


class RouterProfilerService(rpyc.Service):
    def __init__(self, profiler_cmd_queue: RouterProfilerCmdQueue):
        super().__init__()
        self.profiler_cmd_queue = profiler_cmd_queue

    def exposed_profiler_cmd(self, cmd: str):
        cmd = str(cmd)
        if cmd not in ("start", "stop"):
            raise ValueError(f"invalid profiler cmd: {cmd!r}, expected 'start' or 'stop'")
        self.profiler_cmd_queue.append(cmd)
        return


def start_router_profiler_server(args, profiler_cmd_queue: RouterProfilerCmdQueue):
    if not args.enable_profiling:
        return None, None

    from rpyc.utils.server import ThreadedServer
    import lightllm.utils.rpyc_fix_utils as _
    from lightllm.utils.shm_port_args import get_shm_port_args

    router_profiler_port = get_shm_port_args().router_profiler_port
    # the only client is the http server on the same host and commands are plain strings,
    # so listen on loopback only and do not enable pickle.
    server = ThreadedServer(
        RouterProfilerService(profiler_cmd_queue),
        hostname="127.0.0.1",
        port=router_profiler_port,
    )
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    logger.info(f"router profiler rpyc server started on port {router_profiler_port}")
    return server, thread
