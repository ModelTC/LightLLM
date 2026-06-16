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
        self.profiler_cmd_queue.append(cmd)
        return


def start_router_profiler_server(args, profiler_cmd_queue: RouterProfilerCmdQueue):
    if not args.enable_profiling:
        return None, None

    from rpyc.utils.server import ThreadedServer
    import lightllm.utils.rpyc_fix_utils as _

    server = ThreadedServer(
        RouterProfilerService(profiler_cmd_queue),
        port=args.router_profiler_port,
        protocol_config={"allow_pickle": True},
    )
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    logger.info(f"router profiler rpyc server started on port {args.router_profiler_port}")
    return server, thread
