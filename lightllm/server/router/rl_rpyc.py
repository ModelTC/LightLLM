import concurrent.futures
import queue
import threading
from typing import List, Tuple

import rpyc
from rpyc.utils.classic import obtain

from lightllm.server.io_struct import GeneralHttpToModelRpcReq, GeneralModelToHttpRpcRsp
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class RouterRlOpQueue:
    def __init__(self):
        self._queue: "queue.Queue[Tuple[GeneralHttpToModelRpcReq, concurrent.futures.Future]]" = queue.Queue()

    def submit(self, req: GeneralHttpToModelRpcReq, timeout: float = 300.0) -> GeneralModelToHttpRpcRsp:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((req, fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return GeneralModelToHttpRpcRsp(
                success=False,
                msg=f"rl op {req.func_name} timeout after {timeout}s",
                func_name=req.func_name,
            )

    def pop_all(self) -> List[Tuple[GeneralHttpToModelRpcReq, concurrent.futures.Future]]:
        pairs = []
        while True:
            try:
                pairs.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return pairs


class RouterRlRpcService(rpyc.Service):
    def __init__(self, rl_op_queue: RouterRlOpQueue):
        super().__init__()
        self.rl_op_queue = rl_op_queue

    def exposed_rl_op(self, req: GeneralHttpToModelRpcReq):
        return self.rl_op_queue.submit(obtain(req))


def start_router_rl_rpyc_server(args, rl_op_queue: RouterRlOpQueue):
    if args.node_rank != 0:
        return None, None

    from rpyc.utils.server import ThreadedServer
    import lightllm.utils.rpyc_fix_utils as _
    from lightllm.utils.shm_port_args import get_shm_port_args

    rl_rpyc_port = get_shm_port_args().rl_rpyc_port
    server = ThreadedServer(
        RouterRlRpcService(rl_op_queue),
        hostname="127.0.0.1",
        port=rl_rpyc_port,
        protocol_config={"allow_pickle": True, "sync_request_timeout": 600},
    )
    thread = threading.Thread(target=server.start, name="rl_rpyc_server", daemon=True)
    thread.start()
    logger.info(f"router rl rpyc server started on port {rl_rpyc_port}")
    return server, thread
