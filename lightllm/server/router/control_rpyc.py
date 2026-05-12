import threading
import rpyc
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class RouterControlRpcService(rpyc.Service):
    """挂在 master router 进程上的控制面 rpyc service。httpserver 通过 rpyc 同步调用,
    rpyc 线程把请求投递到 router 的 asyncio loop(_control_op_queue),等结果。
    多机情况下,只有 node_rank=0 启动这个 service;slave 通过 router 的 NCCL 广播协同。"""

    def __init__(self, router):
        super().__init__()
        self._router = router

    def exposed_flush_cache(self, request):
        return self._router.submit_control_op("flush_cache", obtain(request))

    def exposed_release_memory_occupation(self, tags):
        return self._router.submit_control_op("release_memory_occupation", obtain(tags))

    def exposed_resume_memory_occupation(self, tags):
        return self._router.submit_control_op("resume_memory_occupation", obtain(tags))

    def exposed_init_weights_update_group(self, request):
        return self._router.submit_control_op("init_weights_update_group", obtain(request))

    def exposed_destroy_weights_update_group(self, request):
        return self._router.submit_control_op("destroy_weights_update_group", obtain(request))

    def exposed_update_weights_from_distributed(self, request):
        return self._router.submit_control_op("update_weights_from_distributed", obtain(request))

    def exposed_update_weights_from_tensor(self, request):
        return self._router.submit_control_op("update_weights_from_tensor", obtain(request))

    def exposed_update_weights_from_ipc(self, request):
        return self._router.submit_control_op("update_weights_from_ipc", obtain(request))


def start_control_rpyc_server(router, port: int) -> None:
    """在 daemon 线程里启动 rpyc ThreadedServer。绑定 127.0.0.1,httpserver 同机调用即可。"""

    def _run():
        try:
            t = ThreadedServer(
                RouterControlRpcService(router),
                hostname="127.0.0.1",
                port=port,
                protocol_config={"allow_pickle": True, "sync_request_timeout": 600},
            )
            logger.info(f"control rpyc server listening on 127.0.0.1:{port}")
            t.start()
        except BaseException as e:
            logger.exception(f"control rpyc server crashed: {e}")

    th = threading.Thread(target=_run, name="control_rpyc_server", daemon=True)
    th.start()
