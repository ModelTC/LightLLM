import asyncio
import concurrent.futures
import queue
import threading
import rpyc
import torch
import torch.distributed as dist

from typing import List, Tuple
from rpyc.utils.classic import obtain
from lightllm.server.io_struct import RlOpReq, RlOpRsp
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class RouterRlOpHelper(rpyc.Service):
    """Router RL 控制面：rpyc 入口 + 主循环 drain / 多机广播 / 下发 model rpc。

    挂到 ``RouterManager`` 上；``RouterRlOpQueue`` 经 ``_get_rl_op_queue`` 延迟创建。
    """

    def exposed_rl_op(self, req: RlOpReq):
        return self._get_rl_op_queue().submit(obtain(req))

    async def process_rl_ops(self):
        # 从本地 RL 队列取出本轮待处理的 (req, future)。
        # 多机 TP 下只有 master 会跑 rpyc service、真正入队；slave 没有提交入口，
        # 理论上不应依赖本接口拿业务请求（pop 结果恒为空），后续靠 broadcast 对齐 reqs。
        # 这里仍调用是为了统一 master/slave 代码路径，slave 侧等价于 no-op。
        pairs = self._get_rl_op_queue().pop_all()
        reqs: List[RlOpReq] = [req for req, _ in pairs]

        # 多机 TP: master 广播 req；slave 在此处收到同一批 reqs 并跟跑
        if self.is_multinode_tp:
            reqs = self._broadcast_rl_ops_to_other_nodes(reqs)

        for i, req in enumerate(reqs):
            assert isinstance(req, RlOpReq), "rl op request must be RlOpReq"
            try:
                ret = await self._rl_op(req)
            except BaseException as e:
                logger.exception(f"rl_op failed for {req.op_name}: {e}")
                ret = RlOpRsp(success=False, msg=f"rl_op error: {e}", op_name=req.op_name)
            # 多机 TP slave 只跟跑 collective，无权回写 future（仅 master / 单机持有提交方）
            if self.is_multinode_tp_slave:
                continue
            _, fut = pairs[i]
            if not fut.done():
                fut.set_result(ret)

    def _get_rl_op_queue(self) -> "RouterRlOpQueue":
        rl_op_queue = getattr(self, "_rl_op_queue", None)
        if rl_op_queue is None:
            self._rl_op_queue = RouterRlOpQueue()
        return self._rl_op_queue

    def _broadcast_rl_ops_to_other_nodes(self, reqs: List[RlOpReq]):
        req_num = len(reqs)
        if self.node_rank == 0:
            req_nums = [len(reqs)]
            dist.broadcast_object_list(req_nums, src=0, group=self.mulitnode_group)
            req_num = req_nums[0]
            if req_num > 0:
                dist.broadcast_object_list(reqs, src=0, group=self.mulitnode_group)
        else:
            req_nums = [None]
            dist.broadcast_object_list(req_nums, src=0, group=self.mulitnode_group)
            req_num = req_nums[0]
            if req_num > 0:
                reqs = [None for _ in range(req_num)]
                dist.broadcast_object_list(reqs, src=0, group=self.mulitnode_group)
        return reqs

    async def _rl_op(self, req: RlOpReq) -> RlOpRsp:
        rl_op_tasks = []
        for model_rpc_client in self.model_rpc_clients:
            rl_op_tasks.append(model_rpc_client.rl_op(req))
        all_ret = await asyncio.gather(*rl_op_tasks)
        # 优先返回第一个失败结果（带具体错误信息）；全部成功则用第一个
        ret: RlOpRsp = all_ret[0]
        for res in all_ret:
            if not res.success:
                ret = res
                break
        ret.success = all(res.success for res in all_ret)

        if self.is_multinode_tp:
            # True/False -> 1/0；MIN all_reduce：任一节点失败则全体 success=False
            success_flag = torch.tensor([1 if ret.success else 0], dtype=torch.int32, device="cpu")
            dist.all_reduce(success_flag, op=dist.ReduceOp.MIN, group=self.mulitnode_group)
            ret.success = success_flag.item() == 1
        return ret


class RouterRlOpQueue:
    def __init__(self):
        self._queue: "queue.Queue[Tuple[RlOpReq, concurrent.futures.Future]]" = queue.Queue()

    def submit(self, req: RlOpReq, timeout: float = 300.0) -> RlOpRsp:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((req, fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return RlOpRsp(
                success=False,
                msg=f"rl op {req.op_name} timeout after {timeout}s",
                op_name=req.op_name,
            )

    def pop_all(self) -> List[Tuple[RlOpReq, concurrent.futures.Future]]:
        pairs = []
        while True:
            try:
                pairs.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return pairs


def start_router_rl_rpyc_server(args, router: RouterRlOpHelper):
    if args.node_rank != 0:
        return None, None

    from rpyc.utils.server import ThreadedServer
    import lightllm.utils.rpyc_fix_utils as _
    from lightllm.utils.shm_port_args import get_shm_port_args

    rl_rpyc_port = get_shm_port_args().rl_rpyc_port
    server = ThreadedServer(
        router,
        hostname="127.0.0.1",
        port=rl_rpyc_port,
        protocol_config={"allow_pickle": True, "sync_request_timeout": 600},
    )
    thread = threading.Thread(target=server.start, name="rl_rpyc_server", daemon=True)
    thread.start()
    logger.info(f"router rl rpyc server started on port {rl_rpyc_port}")
    return server, thread
