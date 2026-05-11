import asyncio
import rpyc
from rpyc.core.async_ import AsyncResultTimeout
from typing import Dict, List, Tuple, Deque, Optional, Union
from lightllm.server.multimodal_params import ImageItem
from .model_rpc import VisualModelRpcServer


# init_model loads weights and runs autotune/cuda-graph capture — slow on cold start
# but bounded; pick a generous timeout so we still detect a truly hung worker.
_INIT_MODEL_TIMEOUT_S = 600
# run_task only enqueues images; the actual inference completion is signaled
# separately via VisualInferResult. If the worker is alive, this call returns in ms;
# anything longer means the RPC server is dead/hung and we should fail fast so the
# manager goes down the abort path instead of blocking on ans.wait forever.
_RUN_TASK_TIMEOUT_S = 30


class VisualModelRpcClient:
    def __init__(self, rpc_conn):
        self.rpc_conn: VisualModelRpcServer = rpc_conn

        def make_bounded(f, timeout_s: float, op_name: str):
            async_f = rpyc.async_(f)

            async def _func(*args, **kwargs):
                ans: rpyc.AsyncResult = async_f(*args, **kwargs)
                # RPyC's AsyncResult.wait() takes no timeout argument (rpyc 5.x);
                # set_expiry() configures the deadline, and wait() raises
                # AsyncResultTimeout on expiry. Replaces the previous unbounded wait
                # that swallowed hung/dead model RPCs and bypassed --visual_infer_timeout.
                ans.set_expiry(timeout_s)
                try:
                    await asyncio.to_thread(ans.wait)
                except AsyncResultTimeout as e:
                    raise TimeoutError(
                        f"{op_name} did not return within {timeout_s}s; visual worker RPC may be hung"
                    ) from e
                return ans.value

            return _func

        self._init_model = make_bounded(self.rpc_conn.root.init_model, _INIT_MODEL_TIMEOUT_S, "init_model")
        self._run_task = make_bounded(self.rpc_conn.root.run_task, _RUN_TASK_TIMEOUT_S, "run_task")

        return

    async def init_model(self, kvargs):
        return await self._init_model(kvargs)

    async def run_task(self, images: List[ImageItem], ref_result_list):
        return await self._run_task(images, ref_result_list)
