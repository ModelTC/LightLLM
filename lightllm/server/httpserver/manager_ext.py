"""HttpServerManager 的扩展 Mixin。

通过多继承挂到 :class:`HttpServerManager` 上，把 RL 控制面 HTTP 转发接口
从 manager 主体中拆出，避免 manager.py 继续膨胀。

约定：宿主类在 ``--enable_rl`` 时提供非空 ``self.rl_controller``；
RL HTTP 路由也仅在该开关下挂载，因此这些接口只应在 enable_rl 场景被调用。
"""

from typing import Tuple

from lightllm.server.io_struct import (
    AbortReq,
    DestroyWeightsUpdateGroupReq,
    FlushCacheReq,
    InitWeightsUpdateGroupReq,
    ReleaseMemoryReq,
    ResumeMemoryReq,
    RlOpRsp,
    UpdateWeightsFromDistributedReq,
    UpdateWeightsFromIPCReq,
    UpdateWeightsFromTensorReq,
)


class HttpRlManagerHelper:
    """RL 控制面接口 Mixin：一律转发到 ``self.rl_controller``。"""

    async def abort_request(self, request: AbortReq) -> Tuple[bool, str]:
        return await self.rl_controller.abort_request(request)

    async def pause_generation(self):
        return await self.rl_controller.pause_generation()

    async def continue_generation(self):
        return await self.rl_controller.continue_generation()

    async def flush_cache(self, request: FlushCacheReq):
        return await self.rl_controller.flush_cache(request)

    async def release_memory_occupation(self, request: ReleaseMemoryReq):
        return await self.rl_controller.release_memory_occupation(request)

    async def resume_memory_occupation(self, request: ResumeMemoryReq):
        return await self.rl_controller.resume_memory_occupation(request)

    async def init_weights_update_group(self, request: InitWeightsUpdateGroupReq):
        return await self.rl_controller.init_weights_update_group(request)

    async def destroy_weights_update_group(self, request: DestroyWeightsUpdateGroupReq):
        return await self.rl_controller.destroy_weights_update_group(request)

    async def update_weights_from_distributed(self, request: UpdateWeightsFromDistributedReq):
        return await self.rl_controller.update_weights_from_distributed(request)

    async def update_weights_from_tensor(self, request: UpdateWeightsFromTensorReq) -> RlOpRsp:
        return await self.rl_controller.update_weights_from_tensor(request)

    async def update_weights_from_ipc(self, request: UpdateWeightsFromIPCReq) -> RlOpRsp:
        return await self.rl_controller.update_weights_from_ipc(request)
