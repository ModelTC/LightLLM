class PDRequestContext:
    """P 上的逻辑请求资源；单次 prefill 完成后仍保留多模态输入供 KV 恢复。"""

    def __init__(self, manager, request_id, multimodal_params):
        self.manager = manager
        self.request_id = request_id
        self.multimodal_params = multimodal_params
        self.prompt_ids = None
        self.work_ids = set()
        self.closed = False
        self.released = False
        self.recovery_epoch = 0

    def acquire(self, work_id):
        if self.closed:
            raise RuntimeError("PD request context is closed")
        self.work_ids.add(work_id)

    async def release(self, work_id):
        self.work_ids.discard(work_id)
        await self._release_if_closed()

    async def close(self):
        self.closed = True
        await self._release_if_closed()

    async def _release_if_closed(self):
        if self.closed and not self.work_ids and not self.released:
            self.released = True
            await self.manager._release_multimodal_resources(self.multimodal_params)
