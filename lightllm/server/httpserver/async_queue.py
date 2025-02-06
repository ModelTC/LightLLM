import asyncio


class AsyncQueue:
    def __init__(self):
        self.datas = []
        self.event = asyncio.Event()
        self.lock = asyncio.Lock()

    async def wait_to_ready(self):
        try:
            await asyncio.wait_for(self.event.wait(), timeout=3)
        except asyncio.TimeoutError:
            pass

    async def get_all_data(self):
        async with self.lock:
            self.event.clear()
            ans = self.datas
            self.datas = []
            return ans

    async def put(self, obj):
        async with self.lock:
            self.datas.append(obj)
            self.event.set()
        return

    async def wait_to_get_all_data(self):
        await self.wait_to_ready()
        handle_list = await self.get_all_data()
        return handle_list
