import asyncio
import traceback


class HttpServerTasks:
    def __init__(self):
        self.closing = False
        self.tasks = set()

    def create(self, coroutine):
        if self.closing:
            coroutine.close()
            raise RuntimeError("HTTP background tasks are closing")
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def close(self):
        # A wait_for completion racing with cancellation can consume the cancel
        # on Python 3.10. Polling loops also observe this explicit stop condition.
        self.closing = True
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            if task.cancelled():
                try:
                    task.result()
                except asyncio.CancelledError as error:
                    # gather creates a new CancelledError for its result list.
                    # Retrieve the original so a retained Task cannot pin SHM
                    # views through locals in its completed coroutine frames.
                    # Python 3.10 attaches that original exception as context.
                    seen = set()
                    while error is not None and id(error) not in seen:
                        seen.add(id(error))
                        traceback.clear_frames(error.__traceback__)
                        error = error.__context__
