import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from lightllm.server.visualserver.manager import VisualManager


def test_pending_images_leave_executor_available_for_receiving_requests():
    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        all_submitted = asyncio.Event()
        completion_events = []

        async def infer_images(dp_index, images, events):
            completion_events.extend(events)
            if len(completion_events) == 4:
                all_submitted.set()

        def finish_images():
            for event in completion_events:
                event.set()

        manager = SimpleNamespace(vit_dp=1, lock=asyncio.Lock(), infer_images=infer_images)
        requests = [asyncio.create_task(VisualManager.handle_images(manager, [object()])) for _ in range(4)]
        try:
            await asyncio.wait_for(all_submitted.wait(), timeout=2)
            # The ZMQ receive uses the same executor.
            assert await asyncio.wait_for(asyncio.to_thread(lambda: True), timeout=2)
            assert all(not request.done() for request in requests)
        finally:
            # Model workers notify completion on an RPyC serving thread.
            notifier = threading.Thread(target=finish_images)
            notifier.start()
            notifier.join()
            await asyncio.wait_for(asyncio.gather(*requests), timeout=2)

    asyncio.run(run())
