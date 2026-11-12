import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from lightllm.server.visualserver.manager import VisualManager


def _visual_manager(*run_tasks):
    manager = VisualManager.__new__(VisualManager)
    manager.vit_dp = len(run_tasks)
    manager.vit_tp = 1
    manager.pending_images_by_dp = [0] * manager.vit_dp
    manager.cur_dp_index = 1 % manager.vit_dp
    manager.model_rpcs = [[SimpleNamespace(run_task=run_task)] for run_task in run_tasks]
    manager.lock = asyncio.Lock()
    return manager


def test_pending_images_leave_executor_available_for_receiving_requests():
    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        all_submitted = asyncio.Event()
        completion_events = []

        async def infer_images(images, events):
            completion_events.extend(events)
            if len(completion_events) == 4:
                all_submitted.set()

        def finish_images():
            for event in completion_events:
                event.set()

        manager = _visual_manager(infer_images)
        requests = [asyncio.create_task(manager.handle_images([object()])) for _ in range(4)]
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
        assert manager.pending_images_by_dp == [0]

    asyncio.run(run())


def test_busy_visual_replica_does_not_keep_receiving_images():
    async def run():
        slow_submitted = asyncio.Event()
        slow_events = []
        fast_images = []

        async def run_slow(images, events):
            slow_events.extend(events)
            slow_submitted.set()

        async def run_fast(images, events):
            fast_images.extend(images)
            for event in events:
                event.set()

        manager = _visual_manager(run_fast, run_slow, *([run_fast] * 6))
        slow_request = asyncio.create_task(manager.handle_images([object()]))
        try:
            await asyncio.wait_for(slow_submitted.wait(), timeout=2)
            for _ in range(40):
                await asyncio.wait_for(manager.handle_images([object()]), timeout=2)
            assert len(fast_images) == 40
            assert len(slow_events) == 1
            assert manager.pending_images_by_dp == [0, 1, 0, 0, 0, 0, 0, 0]
        finally:
            for event in slow_events:
                event.set()
            await asyncio.wait_for(slow_request, timeout=2)
        assert manager.pending_images_by_dp == [0] * 8

    asyncio.run(run())


def test_visual_enqueue_failure_releases_only_the_failed_replica_assignments():
    async def run():
        good_events, bad_events = [], []

        async def run_good(images, events):
            good_events.extend(events)

        async def run_bad(images, events):
            bad_events.extend(events)
            raise RuntimeError("failed to submit images")

        manager = _visual_manager(run_good, run_bad)
        (error,) = await asyncio.gather(manager.handle_images([object(), object()]), return_exceptions=True)
        assert isinstance(error, RuntimeError)
        assert manager.pending_images_by_dp == [1, 0]

        bad_events[0].set()
        good_events[0].set()
        good_events[0].set()
        await asyncio.sleep(0)
        assert manager.pending_images_by_dp == [0, 0]

    asyncio.run(run())
