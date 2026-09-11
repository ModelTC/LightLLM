import queue
import threading

from lightllm.server.visualserver.model_infer.model_rpc import VisualModelRpcServer


def test_force_same_consumer_does_not_hold_the_producer_batch_lock():
    first_taken = threading.Event()
    second_slot_taken = threading.Event()

    class _NotifyingQueue(queue.Queue):
        def get(self, *args, **kwargs):
            item = super().get(*args, **kwargs)
            if item == "first-request":
                first_taken.set()
            return item

    class _NotifyingSemaphore:
        def __init__(self):
            self._semaphore = threading.Semaphore(2)
            self._acquires = 0

        def acquire(self):
            acquired = self._semaphore.acquire()
            self._acquires += 1
            if self._acquires == 2:
                second_slot_taken.set()
            return acquired

        def release(self):
            return self._semaphore.release()

    server = VisualModelRpcServer.__new__(VisualModelRpcServer)
    server.infer_queue = _NotifyingQueue()
    server.infer_queue_batch_lock = threading.Lock()
    server.sempare = _NotifyingSemaphore()
    server.infer_queue.put("first-request")
    result = []

    consumer = threading.Thread(
        target=lambda: result.extend(server._get_image_items_from_infer_queue(2, force_same=True)),
        daemon=True,
    )
    consumer.start()
    assert first_taken.wait(timeout=1.0)
    assert second_slot_taken.wait(timeout=1.0)

    # This models the next request's producer. It must be able to acquire the
    # batch lock while the non-zero TP rank waits for its second image.
    acquired = server.infer_queue_batch_lock.acquire(timeout=1.0)
    try:
        if acquired:
            server.infer_queue.put("second-request")
    finally:
        if acquired:
            server.infer_queue_batch_lock.release()
        else:
            # Unblock a regressed implementation so pytest can terminate.
            server.infer_queue.put("cleanup")

    consumer.join(timeout=1.0)
    assert acquired
    assert not consumer.is_alive()
    assert result == ["first-request", "second-request"]
