import queue
import threading
from typing import List, Optional


def _put_front(infer_queue: "queue.Queue", item) -> None:
    """Push ``item`` back to the front of ``infer_queue``.

    ``queue.Queue.put`` appends to the tail, which would reorder pending items
    relative to other consumers. The ViT scheduler runs rank-0-only admission
    on a queue that every TP rank holds an identical copy of, and rank N
    later pops ``len(images)`` items in FIFO order to follow rank 0's
    decision. If a rejected item moved to the tail of rank 0's queue, the
    queues across ranks would diverge and the next batch would encode
    different images on different ranks. Re-inserting at the head preserves
    FIFO order on rank 0 and keeps all ranks in sync.

    Note: ``Queue.get`` does *not* decrement ``unfinished_tasks`` — only
    ``task_done()`` does. The original ``Queue.put`` already counted this
    item, so we must NOT bump the counter again on re-insert; doing so would
    desync ``Queue.join()``/``task_done()`` accounting (a latent footgun if
    any future caller starts using them on this queue).
    """
    with infer_queue.mutex:
        infer_queue.queue.appendleft(item)
        infer_queue.not_empty.notify()


def pull_batch_with_budget(
    infer_queue: "queue.Queue",
    semaphore: threading.Semaphore,
    max_num: int,
    max_tokens: Optional[int],
    timeout: Optional[float] = None,
) -> List:
    """Pull up to ``max_num`` image items from ``infer_queue`` while keeping the
    cumulative ``item.token_num`` at or below ``max_tokens``.

    Rank-0-only admission logic for the ViT scheduler. The first item is always
    admitted even when it alone exceeds ``max_tokens`` — this avoids a deadlock
    when a single request is larger than the per-step budget. Each subsequent
    item is pulled, inspected, and either kept or pushed back to the front of
    the queue so non-rank-0 workers' FIFO pops stay aligned with rank 0's
    admitted set.

    ``semaphore`` counts share with the caller (see ``_init_taskes``); callers
    acquire before every get and release on over-pull so the permit count stays
    consistent with queue contents.

    When ``timeout`` is not None, the first acquire/get is bounded so rank 0
    can emit a heartbeat broadcast instead of blocking indefinitely on the
    gloo broadcast (avoids the 30-minute NCCL-style timeout on idle workers).
    An empty list is returned on timeout.
    """
    tasks: List = []

    if timeout is not None:
        if not semaphore.acquire(timeout=timeout):
            return tasks
        try:
            first = infer_queue.get(timeout=timeout)
        except queue.Empty:
            semaphore.release()
            return tasks
    else:
        semaphore.acquire()
        first = infer_queue.get(block=True)
    tasks.append(first)
    total_tokens = first.token_num or 0

    while len(tasks) < max_num:
        try:
            task = infer_queue.get(block=False)
        except queue.Empty:
            break
        if not semaphore.acquire(blocking=False):
            _put_front(infer_queue, task)
            break

        next_tokens = task.token_num or 0
        if max_tokens is not None and total_tokens + next_tokens > max_tokens:
            _put_front(infer_queue, task)
            semaphore.release()
            break

        tasks.append(task)
        total_tokens += next_tokens

    return tasks
