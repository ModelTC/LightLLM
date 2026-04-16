import queue
import threading
from typing import List, Optional


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
    item is pulled, inspected, and either kept or returned to the queue.

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
            semaphore.acquire()
            task = infer_queue.get(block=False)
        except queue.Empty:
            semaphore.release()
            break

        next_tokens = task.token_num or 0
        if max_tokens is not None and total_tokens + next_tokens > max_tokens:
            infer_queue.put(task)
            semaphore.release()
            break

        tasks.append(task)
        total_tokens += next_tokens

    return tasks
