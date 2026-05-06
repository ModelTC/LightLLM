import importlib.util
import os
import queue
import threading
import unittest

# Load the helper module directly from its file so the test does not trigger
# `lightllm.server.visualserver.__init__`, which imports heavy GPU/ViT attention
# backends that are unavailable in a plain CPU test environment.
_BATCHING_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "lightllm",
        "server",
        "visualserver",
        "model_infer",
        "batching.py",
    )
)
_spec = importlib.util.spec_from_file_location("_batching_under_test", _BATCHING_PATH)
_batching = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batching)
pull_batch_with_budget = _batching.pull_batch_with_budget


class _FakeImg:
    def __init__(self, token_num):
        self.token_num = token_num


def _setup(token_nums):
    q = queue.Queue()
    for tn in token_nums:
        q.put(_FakeImg(tn))
    sem = threading.Semaphore(len(token_nums))
    return q, sem


class TestPullBatchWithBudget(unittest.TestCase):
    def test_unlimited_budget_acts_like_count_cap(self):
        q, sem = _setup([100, 200, 300, 400])
        got = pull_batch_with_budget(q, sem, max_num=3, max_tokens=None)
        self.assertEqual([g.token_num for g in got], [100, 200, 300])
        self.assertEqual(q.qsize(), 1)

    def test_budget_stops_before_overflow(self):
        q, sem = _setup([100, 200, 300, 400])
        got = pull_batch_with_budget(q, sem, max_num=10, max_tokens=400)
        # 100 + 200 = 300 <= 400; +300 -> 600 > 400 -> stop and put 300 back.
        self.assertEqual([g.token_num for g in got], [100, 200])
        self.assertEqual(q.qsize(), 2)
        remaining = [q.get_nowait().token_num for _ in range(q.qsize())]
        self.assertIn(300, remaining)
        self.assertIn(400, remaining)

    def test_first_image_always_admitted_even_if_over_budget(self):
        q, sem = _setup([10_000, 5])
        got = pull_batch_with_budget(q, sem, max_num=10, max_tokens=100)
        self.assertEqual([g.token_num for g in got], [10_000])
        self.assertEqual(q.qsize(), 1)

    def test_single_item_queue(self):
        q, sem = _setup([42])
        got = pull_batch_with_budget(q, sem, max_num=5, max_tokens=1000)
        self.assertEqual([g.token_num for g in got], [42])
        self.assertEqual(q.qsize(), 0)

    def test_budget_at_exact_boundary_admits(self):
        q, sem = _setup([100, 200, 300])
        got = pull_batch_with_budget(q, sem, max_num=10, max_tokens=300)
        # 100 + 200 = 300 == budget -> admit; +300 -> 600 > 300 -> stop.
        self.assertEqual([g.token_num for g in got], [100, 200])

    def test_none_token_num_treated_as_zero(self):
        q = queue.Queue()
        q.put(_FakeImg(100))
        q.put(_FakeImg(None))
        q.put(_FakeImg(50))
        sem = threading.Semaphore(3)
        got = pull_batch_with_budget(q, sem, max_num=10, max_tokens=100)
        # 100 (admitted first), 0 (None) -> 100 admitted, +50 -> 150 > 100 -> stop.
        self.assertEqual([g.token_num for g in got], [100, None])
        self.assertEqual(q.qsize(), 1)

    def test_max_num_respected_under_budget(self):
        q, sem = _setup([10, 10, 10, 10, 10])
        got = pull_batch_with_budget(q, sem, max_num=3, max_tokens=10_000)
        self.assertEqual(len(got), 3)
        self.assertEqual(q.qsize(), 2)

    def test_semaphore_permits_match_returned_items(self):
        # After the pull, permits consumed must equal len(returned) so the outer
        # backpressure accounting in _store_worker releases the right count.
        q, sem = _setup([100, 200, 300, 400, 500])
        permits_before = sem._value
        got = pull_batch_with_budget(q, sem, max_num=10, max_tokens=400)
        permits_after = sem._value
        self.assertEqual(permits_before - permits_after, len(got))

    def test_semaphore_permits_match_on_queue_empty(self):
        q, sem = _setup([100, 200])
        permits_before = sem._value
        got = pull_batch_with_budget(q, sem, max_num=10, max_tokens=None)
        permits_after = sem._value
        self.assertEqual(permits_before - permits_after, len(got))
        self.assertEqual(len(got), 2)

    def test_rejected_item_returns_to_front_preserves_fifo(self):
        # TP-correctness regression: after rank 0's budget admission, the
        # residual queue must equal the original FIFO order with the admitted
        # prefix removed. Other TP ranks pop ``len(returned)`` from their own
        # identical queues, so any reorder on rank 0 makes ranks encode
        # different images on the next step.
        q, sem = _setup([100, 500, 100])
        got = pull_batch_with_budget(q, sem, max_num=10, max_tokens=200)
        # 100 admitted, 500 over-budget -> rejected, loop breaks.
        self.assertEqual([g.token_num for g in got], [100])
        remaining_in_order = [q.get_nowait().token_num for _ in range(q.qsize())]
        self.assertEqual(remaining_in_order, [500, 100])

    def test_rejected_on_sem_exhaustion_returns_to_front(self):
        # Semaphore-skip path mirrors the budget-skip path: the popped item
        # must end up at the front of the queue, not the tail.
        q = queue.Queue()
        for tn in [100, 200, 300]:
            q.put(_FakeImg(tn))
        # Two permits => first acquire (200) succeeds, second (300) fails.
        sem = threading.Semaphore(2)
        got = pull_batch_with_budget(q, sem, max_num=10, max_tokens=10_000)
        self.assertEqual([g.token_num for g in got], [100, 200])
        remaining_in_order = [q.get_nowait().token_num for _ in range(q.qsize())]
        self.assertEqual(remaining_in_order, [300])

    def test_unfinished_tasks_stays_consistent_through_reject(self):
        # Queue.put bumps unfinished_tasks; Queue.get does NOT decrement it
        # (only task_done() does). The reject path re-inserts at the front
        # and must not bump the counter again — otherwise Queue.join() would
        # hang forever even after every consumed item is task_done()'d.
        q, sem = _setup([100, 500, 100])
        # 3 items put in => unfinished_tasks == 3
        self.assertEqual(q.unfinished_tasks, 3)
        got = pull_batch_with_budget(q, sem, max_num=10, max_tokens=200)
        self.assertEqual([g.token_num for g in got], [100])
        # One item consumed (returned to caller, awaiting task_done), two
        # still pending in the queue. Counter should still match the number
        # of logical outstanding tasks: 3.
        self.assertEqual(q.unfinished_tasks, 3)
        # task_done for the returned item, then drain the rest with task_done.
        q.task_done()
        self.assertEqual(q.qsize(), 2)
        for _ in range(q.qsize()):
            q.get_nowait()
            q.task_done()
        # join() must return promptly; if the counter were corrupted it would hang.
        q.join()


if __name__ == "__main__":
    unittest.main()
