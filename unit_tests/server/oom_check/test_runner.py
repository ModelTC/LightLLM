import importlib.util
import os
import unittest


_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "lightllm",
        "server",
        "oom_check",
        "runner.py",
    )
)


def _load_helpers():
    """Load only the pure helpers from runner.py. The module imports aiohttp/
    requests/transformers at module scope; we read the source and pull out just
    classify_outcome + summarize so tests do not require those deps.
    """
    with open(_PATH, "r") as f:
        src = f.read()

    ns = {"Optional": __import__("typing").Optional, "List": __import__("typing").List}
    for fn_name in ("classify_outcome", "summarize"):
        start = src.index(f"def {fn_name}(")
        tail = src[start:]
        next_def = tail.find("\ndef ", 1)
        next_async = tail.find("\nasync def ", 1)
        ends = [e for e in (next_def, next_async) if e != -1]
        end = min(ends) if ends else len(tail)
        exec(tail[:end], ns)
    return ns


_helpers = _load_helpers()
classify_outcome = _helpers["classify_outcome"]
summarize = _helpers["summarize"]


class TestClassifyOutcome(unittest.TestCase):
    def test_200_is_ok(self):
        self.assertEqual(classify_outcome(200, None), "ok")

    def test_4xx(self):
        self.assertEqual(classify_outcome(422, None), "http_4xx")
        self.assertEqual(classify_outcome(400, None), "http_4xx")
        self.assertEqual(classify_outcome(499, None), "http_4xx")

    def test_5xx(self):
        self.assertEqual(classify_outcome(500, None), "http_5xx")
        self.assertEqual(classify_outcome(503, None), "http_5xx")

    def test_timeout_exception(self):
        self.assertEqual(classify_outcome(None, "TimeoutError()"), "timeout")
        self.assertEqual(classify_outcome(None, "asyncio.TimedOut"), "timeout")

    def test_other_exception(self):
        self.assertEqual(classify_outcome(None, "ConnectionResetError(104)"), "other")

    def test_unknown_status(self):
        self.assertEqual(classify_outcome(None, None), "other")
        self.assertEqual(classify_outcome(302, None), "other")


class TestSummarize(unittest.TestCase):
    def test_all_ok(self):
        outcomes = [
            {"class": "ok", "latency_s": 1.0},
            {"class": "ok", "latency_s": 2.0},
            {"class": "ok", "latency_s": 3.0},
        ]
        s = summarize(outcomes, duration_s=10.5)
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["ok"], 3)
        self.assertEqual(s["failed"], 0)
        self.assertEqual(s["by_class"], {"ok": 3})
        self.assertEqual(s["duration_s"], 10.5)
        self.assertEqual(s["max_s"], 3.0)

    def test_mixed(self):
        outcomes = [
            {"class": "ok", "latency_s": 1.0},
            {"class": "ok", "latency_s": 2.0},
            {"class": "http_5xx", "latency_s": 0.1},
            {"class": "timeout", "latency_s": 1800.0},
        ]
        s = summarize(outcomes, duration_s=2000.0)
        self.assertEqual(s["total"], 4)
        self.assertEqual(s["ok"], 2)
        self.assertEqual(s["failed"], 2)
        self.assertEqual(s["by_class"], {"ok": 2, "http_5xx": 1, "timeout": 1})

    def test_empty(self):
        s = summarize([], duration_s=0.0)
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["ok"], 0)
        self.assertEqual(s["failed"], 0)
        self.assertEqual(s["by_class"], {})
        self.assertIsNone(s["p50_s"])
        self.assertIsNone(s["p95_s"])
        self.assertIsNone(s["max_s"])

    def test_percentiles_sorted(self):
        outcomes = [{"class": "ok", "latency_s": x} for x in [5.0, 1.0, 3.0, 2.0, 4.0]]
        s = summarize(outcomes, duration_s=5.0)
        self.assertEqual(s["max_s"], 5.0)
        self.assertGreaterEqual(s["p95_s"], s["p50_s"])


if __name__ == "__main__":
    unittest.main()
