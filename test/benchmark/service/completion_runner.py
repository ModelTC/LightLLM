from concurrent.futures import ThreadPoolExecutor
import threading

import requests

from .completion import completion_request, completion_tokens


class CompletionClient:
    def __init__(
        self,
        framework,
        model,
        base_url,
        timeout,
        eos_ids,
        raw_directory,
        *,
        concurrency=1,
        session_factory=requests.Session,
    ):
        if type(concurrency) is not int or not 1 <= concurrency <= 8:
            raise ValueError("Completion concurrency must be between one and eight")
        self.framework, self.model, self.base_url = framework, model, base_url
        self.timeout, self.eos_ids, self.raw_directory = timeout, eos_ids, raw_directory
        self.concurrency, self.session_factory = concurrency, session_factory
        self.local = threading.local()
        self.sessions, self.lock, self.executor = [], threading.Lock(), None

    def _complete(self, case):
        index = case["dataset_index"]
        if type(index) is not int or index < 0:
            raise ValueError("Require a nonnegative integer dataset index")
        if not hasattr(self.local, "session"):
            session = self.session_factory()
            session.trust_env = False
            self.local.session = session
            with self.lock:
                self.sessions.append(session)
        endpoint, payload = completion_request(
            self.framework, self.model, case["prompt_ids"], case["max_output_tokens"]
        )
        response = self.local.session.post(self.base_url.rstrip("/") + endpoint, json=payload, timeout=self.timeout)
        (self.raw_directory / f"response-{index}.txt").write_text(response.text)
        response.raise_for_status()
        return completion_tokens(
            response.json(), self.framework, case["prompt_ids"], case["max_output_tokens"], self.eos_ids
        )

    def results(self, cases):
        if self.concurrency == 1:
            return map(self._complete, cases)
        if self.executor is not None:
            raise RuntimeError("A completion client accepts one concurrent workload")
        self.executor = ThreadPoolExecutor(max_workers=self.concurrency)

        return self.executor.map(self._complete, cases)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        for session in self.sessions:
            session.close()
