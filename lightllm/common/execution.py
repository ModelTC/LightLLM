from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock


class _Submission:
    def __init__(self):
        self.retained = []

    def retain(self, *values):
        self.retained.extend(values)


class ExecutionScope:
    def __init__(self, runtime, *, max_pending=2):
        if max_pending < 1:
            raise ValueError("execution must admit at least one pending submission")
        self.runtime = runtime
        self.max_pending = max_pending
        self._lock = Lock()
        self._current = ContextVar("model_submission", default=None)
        self._completion = None
        self._last_stream = None
        self._pending = []
        self._readers = []
        self._quarantined = False

    @contextmanager
    def submit(self, *, retain=(), on_failure=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("model execution already has an active submission")
        submission = _Submission()
        submission.retain(*retain)
        token = None
        try:
            if self._quarantined:
                raise RuntimeError("model execution is quarantined; worker recovery required")
            try:
                self._begin()
            except BaseException:
                self._quarantined = True
                for _, retained in self._pending:
                    submission.retain(*retained)
                raise
            token = self._current.set(submission)
            try:
                yield submission
            except BaseException:
                try:
                    self._finish(submission)
                except BaseException:
                    pass
                raise
            else:
                self._finish(submission)
        except BaseException as error:
            if on_failure is not None:
                on_failure(error, submission)
            raise
        finally:
            if token is not None:
                self._current.reset(token)
            self._lock.release()

    @contextmanager
    def join(self, *, retain=()):
        submission = self._current.get()
        if submission is None:
            with self.submit(retain=retain) as submission:
                yield submission
        else:
            submission.retain(*retain)
            yield submission

    def _begin(self):
        self._pending = [(event, refs) for event, refs in self._pending if event is None or not event.query()]
        while len(self._pending) >= self.max_pending:
            event, _ = self._pending[0]
            event.synchronize()
            self._pending.pop(0)
        stream = self.runtime.current_stream()
        if stream is not None:
            for event in self._readers:
                stream.wait_event(event)
            if self._completion is not None and stream != self._last_stream:
                stream.wait_event(self._completion)
        self._readers.clear()

    def retain(self, *values):
        submission = self._current.get()
        if submission is None:
            raise RuntimeError("resource retention requires an active model submission")
        submission.retain(*values)

    def _finish(self, submission):
        try:
            stream = self.runtime.current_stream()
            if stream is not None:
                event = self.runtime.create_event()
                event.record(stream)
                self._completion, self._last_stream = event, stream
                self._pending.append((event, submission.retained))
        except BaseException:
            self._pending.append((None, submission.retained))
            self._quarantined = True
            raise

    def record_reader(self, event):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot register an output reader during model submission")
        try:
            if self._quarantined:
                raise RuntimeError("model execution is quarantined; worker recovery required")
            if len(self._readers) >= self.max_pending:
                try:
                    self._readers[0].synchronize()
                except BaseException:
                    self._quarantined = True
                    raise
                self._readers.pop(0)
            self._readers.append(event)
        finally:
            self._lock.release()

    def synchronize(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot drain execution during an active submission")
        try:
            if self._quarantined:
                raise RuntimeError("cannot drain quarantined execution without worker recovery")
            if self._completion is not None:
                self._completion.synchronize()
            for event in self._readers:
                event.synchronize()
            self._readers.clear()
            self._pending.clear()
        finally:
            self._lock.release()
