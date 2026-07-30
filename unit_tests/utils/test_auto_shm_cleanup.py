import pytest

from lightllm.utils.auto_shm_cleanup import AutoShmCleanup
from lightllm.utils.start_utils import SubmoduleManager


class FakeLibc:
    def __init__(self):
        self.shmget_calls = []
        self.shmctl_calls = []

    def shmget(self, key, size, flags):
        self.shmget_calls.append((key, size, flags))
        return 5678

    def shmctl(self, shmid, command, buffer):
        self.shmctl_calls.append((shmid, command, buffer))
        return 0


@pytest.mark.parametrize("shmid", [None, 5678])
def test_mark_sysv_shm(shmid):
    cleanup = AutoShmCleanup.__new__(AutoShmCleanup)
    cleanup.libc = FakeLibc()
    cleanup.registered_sysv_shms = {}
    cleanup.register_sysv_shm(1234, shmid)

    assert cleanup.mark_registered_sysv_shm_for_deletion() == 1
    assert cleanup.libc.shmget_calls == ([(1234, 0, 0)] if shmid is None else [])
    assert cleanup.libc.shmctl_calls == [(5678, 0, None)]


def test_processes_get_term_before_kill(monkeypatch):
    events = []

    class PsutilProcess:
        def __init__(self, pid, children=None):
            self.pid = pid
            self._children = children or []

        def children(self, recursive):
            return self._children

        def terminate(self):
            events.append(("term", self.pid))

        def kill(self):
            events.append(("kill", self.pid))

    child = PsutilProcess(11)
    parent = PsutilProcess(10, [child])
    processes = {10: parent, 11: child}
    wait_count = 0

    def wait_procs(procs, timeout):
        nonlocal wait_count
        wait_count += 1
        return ([], [parent]) if wait_count == 1 else (procs, [])

    monkeypatch.setattr("lightllm.utils.start_utils.psutil.Process", lambda pid: processes[pid])
    monkeypatch.setattr("lightllm.utils.start_utils.psutil.wait_procs", wait_procs)

    class Process:
        pid = 10

        def join(self, timeout):
            events.append(("join", timeout))

    SubmoduleManager()._terminate_processes([Process()], graceful=True)

    assert events == [("term", 11), ("term", 10), ("kill", 10), ("join", 1)]
