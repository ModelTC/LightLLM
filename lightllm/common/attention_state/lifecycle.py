from dataclasses import dataclass
from threading import Lock


class UncertainCompletion(RuntimeError):
    pass


@dataclass
class TransferCompletion:
    source_reusable: object = None
    destination_ready: object = None
    network_submitted: bool = False
    network_done: bool = False
    cancelled: bool = False

    def wait_local_accesses(self):
        for fence in (self.source_reusable, self.destination_ready):
            if fence is not None:
                fence.synchronize()

    def quiescent(self, task, transport):
        for fence in (self.source_reusable, self.destination_ready):
            if fence is not None and not fence.query():
                return False
        if task.xfer_handle is not None:

            return transport.check_task_status(task) == "DONE"
        return not self.network_submitted or self.network_done


def transfer_completion(task) -> TransferCompletion:
    completion = getattr(task, "_local_completion", None)
    if completion is None:
        completion = task._local_completion = TransferCompletion()
    return completion


class RetiredTransfers:
    def __init__(self, transport, page_queue, page_attribute):
        self.transport = transport
        self.page_queue = page_queue
        self.page_attribute = page_attribute
        self.pending = {}
        self.active = {}
        self.lock = Lock()

    def watch(self, task):
        with self.lock:
            self.active[task.transfer_id] = task

    def retire(self, task):
        with self.lock:
            self.active.pop(task.transfer_id, None)
            self.pending[task.transfer_id] = task
        self.poll()

    def remote_done(self, transfer_id):
        with self.lock:
            task = self.pending.get(transfer_id) or self.active.get(transfer_id)
            if task is None:
                return False
            transfer_completion(task).network_done = True
        self.poll()
        return True

    def poll(self):
        with self.lock:
            for generation, task in list(self.pending.items()):
                try:
                    if not transfer_completion(task).quiescent(task, self.transport):
                        continue
                except Exception:
                    continue

                if task.xfer_handle is not None:
                    try:
                        self.transport.release_xfer_handle(task.xfer_handle)
                    except Exception:
                        continue
                    task.xfer_handle = None
                page = getattr(task, self.page_attribute)
                if page is not None:
                    self.page_queue.put(page)
                    setattr(task, self.page_attribute, None)
                del self.pending[generation]
