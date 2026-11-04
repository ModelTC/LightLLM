"""Keep staging pages out of the free pool until all access has ended."""

import threading


class PDPageReclaimer:
    def __init__(self, free_pages, transporter):
        self.free_pages = free_pages
        self.transporter = transporter
        self.pending = {}
        self.remote_done = set()
        self.remote_writes = set()
        self.lock = threading.Lock()

    def expect_remote_write(self, key):
        with self.lock:
            self.remote_writes.add(key)

    def retire(self, task, page_index, *, source=False, remote_complete=False):
        if page_index is None:
            return
        if task.local_completion is not None:
            task.local_completion.source_reusable.synchronize()
        with self.lock:
            self.pending[task.get_key()] = (task, page_index, source or remote_complete)
        self.collect()

    def confirm_remote_done(self, key):
        # A late WRITE-done may arrive before the failure worker retires the page.
        with self.lock:
            if key in self.remote_writes:
                self.remote_done.add(key)
        self.collect()

    def collect(self):
        with self.lock:
            for key, (task, page_index, local_only) in list(self.pending.items()):
                if task.xfer_handle is None:
                    if not local_only and key not in self.remote_done:
                        continue
                else:
                    try:
                        if self.transporter.check_task_status(task) != "DONE":
                            continue
                        self.transporter.release_xfer_handle(task.xfer_handle)
                    except Exception:
                        # ERR, a timeout, or a failed handle release cannot prove quiescence.
                        continue
                    task.xfer_handle = None
                del self.pending[key]
                self.remote_done.discard(key)
                self.remote_writes.discard(key)
                self.free_pages.put(page_index)
