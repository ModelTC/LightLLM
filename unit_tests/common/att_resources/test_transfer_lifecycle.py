from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

from lightllm.common.attention_state.lifecycle import RetiredTransfers, transfer_completion


class Fence:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done


def task(generation="first", handle=None):
    return SimpleNamespace(transfer_id=generation, xfer_handle=handle, src_page_index=2, dst_page_index=3)


def test_source_completion_does_not_release_staging_while_transport_reads():
    transport = SimpleNamespace(check_task_status=Mock(return_value="PROC"), release_xfer_handle=Mock())
    pages = Queue()
    retired = RetiredTransfers(transport, pages, "src_page_index")
    transfer = task(handle=object())
    completion = transfer_completion(transfer)
    completion.source_reusable = Fence()
    completion.network_submitted = True
    completion.cancelled = True
    retired.retire(transfer)
    completion.source_reusable.done = True
    retired.poll()
    assert pages.empty()
    transport.check_task_status.return_value = "ERR"
    retired.poll()
    assert pages.empty()
    transport.release_xfer_handle.assert_not_called()
    transport.check_task_status.return_value = "DONE"
    retired.poll()
    assert pages.get_nowait() == 2
    transport.release_xfer_handle.assert_called_once()
    retired.poll()
    assert pages.empty()


def test_remote_done_requires_correct_generation_and_local_receive_completion():
    transport = SimpleNamespace(check_task_status=Mock(return_value="PROC"), release_xfer_handle=Mock())
    pages = Queue()
    retired = RetiredTransfers(transport, pages, "dst_page_index")
    transfer = task(handle=object())
    transfer_completion(transfer).network_submitted = True
    retired.watch(transfer)
    assert not retired.remote_done("old-generation")
    assert retired.remote_done("first")
    retired.retire(transfer)
    assert pages.empty()
    transport.check_task_status.return_value = "DONE"
    retired.poll()
    assert pages.get_nowait() == 3


def test_rdma_page_is_quarantined_until_done_and_unpack_finishes():
    pages = Queue()
    retired = RetiredTransfers(SimpleNamespace(), pages, "dst_page_index")
    transfer = task()
    completion = transfer_completion(transfer)
    completion.network_submitted = True
    completion.destination_ready = Fence()
    retired.retire(transfer)
    assert pages.empty()
    retired.remote_done("first")
    assert pages.empty()
    completion.destination_ready.done = True
    retired.poll()
    assert pages.get_nowait() == 3
