from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.transfer import CopyOperation, CopyPlan, TransferCompletion
from lightllm.server.router.model_infer.mode_backend.pd.page_lifetime import PDPageReclaimer
from lightllm.server.router.model_infer.mode_backend.pd.resource_io import copy_page
from unit_tests.common.transfer.helpers import Event


def task(key="page", status="PROC"):
    handle = SimpleNamespace(status=status)
    return SimpleNamespace(get_key=lambda: key, xfer_handle=handle, local_completion=None)


def reclaimer():
    pages = Queue()
    transporter = SimpleNamespace(
        check_task_status=lambda t: t.xfer_handle.status,
        release_xfer_handle=Mock(),
    )
    return PDPageReclaimer(pages, transporter)


@pytest.mark.parametrize("status", ["PROC", "ERR"])
def test_failed_task_does_not_recycle_an_active_network_page(status):
    pool = reclaimer()
    pending = task(status=status)
    pool.retire(pending, 3, source=True)
    assert pool.free_pages.empty()
    pool.transporter.release_xfer_handle.assert_not_called()
    pending.xfer_handle.status = "DONE"
    pool.collect()
    assert pool.free_pages.get_nowait() == 3
    pool.transporter.release_xfer_handle.assert_called_once()
    assert not pool.pending


def test_handle_release_must_succeed_before_recycling():
    pool = reclaimer()
    done = task(status="DONE")
    pool.transporter.release_xfer_handle.side_effect = RuntimeError("driver still draining")
    pool.retire(done, 4, source=True)
    assert pool.free_pages.empty()
    assert done.xfer_handle is not None
    pool.transporter.release_xfer_handle.side_effect = None
    pool.collect()
    assert pool.free_pages.get_nowait() == 4


@pytest.mark.parametrize("early_done", [True, False])
def test_remote_write_without_local_handle_waits_for_peer_confirmation(early_done):
    pool = reclaimer()
    pending = task()
    pending.xfer_handle = None
    pool.expect_remote_write(pending.get_key())
    if early_done:
        pool.confirm_remote_done(pending.get_key())
    pool.retire(pending, 8)
    if not early_done:
        assert pool.free_pages.empty()
        pool.confirm_remote_done(pending.get_key())
    assert pool.free_pages.get_nowait() == 8
    pool.confirm_remote_done(pending.get_key())
    pool.collect()
    assert pool.free_pages.empty()
    assert not pool.remote_done
    assert not pool.remote_writes


def test_local_copy_finishes_before_releasing_staging_page():
    pool = reclaimer()
    pending = task()
    pending.xfer_handle = None
    event = Event()
    pending.local_completion = TransferCompletion(event, Event())
    pool.retire(pending, 2, source=True)
    assert event.waits == 1
    assert pool.free_pages.get_nowait() == 2
    # A slow failed task does not hold up unrelated successful transfers.
    pool.retire(task("slow"), 3, source=True)
    pool.retire(task("fast", "DONE"), 4, source=True)
    assert pool.free_pages.get_nowait() == 4
    assert pool.free_pages.empty()


def test_pd_validates_all_tp_plans_before_submitting_any_copy():
    copies = []
    layout = {"kv": object()}

    def build(**kwargs):
        return CopyPlan("pack", (CopyOperation("kv", layout, lambda: copies.append(0)),))

    def invalid(**kwargs):
        raise ValueError("unsupported shard format")

    managers = [
        SimpleNamespace(
            get_pd_operator=lambda build=builder: SimpleNamespace(
                get_pd_full_att_page_layouts=lambda: layout,
                get_page_bytes=lambda: 16,
                plan_write_page=build,
            )
        )
        for builder in (build, invalid)
    ]
    with pytest.raises(ValueError, match="unsupported shard format"):
        copy_page(managers, torch.empty(1, 16), 0, 0, 2, torch.tensor([0]), True)
    assert copies == []


def test_nixl_submission_failure_keeps_handle_for_retirement():
    from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import NixlKVTransporter

    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    handle = object()
    transporter.remote_agents = {"decode": SimpleNamespace(page_xfer_handles=2)}
    transporter.page_local_xfer_handles = 1
    transporter.nixl_agent = SimpleNamespace(
        make_prepped_xfer=Mock(return_value=handle),
        transfer=Mock(side_effect=RuntimeError("submit failed")),
    )
    pending = SimpleNamespace(src_page_index=0, dst_page_index=1, decode_agent_name="decode", xfer_handle=None)
    with pytest.raises(RuntimeError, match="submit failed"):
        transporter.write_blocks_paged(pending)
    assert pending.xfer_handle is handle


def test_nccl_submission_failure_keeps_handle_for_retirement():
    from lightllm.server.router.model_infer.mode_backend.pd.nccl_kv_transporter import _NcclPeer

    peer = _NcclPeer(SimpleNamespace(kv_move_buffer=torch.empty(2, 16)), "decode")
    peer._ensure_comm = Mock(return_value=SimpleNamespace(send=Mock(side_effect=RuntimeError("submit failed"))))
    peer._get_stream = Mock(return_value=object())
    pending = SimpleNamespace(src_page_index=0, dst_page_index=1, xfer_handle=None)
    with pytest.raises(RuntimeError, match="submit failed"):
        peer.send_page(pending)
    assert pending.xfer_handle is not None
    assert pending.xfer_handle.check_status() == "PROC"
