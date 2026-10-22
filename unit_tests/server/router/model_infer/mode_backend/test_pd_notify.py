import pickle
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server.pd_io_struct import PDChunckedTransTask
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_trans_process import (
    _SharedTimeOutMark,
)
from lightllm.server.router.model_infer.mode_backend.pd.nccl_kv_transporter import (
    NcclKVTransporter,
)
from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import (
    NixlKVTransporter,
)


def make_task(page_kind):
    task = PDChunckedTransTask(
        request_id=123,
        start_kv_index=4,
        end_kv_index=5 if page_kind == "kv" else 4,
        time_out_secs=10,
        pd_master_node_id=2,
        prefill_dp_index=0,
        decode_dp_index=1,
        src_device_id=0,
        dst_device_id=1,
        mem_indexes=[7] if page_kind == "kv" else [],
        prefill_agent_name="prefill",
        prefill_agent_metadata=b"prefill metadata",
        prefill_num_pages=8,
        prefill_page_reg_desc=b"prefill pages",
        decode_agent_name="decode",
        decode_agent_metadata=b"decode metadata",
        decode_num_pages=9,
        decode_page_reg_desc=b"decode pages",
        first_gen_token_id=42,
        first_gen_token_logprob=-0.5,
        page_kind=page_kind,
        req_idx=17,
        src_page_index=3,
        dst_page_index=6,
        xfer_handle=object(),
    )
    return task


@pytest.mark.parametrize("page_kind", ["kv", "att_state"])
@pytest.mark.parametrize("with_mark", [False, True])
def test_copy_for_notify_preserves_local_task(page_kind, with_mark):
    task = make_task(page_kind)
    if with_mark:
        task.shared_time_out_mark = _SharedTimeOutMark()
        time_mark = task.shared_time_out_mark.time_mark
    original = vars(task).copy()
    mem_indexes = list(task.mem_indexes)

    notify = task.copy_for_notify()

    assert notify is not task
    assert not hasattr(notify, "shared_time_out_mark")
    assert notify.mem_indexes is None
    assert notify.xfer_handle is None
    assert vars(notify) == {
        **{key: value for key, value in original.items() if key != "shared_time_out_mark"},
        "mem_indexes": None,
        "xfer_handle": None,
    }
    assert vars(task) == original
    assert all(vars(task)[key] is value for key, value in original.items())
    assert task.mem_indexes == mem_indexes
    if with_mark:
        assert task.shared_time_out_mark.time_mark == time_mark


_UNPICKLE_WITHOUT_DECODE = """
import importlib.abc
import pickle
import sys
from lightllm.server.pd_io_struct import PDChunckedTransTask

class BlockDecode(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if 'decode_node_impl' in fullname:
            raise ImportError('decode modules are unavailable to the receiver')

assert not any('decode_node_impl' in name for name in sys.modules)
sys.meta_path.insert(0, BlockDecode())
task = pickle.loads(sys.stdin.buffer.read())
assert isinstance(task, PDChunckedTransTask)
assert not hasattr(task, 'shared_time_out_mark')
assert task.mem_indexes is None
assert task.xfer_handle is None
assert not any('decode_node_impl' in name for name in sys.modules)
"""


@pytest.mark.parametrize("backend", [NcclKVTransporter, NixlKVTransporter])
@pytest.mark.parametrize("page_kind", ["kv", "att_state"])
@pytest.mark.parametrize(
    "method,stage,destination",
    [
        ("send_write_done_task_to_decode_node", "done", "decode"),
        ("send_write_request_task_to_decode_node", "request", "decode"),
        ("send_write_ready_task_to_prefill_node", "ready", "prefill"),
        ("send_error_info_to_prefill_node", "error", "prefill"),
        ("send_error_info_to_decode_node", "error", "decode"),
    ],
)
def test_notification_payload(backend, page_kind, method, stage, destination):
    task = make_task(page_kind)
    task.error_info = "transfer failed" if stage == "error" else None
    task.shared_time_out_mark = _SharedTimeOutMark()
    original = vars(task).copy()
    time_mark = task.shared_time_out_mark.time_mark
    transporter = backend.__new__(backend)
    transporter.node_id, transporter.tp_idx = 10, 0
    transporter.num_pages = 12
    transporter.remote_agents = {"prefill": object(), "decode": object()}
    if backend is NcclKVTransporter:
        transporter.host_ip = "127.0.0.1"
        transporter.control_channel = SimpleNamespace(port=1234, send_notif=Mock())
        transporter.kv_move_buffer = SimpleNamespace(dtype="float16")
        transporter.page_size = transporter.num_layers = transporter.kv_head_num = transporter.head_dims = 1
        transporter._get_remote_metadata = Mock(return_value=SimpleNamespace(host_ip="127.0.0.2", control_port=4321))
        transporter._get_peer = Mock(return_value=SimpleNamespace(start_recv=Mock()))
    else:
        transporter.page_reg_desc = object()
        transporter.nixl_agent = SimpleNamespace(
            get_agent_metadata=lambda: b"local metadata",
            get_serialized_descs=lambda _: b"local pages",
            send_notif=Mock(),
        )

    getattr(transporter, method)(task)

    if backend is NcclKVTransporter:
        transporter.control_channel.send_notif.assert_called_once()
        (
            peer,
            host,
            port,
            payload,
        ) = transporter.control_channel.send_notif.call_args.args
        assert (peer, host, port) == (destination, "127.0.0.2", 4321)
    else:
        transporter.nixl_agent.send_notif.assert_called_once()
        kwargs = transporter.nixl_agent.send_notif.call_args.kwargs
        assert kwargs["remote_agent_name"] == destination
        payload = kwargs["notif_msg"]
    result = subprocess.run(
        [sys.executable, "-c", _UNPICKLE_WITHOUT_DECODE],
        input=payload,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    notify = pickle.loads(payload)
    sender = "prefill" if destination == "decode" else "decode"
    expected = {key: value for key, value in original.items() if key != "shared_time_out_mark"}
    expected.update(mem_indexes=None, xfer_handle=None, write_stage=stage)
    expected.update(
        {
            f"{sender}_agent_name": transporter.agent_name,
            f"{sender}_agent_metadata": transporter.agent_metadata,
            f"{sender}_num_pages": transporter.num_pages,
            f"{sender}_page_reg_desc": transporter.local_page_mem_desc,
        }
    )
    if backend is NixlKVTransporter and stage == "done":
        expected.update(decode_agent_metadata=None, decode_page_reg_desc=None)
    assert vars(notify) == expected
    assert vars(task) == original
    assert all(vars(task)[key] is value for key, value in original.items())
    assert task.shared_time_out_mark.time_mark == time_mark


def test_receiver_rejects_payload_with_decode_local_mark():
    task = make_task("kv")
    task.shared_time_out_mark = _SharedTimeOutMark()
    result = subprocess.run(
        [sys.executable, "-c", _UNPICKLE_WITHOUT_DECODE],
        input=pickle.dumps(task),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert b"decode modules are unavailable to the receiver" in result.stderr
