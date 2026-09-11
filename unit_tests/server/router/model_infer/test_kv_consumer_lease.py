from contextlib import nullcontext
from types import SimpleNamespace

# Initialize the basemodel package before importing infer_batch; importing a
# nested Triton kernel first otherwise exposes req_manager's circular import.
import lightllm.common.basemodel  # noqa: F401
from lightllm.server.core.objs import ShmReqManager
from lightllm.server.router.model_infer.mode_backend import (
    base_backend as base_backend_module,
)
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.activation_transfer.shm_kv import PagedKVPublisher


class _LockingShmReqManager:
    @staticmethod
    def get_req_lock_by_index(_shm_index):
        return nullcontext()


class _Buffer:
    def __init__(self, *, ready=False):
        self.ready = ready

    def is_ready(self):
        return self.ready


class _GenerationSafeShmReqManager:
    """Exercise the production generation check without allocating real shm."""

    release_kv_consumer = ShmReqManager.release_kv_consumer
    _validate_kv_consumer_mask = staticmethod(ShmReqManager._validate_kv_consumer_mask)

    def __init__(self, shm_index, request_id, consumer_mask):
        self.max_req_num = shm_index + 1
        self.reqs = [None] * self.max_req_num
        self.reqs[shm_index] = SimpleNamespace(
            request_id=request_id,
            kv_consumer_mask=consumer_mask,
        )

    @staticmethod
    def get_req_lock_by_index(_shm_index):
        return nullcontext()


class _ScalarTensor:
    def __init__(self):
        self.value = 0

    def fill_(self, value):
        self.value = value
        return self

    def detach(self):
        return self

    def item(self):
        return self.value


def _make_req(*, consumer_mask=1, cur_kv_len=8, input_len=8):
    shm_req = SimpleNamespace(
        index_in_shm_mem=3,
        request_id=101,
        input_len=input_len,
        infer_req_idx=-1,
        kv_ready_rank_mask=0,
        kv_consumer_mask=consumer_mask,
    )
    return SimpleNamespace(
        req_id=101,
        req_idx=7,
        shm_index=3,
        cur_kv_len=cur_kv_len,
        shm_req=shm_req,
    )


def test_publish_kv_ready_binds_mapping_and_rank_bit():
    backend = ModeBackend.__new__(ModeBackend)
    backend.rank_in_dp = 2
    backend.shm_req_manager = _LockingShmReqManager()
    backend.activation_publisher = PagedKVPublisher(
        backend.shm_req_manager,
        rank=2,
    )
    req = _make_req()

    backend._publish_kv_ready(req)

    assert req.shm_req.infer_req_idx == 7
    assert req.shm_req.kv_ready_rank_mask == 0b100

    backend.rank_in_dp = 0
    backend.activation_publisher = PagedKVPublisher(
        backend.shm_req_manager,
        rank=0,
    )
    backend._publish_kv_ready(req)
    assert req.shm_req.kv_ready_rank_mask == 0b101


def test_publish_kv_ready_waits_for_complete_prefix_and_consumer():
    backend = ModeBackend.__new__(ModeBackend)
    backend.rank_in_dp = 0
    backend.shm_req_manager = _LockingShmReqManager()
    backend.activation_publisher = PagedKVPublisher(
        backend.shm_req_manager,
        rank=0,
    )

    incomplete = _make_req(cur_kv_len=7)
    backend._publish_kv_ready(incomplete)
    assert incomplete.shm_req.kv_ready_rank_mask == 0

    no_consumer = _make_req(consumer_mask=0)
    backend._publish_kv_ready(no_consumer)
    assert no_consumer.shm_req.kv_ready_rank_mask == 0


def test_direct_kv_release_is_generation_safe():
    manager = _GenerationSafeShmReqManager(
        shm_index=3, request_id=202, consumer_mask=0b101
    )

    # A delayed release for the previous shm-slot generation cannot clear the
    # new request's identically located consumer lease.
    assert manager.release_kv_consumer(3, 101, 0b001) is False
    assert manager.reqs[3].kv_consumer_mask == 0b101
    assert manager.release_kv_consumer(3, 202, 0b100) is True
    assert manager.reqs[3].kv_consumer_mask == 0b001


def test_main_request_ready_uses_the_existing_broadcast(monkeypatch):
    calls = []

    def _record_broadcast(tensor, *, src, group, async_op):
        calls.append(("broadcast", tensor.item(), src, group, async_op))

    backend = ModeBackend.__new__(ModeBackend)
    backend.is_master_in_node = True
    backend.shm_reqs_io_buffer = _Buffer(ready=True)
    backend.node_broadcast_tensor = _ScalarTensor()
    backend.args = SimpleNamespace(node_rank=0)
    backend.node_world_size = 2
    backend.node_nccl_group = "target-tp"
    backend.is_pd_mode = False
    backend._read_reqs_buffer_and_init_reqs = lambda: calls.append("main")
    monkeypatch.setattr(base_backend_module, "broadcast", _record_broadcast)

    backend._try_read_new_reqs_normal()

    assert calls == [
        ("broadcast", 1, 0, "target-tp", False),
        "main",
    ]
