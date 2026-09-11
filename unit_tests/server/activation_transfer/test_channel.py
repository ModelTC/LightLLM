from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from lightllm.server.activation_transfer import (
    ActivationKind,
    ActivationRef,
    ActivationTransport,
    HiddenStatesActivation,
    HiddenStatesDescriptor,
    KVActivation,
)
from lightllm.server.activation_transfer.channel import (
    ActivationChannel,
    ActivationLeaseState,
)
from lightllm.server.core.objs import KV_CONSUMER_GENERATION


class _FakeShmReqManager:
    def __init__(self, req):
        self.req = req
        self.events = []

    def get_req_obj_by_index(self, slot):
        self.events.append(("get", slot))
        return self.req

    def get_req_lock_by_index(self, slot):
        return nullcontext()

    def release_kv_consumer(self, **kwargs):
        self.events.append(("release", kwargs))
        return True

    def put_back_req_obj(self, req):
        self.events.append(("put_back", req.request_id))


def _shm_req(*, ready_mask=0b11):
    return SimpleNamespace(
        index_in_shm_mem=3,
        request_id=17,
        infer_req_idx=5,
        input_len=11,
        kv_ready_rank_mask=ready_mask,
        kv_consumer_mask=KV_CONSUMER_GENERATION,
        is_aborted=False,
        sample_params=SimpleNamespace(seed=23),
    )


def test_kv_reference_acquires_resolves_and_releases_without_server_semantics():
    manager = _FakeShmReqManager(_shm_req())
    channel = ActivationChannel.for_shm_kv(manager, expected_ready_mask=0b11)
    reference = ActivationRef.kv(3, 17, position_offset=7)

    lease = channel.acquire(reference)
    assert lease.state is ActivationLeaseState.HELD
    assert lease.resolve() == KVActivation(3, 17, 5, 11, position_offset=7)

    lease.mark_in_use()
    lease.release()

    assert lease.state is ActivationLeaseState.RELEASED
    assert len(channel) == 0
    assert [event[0] for event in manager.events] == ["get", "release", "put_back"]


def test_kv_reference_stays_unresolved_until_every_rank_is_ready():
    req = _shm_req(ready_mask=0b01)
    channel = ActivationChannel.for_shm_kv(
        _FakeShmReqManager(req),
        expected_ready_mask=0b11,
    )
    lease = channel.acquire(ActivationRef.kv(3, 17))

    assert lease.resolve() is None
    req.kv_ready_rank_mask = 0b11
    assert isinstance(lease.resolve(), KVActivation)
    lease.release()


class _HiddenLease:
    def __init__(self, reference):
        self.reference = reference
        self.released = False

    @property
    def default_seed(self):
        return 0

    def resolve(self):
        return HiddenStatesActivation(
            self.reference,
            HiddenStatesDescriptor(
                shape=(1, 4, 8),
                dtype="float16",
                storage_handle="hidden-slot-2",
                nbytes=64,
            ),
        )

    def release(self):
        self.released = True

    def abandon(self):
        self.released = True


class _HiddenStore:
    kind = ActivationKind.HIDDEN_STATES

    def __init__(self, transport):
        self.transport = transport
        self.leases = []

    def open(self, reference):
        lease = _HiddenLease(reference)
        self.leases.append(lease)
        return lease


@pytest.mark.parametrize(
    "transport",
    (ActivationTransport.CUDA_IPC, ActivationTransport.CPU_SHM),
)
def test_hidden_states_use_the_same_channel_with_transport_specific_stores(
    transport,
):
    store = _HiddenStore(transport)
    channel = ActivationChannel(store)
    reference = ActivationRef.hidden_states(2, 9, transport=transport)

    lease = channel.acquire(reference)
    resolved = lease.resolve()

    assert resolved.reference == reference
    assert resolved.descriptor.shape == (1, 4, 8)
    lease.release()
    assert store.leases[0].released


def test_channel_selects_hidden_store_by_transport():
    cuda_store = _HiddenStore(ActivationTransport.CUDA_IPC)
    cpu_store = _HiddenStore(ActivationTransport.CPU_SHM)
    channel = ActivationChannel(cuda_store, cpu_store)

    cuda = channel.acquire(
        ActivationRef.hidden_states(
            1,
            10,
            transport=ActivationTransport.CUDA_IPC,
        )
    )
    cpu = channel.acquire(
        ActivationRef.hidden_states(
            2,
            20,
            transport=ActivationTransport.CPU_SHM,
        )
    )

    assert cuda.resolve().reference.transport is ActivationTransport.CUDA_IPC
    assert cpu.resolve().reference.transport is ActivationTransport.CPU_SHM
    channel.release_all()
