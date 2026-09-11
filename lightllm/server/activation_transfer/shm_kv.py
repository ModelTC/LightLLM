"""LightLLM shared-request implementation of the KV activation store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lightllm.server.core.objs import (
    KV_CONSUMER_GENERATION,
    KV_CONSUMER_MASK_BITS,
    KVActivation,
)

from .protocol import ActivationKind, ActivationRef, ActivationTransport


@dataclass(frozen=True)
class _KVSnapshot:
    infer_req_idx: int
    seq_len: int
    ready: bool
    has_consumer: bool
    aborted: bool


class _ShmKVStoreLease:
    def __init__(
        self,
        store: "ShmKVActivationStore",
        reference: ActivationRef,
        shm_req: Any,
    ) -> None:
        self._store = store
        self._manager = store.manager
        self._shm_req = shm_req
        self.reference = reference
        self._released = False

    @property
    def default_seed(self) -> int:
        """Compatibility metadata; new consumers carry seed in their task."""

        self._require_live()
        return int(self._shm_req.sample_params.seed)

    def snapshot(self) -> _KVSnapshot:
        self._require_live()
        slot = self.reference.slot_index
        with self._manager.get_req_lock_by_index(slot):
            if int(self._shm_req.request_id) != self.reference.generation:
                raise RuntimeError("ShmReq slot was reused before activation dispatch")
            ready_mask = int(self._shm_req.kv_ready_rank_mask)
            consumer_mask = int(self._shm_req.kv_consumer_mask)
            return _KVSnapshot(
                infer_req_idx=int(self._shm_req.infer_req_idx),
                seq_len=int(self._shm_req.input_len),
                ready=(ready_mask & self._store.expected_ready_mask)
                == self._store.expected_ready_mask,
                has_consumer=bool(consumer_mask & self._store.consumer_mask),
                aborted=bool(self._shm_req.is_aborted),
            )

    def resolve(self) -> KVActivation | None:
        snapshot = self.snapshot()
        if snapshot.aborted:
            raise RuntimeError(f"activation producer aborted: {self.reference.owner}")
        if not snapshot.has_consumer:
            raise RuntimeError(f"activation lease disappeared: {self.reference.owner}")
        if not snapshot.ready:
            return None
        if snapshot.infer_req_idx < 0:
            raise RuntimeError("producer published KV-ready without infer_req_idx")
        return KVActivation(
            shm_index=self.reference.slot_index,
            request_id=self.reference.generation,
            infer_req_idx=snapshot.infer_req_idx,
            seq_len=snapshot.seq_len,
            position_offset=self.reference.position_offset,
        )

    def release(self) -> None:
        self._require_live()
        try:
            released = self._manager.release_kv_consumer(
                req_index_in_mem=self.reference.slot_index,
                request_id=self.reference.generation,
                consumer_mask=self._store.consumer_mask,
            )
        finally:
            self._released = True
            self._manager.put_back_req_obj(self._shm_req)
        if not released:
            raise RuntimeError("ShmReq generation changed before activation release")

    def abandon(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager.put_back_req_obj(self._shm_req)

    def _require_live(self) -> None:
        if self._released:
            raise RuntimeError(f"activation store lease was released: {self.reference.owner}")


class ShmKVActivationStore:
    """Paged KV backed by the shared LightLLM request table."""

    kind = ActivationKind.KV
    transport = ActivationTransport.CUDA_IPC

    def __init__(
        self,
        shm_req_manager: Any,
        *,
        expected_ready_mask: int,
        consumer_mask: int = KV_CONSUMER_GENERATION,
    ) -> None:
        if expected_ready_mask <= 0:
            raise ValueError("expected_ready_mask must be positive")
        self.manager = shm_req_manager
        self.expected_ready_mask = expected_ready_mask
        self.consumer_mask = consumer_mask

    def open(self, reference: ActivationRef) -> _ShmKVStoreLease:
        if reference.kind is not self.kind:
            raise TypeError("ShmKVActivationStore only accepts KV references")
        shm_req = self.manager.get_req_obj_by_index(reference.slot_index)
        lease = _ShmKVStoreLease(self, reference, shm_req)
        try:
            if not lease.snapshot().has_consumer:
                raise RuntimeError(
                    "activation consumer was not registered before router fan-out"
                )
        except BaseException:
            lease.abandon()
            raise
        return lease


class PagedKVPublisher:
    """Producer hook that publishes a rank-local paged-KV commit."""

    def __init__(self, shm_req_manager: Any, *, rank: int) -> None:
        if not 0 <= rank < KV_CONSUMER_MASK_BITS:
            raise ValueError(
                "activation transfer supports at most "
                f"{KV_CONSUMER_MASK_BITS} producer ranks, got {rank}"
            )
        self._manager = shm_req_manager
        self._rank = int(rank)

    def publish(
        self,
        reference: ActivationRef,
        *,
        shm_req: Any,
        infer_req_idx: int,
        committed_tokens: int,
    ) -> bool:
        if reference.kind is not ActivationKind.KV:
            raise TypeError("PagedKVPublisher only publishes KV activations")
        if (
            int(shm_req.index_in_shm_mem) != reference.slot_index
            or int(shm_req.request_id) != reference.generation
        ):
            return False
        if shm_req.kv_consumer_mask == 0 or committed_tokens < shm_req.input_len:
            return False
        rank_bit = 1 << self._rank
        if shm_req.kv_ready_rank_mask & rank_bit:
            return False

        with self._manager.get_req_lock_by_index(reference.slot_index):
            if int(shm_req.request_id) != reference.generation:
                return False
            if shm_req.infer_req_idx == -1:
                shm_req.infer_req_idx = infer_req_idx
            elif shm_req.infer_req_idx != infer_req_idx:
                raise RuntimeError(
                    f"inconsistent activation request row for {reference.generation}: "
                    f"shm={shm_req.infer_req_idx}, local={infer_req_idx}"
                )
            shm_req.kv_ready_rank_mask = int(shm_req.kv_ready_rank_mask) | rank_bit
        return True


__all__ = ["PagedKVPublisher", "ShmKVActivationStore"]
