import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server.activation_transfer import ActivationRef
from lightllm.server.activation_transfer.channel import ActivationChannel
from lightllm.server.core.objs import KV_CONSUMER_GENERATION
from lightllm.server.activation_transfer import (
    ConsumerExecutionError as GenerationBackendError,
)
from lightllm.server.generationserver.image.backend import ImageGenerationResult
from lightllm.server.generationserver.image.protocol import (
    ImageActivationTask,
    ImageGenerationInput,
    ImageOutcome,
)
from lightllm.server.generationserver.image.server import (
    ImageGenerationServer,
    _GenerationGroup,
    _PrefixRecord,
)
from lightllm.server.activation_transfer import KVActivation


def _prefix_task(generation, role, slot, request_id):
    return ImageActivationTask(
        activation=ActivationRef.kv(
            slot,
            request_id,
            position_offset=23 if role == "cond" else 7,
        ),
        role=role,
        generation=generation,
        correlation_id=f"{role}-corr",
    )


class _Lease:
    def __init__(self, handle):
        self.handle = handle
        self.claimed = False
        self.released = False

    def claim(self):
        self.claimed = True

    def mark_in_use(self):
        self.claimed = True

    def resolve(self):
        return self.handle

    def release(self):
        self.released = True


def _group(generation):
    records = {}
    for role, slot in (("cond", 2), ("uncond", 3)):
        records[role] = _PrefixRecord(
            _prefix_task(generation, role, slot, 10 + slot),
            (role.encode(),),
            _Lease(KVActivation(slot, 10 + slot, slot + 10, 8)),
        )
    return _GenerationGroup(generation, records)


def test_complete_image_group_is_owned_by_the_image_server():
    async def scenario():
        generation = ImageGenerationInput(
            prefix_roles=("cond", "uncond"),
            width=512,
            height=512,
            timeout=5.0,
        )
        group = _group(generation)
        server = ImageGenerationServer.__new__(ImageGenerationServer)
        server._relay_tasks = set()
        server._run_group = AsyncMock()

        await server._dispatch_group(group)
        await asyncio.gather(*server._relay_tasks)

        assert group.dispatched
        assert all(record.lease.claimed for record in group.prefixes.values())
        assert server._run_group.await_args.args == (group,)

    asyncio.run(scenario())


def test_image_server_releases_dense_generation_inputs_after_safe_ack():
    async def scenario():
        generation = ImageGenerationInput(
            prefix_roles=("cond", "uncond"),
            width=64,
            height=64,
            timeout=5.0,
        )
        group = _group(generation)
        for record in group.prefixes.values():
            record.lease.claim()
        server = ImageGenerationServer.__new__(ImageGenerationServer)
        server._concurrency = asyncio.Semaphore(1)
        server.restart_required = False
        server.backend = SimpleNamespace(
            generate=AsyncMock(
                return_value=ImageGenerationResult(
                    generation.generation_id,
                    [],
                    2.0,
                )
            )
        )
        server._reply_group = AsyncMock()
        server._forget_group = lambda _group: None
        await server._run_group(group)

        assert all(record.lease.released for record in group.prefixes.values())
        assert not server.restart_required

    asyncio.run(scenario())


def test_image_server_pins_inputs_when_worker_ack_is_unsafe():
    async def scenario():
        generation = ImageGenerationInput(
            prefix_roles=("cond", "uncond"),
            width=64,
            height=64,
            timeout=5.0,
        )
        group = _group(generation)
        for record in group.prefixes.values():
            record.lease.claim()
        server = ImageGenerationServer.__new__(ImageGenerationServer)
        server._concurrency = asyncio.Semaphore(1)
        server.restart_required = False
        server.backend = SimpleNamespace(
            generate=AsyncMock(
                side_effect=GenerationBackendError("unsafe", safe_to_release=False)
            )
        )
        server._reply_group = AsyncMock()
        server._forget_group = lambda _group: None

        await server._run_group(group)

        assert not any(record.lease.released for record in group.prefixes.values())
        assert server.restart_required
        assert server._reply_group.await_args.args[1] is ImageOutcome.RESTART_REQUIRED

    asyncio.run(scenario())


def test_restart_rejection_adopts_and_releases_the_registered_consumer():
    async def scenario():
        generation = ImageGenerationInput(
            prefix_roles=("cond",),
            width=64,
            height=64,
            timeout=1.0,
        )
        task = _prefix_task(generation, "cond", 3, 17)
        req = SimpleNamespace(
            index_in_shm_mem=3,
            request_id=17,
            kv_ready_rank_mask=1,
            kv_consumer_mask=KV_CONSUMER_GENERATION,
            infer_req_idx=5,
            input_len=11,
            is_aborted=False,
            sample_params=SimpleNamespace(seed=7),
        )
        events = []
        shm_manager = SimpleNamespace(
            get_req_obj_by_index=lambda _slot: req,
            get_req_lock_by_index=lambda _slot: nullcontext(),
            release_kv_consumer=lambda **kwargs: events.append(("release", kwargs))
            or True,
            put_back_req_obj=lambda _req: events.append(("put_back", 17)),
        )
        server = ImageGenerationServer.__new__(ImageGenerationServer)
        server.transfer = ActivationChannel.for_shm_kv(
            shm_manager,
            expected_ready_mask=1,
        )
        server.by_correlation = {}
        server.restart_required = True

        with pytest.raises(RuntimeError, match="restart"):
            await server._accept_prefix(task, (b"http",))

        assert len(server.transfer) == 0
        assert [event[0] for event in events] == ["release", "put_back"]

    asyncio.run(scenario())


def test_duplicate_role_with_new_correlation_gets_its_own_error_reply():
    async def scenario():
        generation = ImageGenerationInput(
            prefix_roles=("cond", "uncond"),
            width=512,
            height=512,
            timeout=5.0,
        )
        original = _prefix_task(generation, "cond", 2, 12)
        duplicate = ImageActivationTask(
            activation=ActivationRef.kv(3, 13, position_offset=23),
            role="cond",
            generation=generation,
            correlation_id="new-correlation",
        )
        group = _GenerationGroup(
            generation,
            {
                "cond": _PrefixRecord(
                    original,
                    (b"old-route",),
                    SimpleNamespace(),
                )
            },
        )
        group.dispatched = True
        server = ImageGenerationServer.__new__(ImageGenerationServer)
        server.groups = {generation.generation_id: group}
        server._send = AsyncMock()

        await server._reject_prefix(
            duplicate,
            (b"new-route",),
            ValueError("duplicate generation prefix role"),
        )

        route, result = server._send.await_args.args
        assert route == (b"new-route",)
        assert result.correlation_id == "new-correlation"
        assert result.outcome is ImageOutcome.ERROR

    asyncio.run(scenario())
