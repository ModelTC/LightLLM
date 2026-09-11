import asyncio
import pickle
from types import SimpleNamespace

import pytest

from lightllm.server.activation_transfer import ActivationRef
from lightllm.server.generationserver.image.client import (
    ImageGenerationClient,
    ImageDispatchNotSent,
)
from lightllm.server.generationserver.action.client import (
    ActionDispatchNotSent,
    ActionGenerationClient,
)
from lightllm.server.generationserver.action.protocol import (
    ActionCancel,
    ActionOutcome,
    ActionRequest,
    ActionResult,
    ActionTask,
)
from lightllm.server.generationserver.image.protocol import (
    ImageCancel,
    ImageActivationTask,
    ImageGenerationInput,
    ImageOutcome,
    ImageResult,
)


CLIENT_CASES = (ActionGenerationClient, ImageGenerationClient)


def _client(client_type, socket):
    client = client_type.__new__(client_type)
    client.socket = socket
    client._pending = {}
    client._send_lock = asyncio.Lock()
    client._receiver_task = None
    client._ensure_receiver = lambda: None
    return client


def _message(client_type, correlation_id="corr-1"):
    if client_type is ActionGenerationClient:
        return ActionTask(
            activation=ActivationRef.kv(3, 17),
            action_request=ActionRequest(state=[0.0]),
            correlation_id=correlation_id,
        )
    return ImageActivationTask(
        activation=ActivationRef.kv(3, 17, position_offset=11),
        role="cond",
        generation=ImageGenerationInput(
            prefix_roles=("cond", "uncond"),
            width=512,
            height=512,
        ),
        correlation_id=correlation_id,
    )


def _result(client_type, message):
    if client_type is ActionGenerationClient:
        return ActionResult(
            correlation_id=message.correlation_id,
            request_id=None,
            outcome=ActionOutcome.SUCCESS,
        )
    return ImageResult(
        correlation_id=message.correlation_id,
        generation_id=message.generation.generation_id,
        request_id=message.generation.request_id,
        outcome=ImageOutcome.SUCCESS,
    )


@pytest.mark.parametrize("client_type", CLIENT_CASES)
def test_dispatch_returns_waiter_only_after_send(client_type):
    async def scenario():
        class Socket:
            def __init__(self):
                self.sent = []

            async def send(self, payload):
                self.sent.append(pickle.loads(payload))

        socket = Socket()
        client = _client(client_type, socket)
        message = _message(client_type)
        waiter = await client.dispatch(message, timeout=1.0)
        assert socket.sent == [message]

        client._pending[message.correlation_id].set_result(
            _result(client_type, message)
        )
        assert (await waiter).ok
        assert message.correlation_id not in client._pending

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("client_type", "cancel_type"),
    ((ActionGenerationClient, ActionCancel), (ImageGenerationClient, ImageCancel)),
)
def test_cancel_during_dispatch_finishes_send_then_cancels_owner(
    client_type, cancel_type
):
    async def scenario():
        class Socket:
            def __init__(self):
                self.sent = []
                self.first_started = asyncio.Event()
                self.release_first = asyncio.Event()

            async def send(self, payload):
                message = pickle.loads(payload)
                if not self.sent:
                    self.first_started.set()
                    await self.release_first.wait()
                self.sent.append(message)

        socket = Socket()
        client = _client(client_type, socket)
        message = _message(client_type, "cancel-during-send")
        dispatch = asyncio.create_task(client.dispatch(message, timeout=1.0))
        await socket.first_started.wait()
        dispatch.cancel()
        socket.release_first.set()

        with pytest.raises(asyncio.CancelledError):
            await dispatch
        assert socket.sent[0] == message
        assert isinstance(socket.sent[1], cancel_type)
        assert socket.sent[1].correlation_id == message.correlation_id
        assert message.correlation_id not in client._pending

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("client_type", "error_type"),
    (
        (ActionGenerationClient, ActionDispatchNotSent),
        (ImageGenerationClient, ImageDispatchNotSent),
    ),
)
def test_send_failure_is_explicitly_not_owned(client_type, error_type):
    async def scenario():
        class Socket:
            async def send(self, _payload):
                raise OSError("send failed")

        client = _client(client_type, Socket())
        message = _message(client_type, "not-sent")
        with pytest.raises(error_type):
            await client.dispatch(message, timeout=1.0)
        assert message.correlation_id not in client._pending

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("client_type", "error_type"),
    (
        (ActionGenerationClient, ActionDispatchNotSent),
        (ImageGenerationClient, ImageDispatchNotSent),
    ),
)
def test_pre_send_validation_is_explicitly_not_owned(client_type, error_type):
    async def scenario():
        class Socket:
            def __init__(self):
                self.sent = []

            async def send(self, payload):
                self.sent.append(payload)

        socket = Socket()
        client = _client(client_type, socket)

        with pytest.raises(error_type):
            await client.dispatch(SimpleNamespace(correlation_id=""), timeout=1.0)

        duplicate = _message(client_type, "duplicate")
        client._pending[
            duplicate.correlation_id
        ] = asyncio.get_running_loop().create_future()
        with pytest.raises(error_type):
            await client.dispatch(duplicate, timeout=1.0)

        assert socket.sent == []

    asyncio.run(scenario())
