import asyncio
from unittest.mock import AsyncMock

import pytest
import torch

from lightllm.server.activation_transfer import ActivationRef, KVActivation
from lightllm.server.generationserver.action.backend import ActionGenerationResult
from lightllm.server.generationserver.action.protocol import (
    ActionRequest,
    ActionTask,
    PrefixContextIdentity,
    PrefixContextOp,
    PrefixContextRef,
)
from lightllm.server.generationserver.action.server import (
    VLAServer,
    _ActionContextMetadata,
    _ActionExecutionResult,
    _ActionSubmission,
    _VLAContext,
)
def _context(*, domain="libero"):
    return _VLAContext(
        context_id="robot",
        version=1,
        lease=object(),
        handle=KVActivation(3, 17, 5, 11),
        metadata=_ActionContextMetadata(domain),
    )


def test_vla_server_validates_context_identity_without_a_generic_manager():
    server = VLAServer.__new__(VLAServer)
    server.server_epoch = "epoch"
    context = _context()
    server._contexts = {"robot": context}

    identity = PrefixContextIdentity("robot", 1, "epoch")
    request = ActionRequest(
        state=[0.0],
        prefix_context=PrefixContextRef(PrefixContextOp.REUSE, identity=identity),
    )
    assert server._validate_context_identity(request) is context

    stale = ActionRequest(
        state=[0.0],
        prefix_context=PrefixContextRef(
            PrefixContextOp.REUSE,
            identity=PrefixContextIdentity("robot", 1, "old-epoch"),
        ),
    )
    with pytest.raises(RuntimeError, match="another server"):
        server._validate_context_identity(stale)


def test_persistent_context_keeps_its_action_domain():
    context = _context(domain="libero")
    inherited = ActionRequest(state=[0.0], domain=None)
    VLAServer._apply_context_domain(inherited, context)
    assert inherited.domain == "libero"

    with pytest.raises(ValueError, match="context domain"):
        VLAServer._apply_context_domain(
            ActionRequest(state=[0.0], domain="droid"),
            context,
        )


def test_vla_result_is_adapted_to_the_http_wire_response():
    async def scenario():
        server = VLAServer.__new__(VLAServer)
        server._submissions = {}
        server._send_result = AsyncMock()
        request = ActionRequest(state=[0.0], request_id="request")
        task = ActionTask(
            ActivationRef.kv(3, 17),
            request,
            correlation_id="corr",
        )
        submission = _ActionSubmission(
            task=task,
            route=(b"http",),
            request_config={"action_horizon": 2, "action_dim": 1},
        )
        server._submissions[task.correlation_id] = submission
        execution = asyncio.get_running_loop().create_future()
        execution.set_result(
            _ActionExecutionResult(
                ActionGenerationResult(
                    task_id="corr",
                    actions=torch.tensor([[1.0], [2.0]]),
                    action_expert_ms=3.0,
                    backend_ms=4.0,
                ),
                None,
            )
        )

        await server._relay_result(submission, execution)
        response = server._send_result.await_args.args[1]
        assert response.ok
        torch.testing.assert_close(response.actions, torch.tensor([[1.0], [2.0]]))
        assert response.policy_timing["action_expert_ms"] == 3.0
        assert response.policy_timing["action_backend_ms"] == 4.0
        assert response.policy_timing["action_overhead_ms"] == 1.0
        assert "corr" not in server._submissions

    asyncio.run(scenario())
