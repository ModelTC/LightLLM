import asyncio
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.generationserver.action.protocol import (
    ActionOutcome,
    ActionRequest,
    ActionTaskIdentity,
    ActionWorkerAck,
)
from lightllm.server.activation_transfer import KVActivation
from lightllm.server.activation_transfer import (
    ConsumerExecutionError as GenerationBackendError,
)
from lightllm.server.generationserver.image.backend import (
    ImageBackend,
)
from lightllm.server.generationserver.action.backend import (
    ActionBackend,
    ActionGenerationTask,
)
from lightllm.server.generationserver.image.protocol import (
    GeneratedArtifact,
    ImageGenerationInput,
    ImageOutcome,
    ImageWorkerResult,
)


def test_image_backend_converts_public_prefixes_to_the_worker_rpc():
    async def scenario():
        generation = ImageGenerationInput(
            prefix_roles=("cond", "uncond"),
            width=64,
            height=64,
            steps=2,
            seed=7,
        )
        artifact = GeneratedArtifact("image/jpeg", "eA==")
        rpc = SimpleNamespace()

        async def run_task(model_task):
            assert set(model_task.activations) == {"cond", "uncond"}
            return ImageWorkerResult(
                generation_id=generation.generation_id,
                rank=0,
                outcome=ImageOutcome.SUCCESS,
                safe_to_release=True,
                artifacts=[artifact],
                generation_ms=12.0,
            )

        rpc.run_task = run_task
        backend = ImageBackend([rpc])
        result = await backend.generate(
            generation,
            {
                "cond": KVActivation(1, 10, 3, 8),
                "uncond": KVActivation(2, 20, 4, 9),
            },
        )
        assert result.image == [artifact]
        assert result.generation_ms == 12.0

    asyncio.run(scenario())


def test_image_backend_never_guesses_that_a_failed_rpc_is_cuda_safe():
    async def scenario():
        async def fail(_task):
            raise RuntimeError("worker lost")

        generation = ImageGenerationInput(
            prefix_roles=("cond", "uncond"),
            width=64,
            height=64,
            steps=2,
        )
        backend = ImageBackend([SimpleNamespace(run_task=fail)])
        with pytest.raises(GenerationBackendError) as error:
            await backend.generate(
                generation,
                {
                    "cond": KVActivation(1, 10, 3, 8),
                    "uncond": KVActivation(2, 20, 4, 9),
                },
            )
        assert not error.value.safe_to_release

    asyncio.run(scenario())


def test_pi_backend_keeps_action_model_details_out_of_the_manager():
    async def scenario():
        config = {
            "model_type": "pi0",
            "action_dim": 1,
            "action_horizon": 2,
            "num_denoise_steps": 1,
            "max_action_dim": 32,
            "action_expert_uses_state": True,
        }
        identity = ActionTaskIdentity(3, 17, 17)
        request = ActionRequest(state=[0.5])
        processor = SimpleNamespace(
            normalize_state=lambda state: torch.tensor(state),
            postprocess_actions=lambda actions, **_kwargs: actions,
        )

        async def run_task(model_task):
            assert model_task.identity == identity
            assert model_task.prefix_kv == KVActivation(3, 17, 5, 11)
            assert model_task.noisy_actions.shape == (1, 2, config["max_action_dim"])
            return ActionWorkerAck(
                slot_index=3,
                generation=17,
                task_id=17,
                rank=0,
                outcome=ActionOutcome.SUCCESS,
                safe_to_release=True,
                actions=torch.tensor([[[1.0], [2.0]]]),
                action_expert_ms=4.0,
            )

        backend = ActionBackend(
            [SimpleNamespace(run_task=run_task)],
            processor,
            tp_size=1,
        )
        task = ActionGenerationTask(
            task_id="action",
            action_horizon=2,
            action_dimension=1,
            identity=identity,
            action_request=request,
            request_config=config,
            seed=7,
        )
        result = await backend.generate(
            task,
            KVActivation(3, 17, 5, 11),
        )
        torch.testing.assert_close(result.actions, torch.tensor([[1.0], [2.0]]))
        assert result.action_expert_ms == 4.0

    asyncio.run(scenario())


def test_action_postprocess_failure_preserves_cuda_safe_ack():
    async def scenario():
        config = {
            "model_type": "pi0",
            "action_dim": 1,
            "action_horizon": 1,
            "num_denoise_steps": 1,
            "max_action_dim": 32,
            "action_expert_uses_state": False,
        }
        identity = ActionTaskIdentity(3, 17, 17)
        request = ActionRequest(state=[0.5])

        async def run_task(_model_task):
            return ActionWorkerAck(
                slot_index=3,
                generation=17,
                task_id=17,
                rank=0,
                outcome=ActionOutcome.SUCCESS,
                safe_to_release=True,
                actions=torch.zeros(1, 1, 1),
            )

        def fail_postprocess(*_args, **_kwargs):
            raise ValueError("invalid domain output")

        backend = ActionBackend(
            [SimpleNamespace(run_task=run_task)],
            SimpleNamespace(postprocess_actions=fail_postprocess),
            tp_size=1,
        )
        task = ActionGenerationTask(
            task_id="action-postprocess",
            action_horizon=1,
            action_dimension=1,
            identity=identity,
            action_request=request,
            request_config=config,
            seed=7,
        )

        with pytest.raises(GenerationBackendError) as error:
            await backend.generate(
                task,
                KVActivation(3, 17, 5, 11),
            )
        assert error.value.safe_to_release

    asyncio.run(scenario())


def test_pi_task_preserves_lightllm_default_signed_seed():
    config = {
        "model_type": "pi0",
        "action_dim": 1,
        "action_horizon": 1,
        "num_denoise_steps": 1,
        "max_action_dim": 32,
        "action_expert_uses_state": True,
    }
    request = ActionRequest(state=[0.0])
    task = ActionGenerationTask(
        task_id="signed-seed",
        action_horizon=1,
        action_dimension=1,
        identity=ActionTaskIdentity(3, 17, 17),
        action_request=request,
        request_config=config,
        seed=-1,
    )
    assert task.seed == -1


def test_pi05_backend_does_not_repeat_prefix_state_normalization():
    config = {
        "model_type": "pi05",
        "action_dim": 1,
        "action_horizon": 1,
        "num_denoise_steps": 1,
        "max_action_dim": 32,
        "action_expert_uses_state": False,
    }
    request = ActionRequest(state=[0.0])
    processor = SimpleNamespace(
        normalize_state=lambda _state: pytest.fail(
            "Pi0.5 state was already normalized into the prefix"
        )
    )
    backend = ActionBackend([], processor, tp_size=1)
    task = ActionGenerationTask(
        task_id="pi05-state",
        action_horizon=1,
        action_dimension=1,
        identity=ActionTaskIdentity(3, 17, 17),
        action_request=request,
        request_config=config,
        seed=7,
    )

    prepared = backend._prepare_model_task(
        task,
        KVActivation(3, 17, 5, 11),
    )

    assert prepared.state is None
