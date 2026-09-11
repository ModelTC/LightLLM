import base64
from types import SimpleNamespace

import pytest

from lightllm.server.activation_transfer import (
    ActivationRef,
    ActivationTransport,
    HiddenStatesActivation,
    HiddenStatesDescriptor,
)
from lightllm.models.neo_chat.tokenizer import (
    IMG_START_TOKEN,
    IMG_TOKEN,
    NeoChatTokenizer,
)
from lightllm.server.generationserver.image.protocol import (
    GeneratedArtifact,
    ImageActivationTask,
    ImageGenerationInput,
    ImageModelTask,
    ImageOutcome,
    ImageResult,
)
from lightllm.server.httpserver.omni_adapter import _normalize_image


def test_generation_input_validates_backend_neutral_fields():
    generation = ImageGenerationInput(
        prefix_roles=("cond", "uncond"),
        width=512,
        height=768,
        steps=8,
        num_outputs=2,
    )

    assert generation.prefix_roles == ("cond", "uncond")
    assert generation.to_dict()["prefix_roles"] == ["cond", "uncond"]
    assert generation.to_dict()["cfg_interval"] == [-1.0, 2.0]

    with pytest.raises(ValueError, match="prefix_roles"):
        ImageGenerationInput(prefix_roles=("same", "same"))
    with pytest.raises(ValueError, match="divisible by 32"):
        ImageGenerationInput(prefix_roles=("cond", "uncond"), width=513)
    with pytest.raises(ValueError, match="positive integer"):
        ImageGenerationInput(prefix_roles=("cond", "uncond"), steps=True)
    with pytest.raises(ValueError, match="ordered"):
        ImageGenerationInput(prefix_roles=("cond", "uncond"), cfg_interval=(1.0, 0.0))


def test_image_protocol_accepts_hidden_activation_descriptors():
    generation = ImageGenerationInput(
        prefix_roles=("condition",),
        width=64,
        height=64,
    )
    reference = ActivationRef.hidden_states(
        2,
        17,
        transport=ActivationTransport.CUDA_IPC,
    )
    task = ImageActivationTask(
        activation=reference,
        role="condition",
        generation=generation,
    )
    activation = HiddenStatesActivation(
        reference,
        HiddenStatesDescriptor(
            shape=(1, 32, 2048),
            dtype="bfloat16",
            storage_handle=b"cuda-ipc-handle",
            device_index=0,
            nbytes=131072,
            ready_event_handle=b"cuda-event-handle",
        ),
    )

    model_task = ImageModelTask(
        generation=generation,
        activations={task.role: activation},
    )

    assert model_task.activations["condition"].descriptor.shape == (1, 32, 2048)


def test_generation_result_serializes_artifacts_without_backend_objects():
    artifact = GeneratedArtifact("image/jpeg", base64.b64encode(b"jpeg").decode())
    result = ImageResult(
        correlation_id="corr",
        generation_id="generation",
        request_id="request",
        outcome=ImageOutcome.SUCCESS,
        artifacts=[artifact],
        generation_ms=12.5,
    )

    assert result.ok
    assert result.to_dict()["data"] == [
        {
            "type": "image",
            "mime_type": "image/jpeg",
            "b64_json": artifact.data,
        }
    ]


class _FakeTokenizer:
    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"chat:{conversation[0]['content']}"


def _neo_tokenizer():
    tokenizer = NeoChatTokenizer.__new__(NeoChatTokenizer)
    tokenizer.tokenizer = _FakeTokenizer()
    return tokenizer


def test_neo_tokenizer_owns_cond_uncond_query_planning():
    image = {"type": "url", "data": "https://example.invalid/input.jpg"}
    prefixes = _neo_tokenizer().build_generation_prefixes(
        prompt="draw a variation",
        images=[image],
        generation=ImageGenerationInput(
            prefix_roles=("cond", "uncond"),
            width=512,
            height=512,
        ),
    )

    assert [prefix.role for prefix in prefixes] == ["cond", "uncond"]
    assert prefixes[0].prompt == f"chat:{IMG_TOKEN}\ndraw a variation{IMG_START_TOKEN}"
    assert prefixes[1].prompt == f"chat:{IMG_TOKEN}{IMG_START_TOKEN}"
    assert prefixes[0].images == prefixes[1].images == (image,)


def test_normalize_image_stays_in_the_omni_http_layer():
    payload = base64.b64encode(b"image bytes").decode()
    assert _normalize_image(f"data:image/png;base64,{payload}") == {
        "type": "base64",
        "data": payload,
    }
    assert _normalize_image("https://example.invalid/image.png") == {
        "type": "url",
        "data": "https://example.invalid/image.png",
    }
    with pytest.raises(ValueError):
        _normalize_image("not base64")


def test_generation_position_offset_is_independent_of_physical_kv_length():
    images = [
        SimpleNamespace(grid_thwd=(1, 4, 4, -15)),
        SimpleNamespace(grid_thwd=(1, 2, 3, -5)),
    ]
    multimodal = SimpleNamespace(images=images)
    tokenizer = _neo_tokenizer()

    # 40 physical prefix tokens include 16 + 6 visual tokens, while each
    # image occupies one logical temporal position for the following RoPE.
    assert tokenizer.get_generation_position_offset(40, multimodal) == 20

    images[0].grid_thwd = None
    with pytest.raises(ValueError, match="position metadata"):
        tokenizer.get_generation_position_offset(40, multimodal)
