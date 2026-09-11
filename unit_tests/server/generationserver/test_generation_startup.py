import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightllm.models.neo_chat.generation import NeoImageGenerationModel
from lightllm.models.pi0.model import Pi0ActionExpertModel
from lightllm.models.pi05.model import Pi05ActionExpertModel
from lightllm.models.registry import (
    GenerationModelRegistry,
    get_generation_model_class,
    get_output_modality,
)
from lightllm.server.generationserver import image
from lightllm.server.generationserver.action import io as action_io


def test_omni_startup_is_inert_for_ordinary_models():
    assert get_output_modality({"model_type": "qwen3"}) is None


def test_omni_implementations_are_selected_by_model_type():
    assert get_output_modality({"type": "pi0"}) == "action"
    assert get_output_modality({"type": "pi05"}) == "action"
    assert (
        get_output_modality(
            {"model_type": "neo_chat", "llm_config": {"model_type": "qwen3"}}
        )
        == "image"
    )
    assert get_generation_model_class({"type": "pi0"}, "action") is Pi0ActionExpertModel
    assert (
        get_generation_model_class({"type": "pi05"}, "action") is Pi05ActionExpertModel
    )
    assert (
        get_generation_model_class(
            {"model_type": "neo_chat", "llm_config": {"model_type": "qwen3"}},
            "image",
        )
        is NeoImageGenerationModel
    )


def test_generation_registry_rejects_duplicate_model_capabilities():
    registry = type(GenerationModelRegistry)()

    @registry("example", output_modality="action")
    class ExampleActionModel:
        pass

    with pytest.raises(ValueError, match="duplicate generation model"):
        registry("example", output_modality="action")(ExampleActionModel)


def test_image_mode_only_applies_required_generation_defaults():
    args = SimpleNamespace(
        generation_backend="x2v",
        generation_model_dir=None,
        model_dir="/model",
        disable_dynamic_prompt_cache=False,
        disable_chunked_prefill=False,
    )

    image.configure_defaults(args)

    assert args.generation_model_dir == args.model_dir
    assert args.disable_dynamic_prompt_cache
    assert args.disable_chunked_prefill


def test_action_io_is_selected_from_the_registered_model(monkeypatch):
    class FakeProcessor:
        def __init__(self, config):
            self.config = config

    class FakeActionModel:
        generation_processor_class = FakeProcessor

        @staticmethod
        def generation_config_loader(args):
            return {"model_type": "fake_action", "model_dir": args.model_dir}

    monkeypatch.setattr(
        action_io, "get_config_json", lambda _model_dir: {"type": "fake_action"}
    )
    monkeypatch.setattr(
        action_io, "_action_model_class", lambda _config: FakeActionModel
    )

    config = action_io.load_action_config(SimpleNamespace(model_dir="/fake"))
    processor = action_io.create_action_processor(config)

    assert config == {"model_type": "fake_action", "model_dir": "/fake"}
    assert isinstance(processor, FakeProcessor)
    assert processor.config is config


def test_common_action_flow_does_not_import_concrete_models():
    action_dir = Path(action_io.__file__).parent
    concrete_imports = []
    for path in action_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        concrete_imports.extend(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("lightllm.models.")
            and node.module != "lightllm.models.registry"
        )
    assert concrete_imports == []


def test_common_image_flow_does_not_import_concrete_models():
    image_dir = Path(image.__file__).parent
    concrete_imports = []
    for path in image_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        concrete_imports.extend(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("lightllm.models.")
            and node.module != "lightllm.models.registry"
        )
    assert concrete_imports == []
