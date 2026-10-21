import sys
from types import ModuleType

import pytest

from lightllm import models
from lightllm.models.registry import ModelRegistry


@pytest.mark.parametrize(
    "llm_type,module_name,class_name",
    [
        ("qwen3", "lightllm.models.neo_chat.model", "NeoTpPartModel"),
        ("qwen3_moe", "lightllm.models.neo_chat_moe.model", "NeoTpMOEPartModel"),
    ],
)
def test_neo_models_resolve_through_lazy_registry(monkeypatch, llm_type, module_name, class_name):
    config = {"model_type": "neo_chat", "llm_config": {"model_type": llm_type}}
    registration = ModelRegistry.get_model_config(config)
    assert registration.is_multimodal
    assert registration.model_class == f"{module_name}:{class_name}"

    implementation = ModuleType(module_name)
    model_class = type(class_name, (), {})
    setattr(implementation, class_name, model_class)
    monkeypatch.setitem(sys.modules, module_name, implementation)

    assert models.get_model_class(config) is model_class
    assert getattr(models, class_name) is model_class
    assert models.get_model_class(config) is model_class
