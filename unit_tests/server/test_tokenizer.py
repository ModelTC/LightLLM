"""Tokenizer entry points must not eagerly load unrelated model implementations."""

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("model_type", ["llama", "deepseek_v32"])
def test_tokenizer_imports_are_isolated(model_type):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoConfig, PreTrainedTokenizerFast

model_type = sys.argv[1]
class RejectUnrelatedModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("lightllm.models."):
            allowed = {"lightllm.models.registry", "lightllm.models.draft_registry", "lightllm.models.builtin"}
            if model_type == "deepseek_v32":
                allowed.add("lightllm.models.deepseek3_2")
            if fullname not in allowed:
                raise AssertionError("Unexpected model import: " + fullname)
sys.meta_path.insert(0, RejectUnrelatedModels())
from lightllm.server import tokenizer as module
assert not any(name.startswith("lightllm.models.") for name in sys.modules)

if model_type == "deepseek_v32":
    # Keep the real HF config registration, replacing only the implementation wrapper.
    wrapper = ModuleType("lightllm.models.deepseek3_2.model")
    wrapper.DeepSeekV32Tokenizer = lambda tokenizer: tokenizer
    sys.modules[wrapper.__name__] = wrapper
    original = module.AutoTokenizer.from_pretrained
    def load(*args, **kwargs):
        assert AutoConfig.for_model("deepseek_v32").model_type == "deepseek_v32"
        return original(*args, **kwargs)
    module.AutoTokenizer.from_pretrained = load

with tempfile.TemporaryDirectory() as directory:
    fast = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"hello": 0, "[UNK]": 1}, unk_token="[UNK]")))
    fast.save_pretrained(directory)
    Path(directory, "config.json").write_text(json.dumps({
        "model_type": model_type, "architectures": ["LlamaForCausalLM"],
    }))
    tokenizer = module.get_tokenizer(directory)
    assert tokenizer.encode("hello") == [0]
""",
            model_type,
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "config,module_name,class_name",
    [
        (
            {"model_type": "llava", "architectures": ["TarsierForConditionalGeneration"]},
            "tarsier2.model",
            "Tarsier2Tokenizer",
        ),
        ({"model_type": "llava"}, "llava.model", "LlavaTokenizer"),
        ({"model_type": "internlmxcomposer2"}, "llava.model", "LlavaTokenizer"),
        ({"model_type": "qwen", "visual": {}}, "qwen_vl.model", "QWenVLTokenizer"),
        ({"model_type": "qwen2_vl", "vision_config": {}}, "qwen2_vl.model", "QWen2VLTokenizer"),
        ({"model_type": "qwen2_5_vl", "vision_config": {}}, "qwen2_vl.model", "QWen2VLTokenizer"),
        ({"model_type": "qwen3_vl", "vision_config": {}}, "qwen3_vl.model", "QWen3VLTokenizer"),
        ({"model_type": "qwen3_vl_moe", "vision_config": {}}, "qwen3_vl.model", "QWen3VLTokenizer"),
        ({"model_type": "qwen3_5", "vision_config": {}}, "qwen3_5.model", "QWen3_5Tokenizer"),
        ({"model_type": "qwen3_5_moe", "vision_config": {}}, "qwen3_5.model", "QWen3_5Tokenizer"),
        ({"thinker_config": {}}, "qwen3_omni_moe_thinker.model", "QWen3OmniTokenizer"),
        ({"model_type": "internvl_chat"}, "internvl.model", "InternvlTokenizer"),
        ({"model_type": "gemma3"}, "gemma3.model", "Gemma3Tokenizer"),
        ({"model_type": "gemma4"}, "gemma4.tokenizer", "Gemma4Tokenizer"),
        ({"model_type": "gemma4", "vision_config": {}}, "gemma4.tokenizer", "Gemma4Tokenizer"),
    ],
)
def test_selected_multimodal_tokenizer_is_loaded(monkeypatch, config, module_name, class_name):
    import transformers
    from lightllm.server import tokenizer as module

    config = {"architectures": ["Example"], **config}
    base_tokenizer = object()
    monkeypatch.setattr(module, "load_model_config_dict", lambda *a, **kw: config)
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **kw: base_tokenizer)
    processor = SimpleNamespace(image_processor=object())
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    vision = ModuleType("lightllm.models.qwen2_vl.vision_process")
    vision.Qwen2VLImageProcessor = SimpleNamespace(from_pretrained=lambda *a, **kw: processor.image_processor)
    monkeypatch.setitem(sys.modules, vision.__name__, vision)
    implementation = ModuleType("lightllm.models." + module_name)
    constructor = Mock()
    setattr(implementation, class_name, constructor)
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)

    assert module.get_tokenizer("local-checkpoint") is constructor.return_value
    constructor.assert_called_once()
    args, kwargs = constructor.call_args
    assert (args[0] if args else kwargs["tokenizer"]) is base_tokenizer
    assert (args[1] if len(args) > 1 else kwargs["model_cfg"]) is config.get("thinker_config", config)
