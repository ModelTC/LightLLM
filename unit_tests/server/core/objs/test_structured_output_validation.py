from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
import xgrammar

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.deepseek3_2.model import DeepSeekV32Tokenizer
from lightllm.server.core.objs import sampling_params
from lightllm.server.tokenizer import get_xgrammar_tokenizer


@pytest.mark.parametrize("mode", ["none", "outlines"])
@pytest.mark.parametrize("constraint", [{"guided_json": "{}"}, {"guided_grammar": "json"}])
def test_constraints_are_rejected_when_xgrammar_is_not_enabled(monkeypatch, mode, constraint):
    monkeypatch.setattr(sampling_params, "get_env_start_args", lambda: SimpleNamespace(output_constraint_mode=mode))
    with pytest.raises(ValueError, match="requires.*xgrammar"):
        sampling_params.SamplingParams().init(tokenizer=None, **constraint)


def test_guided_json_checks_utf8_bytes_and_raises_validation_error():
    with pytest.raises(ValueError, match="too long"):
        sampling_params.GuidedJsonSchema().initialize("a" * sampling_params.JSON_SCHEMA_MAX_LENGTH, None)
    with pytest.raises(ValueError, match="too long"):
        sampling_params.GuidedJsonSchema().initialize("中" * sampling_params.JSON_SCHEMA_MAX_LENGTH, None)
    schema = sampling_params.GuidedJsonSchema()
    schema.initialize("a" * (sampling_params.JSON_SCHEMA_MAX_LENGTH - 1), None)
    assert schema.length == sampling_params.JSON_SCHEMA_MAX_LENGTH - 1


def test_wrapped_fast_tokenizer_compiles_real_xgrammar(monkeypatch):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "a": 2}, unk_token="<unk>")),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    wrapper = DeepSeekV32Tokenizer(tokenizer)
    assert get_xgrammar_tokenizer(wrapper) is tokenizer
    assert get_xgrammar_tokenizer(tokenizer) is tokenizer
    # xgrammar inspects the concrete HF type, so delegation alone is insufficient.
    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        xgrammar.TokenizerInfo.from_huggingface(wrapper)
    grammar = sampling_params.GuidedGrammar()
    grammar.initialize('root ::= "a"', wrapper)
    json_schema = sampling_params.GuidedJsonSchema()
    json_schema.initialize('{"type":"string"}', wrapper)
    monkeypatch.setattr(
        sampling_params, "get_env_start_args", lambda: SimpleNamespace(output_constraint_mode="xgrammar")
    )
    params = sampling_params.SamplingParams()
    params.init(tokenizer=wrapper, guided_grammar="json")
    assert params.guided_grammar.to_str() == "json"


def test_multimodal_base_exposes_the_same_text_tokenizer():
    # Exercise the base constructor without loading an image processor.
    wrapper = SimpleNamespace()
    tokenizer = object()
    BaseMultiModalTokenizer.__init__(wrapper, tokenizer)
    assert get_xgrammar_tokenizer(wrapper) is tokenizer
