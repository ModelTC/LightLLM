import json
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from lightllm.utils.config_utils import get_eos_token_ids


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text", "qwen3_5", "llama"])
@pytest.mark.parametrize("generation_eos", [[248046, 248044], 248046, None])
def test_chat_end_and_end_of_text_are_preserved(tmp_path, monkeypatch, model_type, generation_eos):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type, "text_config": {"eos_token_id": 248044}})
    )
    if generation_eos is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": generation_eos}))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(eos_token_id=248046))
    expected = {248044} if model_type == "llama" else {248044, 248046}
    assert set(get_eos_token_ids(str(tmp_path))) == expected
