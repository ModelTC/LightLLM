import pytest

from lightllm.utils import config_utils


@pytest.fixture
def fake_config(monkeypatch):
    def install(cfg_json):
        monkeypatch.setattr(config_utils, "get_config_json", lambda model_path: cfg_json)
        return "dummy/model/dir"

    return install


def test_get_keyvalue_reads_top_level_key(fake_config):
    model_path = fake_config({"model_type": "llama", "hidden_size": 4096})

    assert config_utils._get_config_llm_keyvalue(model_path, ["hidden_size"]) == 4096


def test_get_keyvalue_does_not_crash_when_thinker_config_lacks_text_config(fake_config):
    cfg = {"model_type": "qwen3_omni_moe", "hidden_size": 4096, "thinker_config": {"model_type": "text"}}
    model_path = fake_config(cfg)

    assert config_utils._get_config_llm_keyvalue(model_path, ["hidden_size"]) == 4096


def test_get_keyvalue_keeps_top_level_value_when_thinker_text_config_lacks_key(fake_config):
    cfg = {
        "model_type": "qwen3_omni_moe",
        "hidden_size": 4096,
        "thinker_config": {"text_config": {"model_type": "text"}},
    }
    model_path = fake_config(cfg)

    assert config_utils._get_config_llm_keyvalue(model_path, ["hidden_size"]) == 4096


def test_get_keyvalue_prefers_thinker_text_config_value_when_present(fake_config):
    cfg = {
        "model_type": "qwen3_omni_moe",
        "hidden_size": 4096,
        "thinker_config": {"text_config": {"hidden_size": 2048}},
    }
    model_path = fake_config(cfg)

    assert config_utils._get_config_llm_keyvalue(model_path, ["hidden_size"]) == 2048


def test_get_keyvalue_returns_none_when_key_missing_everywhere(fake_config):
    model_path = fake_config({"model_type": "llama", "hidden_size": 4096})

    assert config_utils._get_config_llm_keyvalue(model_path, ["head_dim"]) is None
