from types import SimpleNamespace

import pytest

from lightllm.utils import config_utils


@pytest.fixture(autouse=True)
def clear_fixed_kv_len_cache():
    config_utils.get_fixed_kv_len.cache_clear()
    yield
    config_utils.get_fixed_kv_len.cache_clear()


def test_fixed_kv_len_supports_page_size_one(monkeypatch):
    monkeypatch.setattr(config_utils, "get_env_start_args", lambda: SimpleNamespace(page_size=1, model_dir="model"))
    monkeypatch.setattr(config_utils, "get_config_json", lambda _: {"prompt_cache_token_ids": [1, 2, 3]})

    assert config_utils.get_fixed_kv_len() == 3


def test_fixed_kv_len_uses_only_complete_pages(monkeypatch):
    monkeypatch.setattr(config_utils, "get_env_start_args", lambda: SimpleNamespace(page_size=2, model_dir="model"))
    monkeypatch.setattr(config_utils, "get_config_json", lambda _: {"prompt_cache_token_ids": [1, 2, 3]})

    assert config_utils.get_fixed_kv_len() == 2


def test_no_fixed_kv_supports_paged_kv(monkeypatch):
    monkeypatch.setattr(config_utils, "get_env_start_args", lambda: SimpleNamespace(page_size=4, model_dir="model"))
    monkeypatch.setattr(config_utils, "get_config_json", lambda _: {})

    assert config_utils.get_fixed_kv_len() == 0
