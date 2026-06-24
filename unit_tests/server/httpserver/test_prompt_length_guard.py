from types import SimpleNamespace

import pytest

from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster


class TokenizerMustNotRun:
    vocab_size = 32000

    def encode(self, *args, **kwargs):
        raise AssertionError("tokenizer should not run for oversized text prompts")


def _fake_manager():
    return SimpleNamespace(
        max_req_total_len=4,
        tokenizer=TokenizerMustNotRun(),
        args=SimpleNamespace(max_image_token_count=1024),
    )


def _empty_multimodal_params():
    return SimpleNamespace(images=[], audios=[])


@pytest.mark.parametrize("manager_cls", [HttpServerManager, HttpServerManagerForPDMaster])
def test_tokens_rejects_oversized_text_prompt_before_tokenization(manager_cls):
    prompt = "x" * 33

    with pytest.raises(ValueError, match="prompt text length 33 exceeds the character limit 32"):
        manager_cls.tokens(_fake_manager(), prompt, _empty_multimodal_params(), SimpleNamespace())
