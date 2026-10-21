import json

import pytest


@pytest.fixture
def single_http_worker_start_args(monkeypatch):
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args

    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(vars(StartArgs(httpserver_workers=1))),
    )
    get_env_start_args.cache_clear()
    yield
    get_env_start_args.cache_clear()


@pytest.fixture
def compiler():
    from unittest.mock import patch

    import xgrammar as xgr
    from lightllm.server.httpserver.grammar import OutputGrammarCompiler
    from lightllm.server.router.model_infer.structured_output.state_factory import OutputConstraintStateFactory

    # A complete byte vocabulary makes these tests independent of model files.
    vocab = [bytes([i]) for i in range(256)] + [b"<eos>"]
    tokenizer_info = xgr.TokenizerInfo(vocab, stop_token_ids=[256])
    with patch.object(xgr.TokenizerInfo, "from_huggingface", return_value=tokenizer_info):
        compiler = OutputGrammarCompiler(object(), 257, [256])
        compiler.output_constraint_state_factory = OutputConstraintStateFactory(object(), 257, [256])
    yield compiler
    compiler.shutdown()
