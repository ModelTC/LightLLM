import pytest


@pytest.fixture
def compiler():
    from unittest.mock import patch

    import xgrammar as xgr
    from lightllm.server.httpserver.grammar import OutputGrammarCompiler
    from lightllm.server.router.model_infer.structured_output.grammar_cache import OutputGrammarCache

    # A complete byte vocabulary makes these tests independent of model files.
    vocab = [bytes([i]) for i in range(256)] + [b"<eos>"]
    tokenizer_info = xgr.TokenizerInfo(vocab, stop_token_ids=[256])
    with patch.object(xgr.TokenizerInfo, "from_huggingface", return_value=tokenizer_info):
        compiler = OutputGrammarCompiler(object(), 257, [256])
        compiler.grammar_cache = OutputGrammarCache(object(), 257, [256])
    yield compiler
    compiler.shutdown()
