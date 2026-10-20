import pytest


@pytest.fixture
def cpu_cache_factory(monkeypatch):
    from types import SimpleNamespace
    import torch
    from lightllm.common.cpu_cache import CpuCacheCreator
    from lightllm.server.multi_level_kv_cache import cpu_cache_client, shm_objs

    monkeypatch.setattr(
        shm_objs, "_create_shm", lambda name, byte_size: SimpleNamespace(size=byte_size, buf=bytearray(byte_size))
    )
    monkeypatch.setattr(
        cpu_cache_client,
        "AtomicShmLock",
        lambda **kwargs: SimpleNamespace(acquire_sleep1ms=lambda: None, release=lambda: None),
    )
    monkeypatch.setattr(cpu_cache_client, "get_disk_cache_prompt_limit_length", lambda: 0)
    monkeypatch.setattr(
        CpuCacheCreator,
        "create_or_attach",
        lambda self, **kwargs: torch.zeros(
            self.tensor_spec.shape, dtype=self.tensor_spec.dtype, pin_memory=torch.cuda.is_available()
        ),
    )

    def create(meta):
        monkeypatch.setattr(
            cpu_cache_client,
            "get_env_start_args",
            lambda: SimpleNamespace(cpu_kv_cache_shm_id=0),
        )
        monkeypatch.setattr(cpu_cache_client, "calcu_cpu_cache_meta", lambda: meta)
        return cpu_cache_client.CpuKvCacheClient(only_create_meta_data=False, init_shm_data=True)

    return create


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
