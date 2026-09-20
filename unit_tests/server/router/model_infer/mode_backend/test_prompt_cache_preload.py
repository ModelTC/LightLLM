from types import SimpleNamespace

import torch

from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


def test_preload_prompt_cache_keeps_only_complete_kv_pages(monkeypatch):
    loaded = {}

    class FakeMemManager:
        def alloc(self, size):
            loaded["alloc_size"] = size
            return torch.arange(size, dtype=torch.int32)

        def load_index_kv_buffer(self, indexes, buffers):
            loaded["indexes"] = indexes.clone()
            loaded["buffers"] = buffers

    class FakeRadixCache:
        def __init__(self):
            self.mem_manager = FakeMemManager()

        def insert(self, token_ids, indexes):
            loaded["insert"] = (token_ids.clone(), indexes.clone())

        def match_prefix(self, token_ids, update_refs):
            loaded["match"] = (token_ids.clone(), update_refs)

    backend = ModeBackend.__new__(ModeBackend)
    backend.logger = SimpleNamespace(info=lambda _: None)
    backend.weight_dir = "/model"
    backend.args = SimpleNamespace(page_size=4)
    backend.radix_cache = FakeRadixCache()
    kv_buffer = torch.arange(12).reshape(1, 6, 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"kv_buffer": kv_buffer})

    backend.preload_prompt_cache_kv_buffer(
        {
            "prompt_cache_kv_buffer": {"rank_0": "prompt_cache.pt"},
            "prompt_cache_token_ids": [10, 11, 12, 13, 14, 15],
        }
    )

    assert loaded["alloc_size"] == 4
    assert loaded["indexes"].tolist() == [0, 1, 2, 3]
    assert loaded["buffers"]["kv_buffer"].shape == (1, 4, 2)
    assert loaded["insert"][0].tolist() == [10, 11, 12, 13]
    assert loaded["insert"][1].tolist() == [0, 1, 2, 3]
    assert loaded["match"][0].tolist() == [10, 11, 12, 13]
    assert loaded["match"][1] is True
