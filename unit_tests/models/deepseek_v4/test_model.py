from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel


def test_memory_check_rejects_sequence_larger_than_full_token_pool():
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.mem_manager = SimpleNamespace(size=60416)
    model.batch_max_tokens = 8192
    model.max_seq_length = 262169
    model.args = SimpleNamespace(performance_mode=None)

    with pytest.raises(AssertionError, match="max_total_token_num must be >= max_seq_length"):
        model._check_mem_size()


def test_memory_check_still_requires_one_prefill_batch():
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.mem_manager = SimpleNamespace(size=8192)
    model.batch_max_tokens = 8192
    model.max_seq_length = 262169
    model.args = SimpleNamespace(performance_mode=None)

    with pytest.raises(AssertionError, match="greater than batch_max_tokens"):
        model._check_mem_size()


def test_decode_autotune_uses_current_mtp_capacity_api(monkeypatch):
    lengths = []
    calls = []
    monkeypatch.setattr("lightllm.common.basemodel.basemodel.Autotuner.start_autotune_warmup", lambda *_: None)
    monkeypatch.setattr("lightllm.common.basemodel.basemodel.Autotuner.end_autotune_warmup", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        "lightllm.common.basemodel.basemodel.tqdm",
        lambda values, **kwargs: lengths.extend(values) or [],
    )

    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.args = SimpleNamespace(run_mode="decode")
    model.batch_max_tokens = 8192
    model.max_req_num = 4
    model.is_mtp_draft_model = False
    model.mtp_manager = SimpleNamespace(
        get_decode_tokens_per_request=lambda is_draft: calls.append(is_draft) or 4,
    )
    model.layers_num = 1
    model.autotune_layers = lambda: 1

    model._autotune_warmup()

    assert calls == [False]
    assert lengths == [16, 8, 4, 1]
