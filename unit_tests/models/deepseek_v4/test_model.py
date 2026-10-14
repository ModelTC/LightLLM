from types import SimpleNamespace

import pytest

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
