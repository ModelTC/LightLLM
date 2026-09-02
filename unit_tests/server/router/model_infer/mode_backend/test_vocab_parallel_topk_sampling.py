from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.server.router.model_infer.mode_backend.generic_post_process import (
    _can_use_unmodified_greedy_logits,
    can_use_vocab_parallel_topk,
    sample,
)
from lightllm.utils.envs_utils import enable_env_vars


def make_req(**overrides):
    values = {
        "top_k": 1,
        "temperature": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_new_tokens": 1,
        "decay_factor": 1.0,
        "invalid_token_ids": [],
        "output_len": 0,
    }
    values.update(overrides)
    shm_param = SimpleNamespace(
        top_k=values["top_k"],
        temperature=values["temperature"],
        presence_penalty=values["presence_penalty"],
        frequency_penalty=values["frequency_penalty"],
        repetition_penalty=values["repetition_penalty"],
        min_new_tokens=values["min_new_tokens"],
        exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (1, values["decay_factor"])),
    )
    input_len = 10
    return SimpleNamespace(
        sampling_param=SimpleNamespace(
            shm_param=shm_param,
            invalid_token_ids=values["invalid_token_ids"],
        ),
        shm_req=SimpleNamespace(input_len=input_len),
        get_cur_total_len=lambda: input_len + values["output_len"],
    )


def test_accepts_unmodified_greedy_requests():
    assert _can_use_unmodified_greedy_logits([make_req(), make_req(output_len=5)])


def test_feature_gate_requires_environment_and_eligible_batch(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_VOCAB_PARALLEL_TOPK", raising=False)
    enable_env_vars.cache_clear()
    assert not can_use_vocab_parallel_topk([make_req()])

    monkeypatch.setenv("LIGHTLLM_VOCAB_PARALLEL_TOPK", "1")
    enable_env_vars.cache_clear()
    assert can_use_vocab_parallel_topk([make_req()])
    assert not can_use_vocab_parallel_topk([make_req(top_k=2)])
    enable_env_vars.cache_clear()


def test_samples_sparse_candidates_and_maps_global_token_ids():
    model_output = ModelOutput(
        logits=torch.tensor([[9.0, 7.0, 1.0], [4.0, 6.0, 5.0]], dtype=torch.float32),
        logits_token_ids=torch.tensor([[17, 3, 8], [3, 20, 11]], dtype=torch.int64),
    )

    token_ids, token_logprobs = sample(model_output, [make_req(), make_req()])

    expected_probs = torch.softmax(model_output.logits, dim=-1).max(dim=-1).values
    torch.testing.assert_close(token_ids, torch.tensor([17, 20]))
    torch.testing.assert_close(token_logprobs, torch.log(expected_probs))


@pytest.mark.parametrize(
    "override",
    [
        {"top_k": 2},
        {"temperature": 0.5},
        {"presence_penalty": 0.1},
        {"frequency_penalty": 0.1},
        {"repetition_penalty": 1.1},
        {"decay_factor": 1.1},
        {"invalid_token_ids": [7]},
        {"min_new_tokens": 2},
    ],
)
def test_rejects_logits_modifiers(override):
    assert not _can_use_unmodified_greedy_logits([make_req(**override)])
