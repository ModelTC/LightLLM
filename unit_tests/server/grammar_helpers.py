"""Shared byte-vocabulary fixtures and CPU references for grammar tests."""

import asyncio
import json
from itertools import count
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lightllm.server.core.objs import FinishStatus, StartArgs
from lightllm.server.router.model_infer.mode_backend.generic_post_process import (
    _get_post_sample_tensors,
    g_pin_mem_manager,
)
from lightllm.utils.envs_utils import get_env_start_args


@pytest.fixture(autouse=True)
def reset_args_cache():
    get_env_start_args.cache_clear()
    yield
    get_env_start_args.cache_clear()


def configure(monkeypatch, **kwargs):
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(vars(StartArgs(**kwargs))))
    get_env_start_args.cache_clear()


req_indices = count()


class SamplingRequest(SimpleNamespace):
    def get_cur_total_len(self):
        return self.shm_req.input_len + self.cur_output_len


def make_req(**constraints):
    params = dict(
        guided_json=None,
        guided_grammar=None,
        regular_constraint=None,
        guided_reasoning_end=(),
        invalid_token_ids=[],
        shm_param=SimpleNamespace(
            pd_previous_output_len=0,
            exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (0, 1.0)),
            min_new_tokens=0,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
        ),
    )
    params.update(constraints)
    return SamplingRequest(
        req_idx=next(req_indices),
        vocab_size=257,
        generator=None,
        sampling_param=SimpleNamespace(**params),
        output_constraint=None,
        finish_status=FinishStatus(),
        cur_output_len=0,
        shm_req=SimpleNamespace(
            input_len=1, shm_prompt_ids=SimpleNamespace(arr=[ord("P")]), get_compiled_grammar=lambda: b""
        ),
    )


def init_request(compiler, req):
    for kind, value in (
        ("grammar", req.sampling_param.guided_grammar),
        ("regex", req.sampling_param.regular_constraint),
        ("json", req.sampling_param.guided_json),
    ):
        if value:
            payload = asyncio.run(compiler.compile(kind, value))
            req.shm_req.get_compiled_grammar = lambda: payload
            req.output_constraint = compiler.grammar_cache.create_state(req.shm_req, req.sampling_param)
            return


def make_mask_buffers(reqs, verify_width=1):
    capacity = max((req.req_idx for req in reqs), default=0) + 1
    return SimpleNamespace(
        vocab_size=257,
        req_to_next_token_ids=torch.zeros((capacity, verify_width), dtype=torch.int64),
        req_to_bitmask=torch.empty((capacity, verify_width, 9), dtype=torch.int32),
        req_to_bitmask_enabled=torch.zeros(capacity, dtype=torch.bool),
    )


def prepare_sampling_tensors(reqs, manager, has_output=None):
    # Exercise production sampling preparation with CPU parameter tensors;
    # CUDA integration tests use the real pinned-memory manager and transfers.
    def cpu_tensor(key, data, dtype):
        tensor = torch.tensor(data, dtype=dtype)
        return SimpleNamespace(cuda=lambda non_blocking: tensor)

    with patch.object(g_pin_mem_manager, "gen_from_list", side_effect=cpu_tensor):
        return _get_post_sample_tensors(reqs, manager, has_output)


def build_mask(reqs, has_output=None, draft_input_ids=None):
    manager = make_mask_buffers(reqs, len(reqs))
    if draft_input_ids is not None:
        for index, req in enumerate(reqs):
            if index == 0 or reqs[index - 1] is not req:
                request_start = index
            manager.req_to_next_token_ids[req.req_idx, index - request_start] = draft_input_ids[index]
    return manager if prepare_sampling_tensors(reqs, manager, has_output)[-1] else None


def apply_masks(reqs, logits, manager, b_mtp_index=None):
    if manager is None:
        return
    if b_mtp_index is None:
        positions = []
        for index, req in enumerate(reqs):
            if index == 0 or reqs[index - 1] is not req:
                request_start = index
            positions.append(index - request_start)
        b_mtp_index = torch.tensor(positions, dtype=torch.int32, device=logits.device)
    # Independent CPU reference; CUDA integration tests use the production kernel.
    token_ids = torch.arange(manager.vocab_size)
    for index, req in enumerate(reqs):
        if manager.req_to_bitmask_enabled[req.req_idx]:
            bitmask = manager.req_to_bitmask[req.req_idx, b_mtp_index[index]]
            allowed = ((bitmask[token_ids // 32] >> (token_ids % 32)) & 1).bool()
            logits[index, : manager.vocab_size].masked_fill_(~allowed, float("-inf"))
    logits[:, manager.vocab_size :].fill_(float("-inf"))


def allowed(reqs, has_output=None):
    mask = build_mask(reqs, has_output)
    logits = torch.zeros(len(reqs), 257)
    apply_masks(reqs, logits, mask)
    return torch.isfinite(logits)


def commit_token(req, token):
    req.shm_req.shm_prompt_ids.arr.append(token)
    req.cur_output_len += 1
    req.output_constraint.commit(token)
    return req.output_constraint.is_terminated()
