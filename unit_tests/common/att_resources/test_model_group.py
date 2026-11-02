from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.basemodel.model_group import initialize_model_group


@pytest.mark.parametrize("fail_runtime", [False, True])
def test_shared_initialization(monkeypatch, fail_runtime):
    prepared, warmed, ready = [], [], []
    req_manager = object()
    create_requests = Mock(return_value=req_manager)
    monkeypatch.setattr("lightllm.common.req_manager.ReqManager", create_requests)
    monkeypatch.setattr("lightllm.utils.envs_utils.set_model_init_status", ready.append)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    manager = Mock()
    models = []

    def allocate(args, token_capacity, mem_fraction):
        assert prepared == models
        assert all(model.req_manager is req_manager for model in models)
        manager.finish_state_registration.assert_called_once_with()
        manager.load_full_att_calibration.assert_called_once_with(args=args)
        assert token_capacity == 32 and mem_fraction == 0.9
        manager.allocated = True

    manager.initialize_storage.side_effect = allocate
    manager.allocated = False

    def prepare(model):
        assert all(item.req_manager is req_manager for item in models)
        model.att_state_manager = manager
        prepared.append(model)

    def warmup(model):
        assert manager.allocated and not ready
        if fail_runtime and model is models[-1]:
            raise RuntimeError("warmup failed")
        warmed.append(model)

    for seq_len in (16, 48, 24):
        model = SimpleNamespace(
            max_req_num=2,
            max_total_token_num=32,
            mem_fraction=0.9,
            args=object(),
            torch_memory_saver=SimpleNamespace(region=lambda **kwargs: nullcontext()),
            get_req_manager_max_seq_len=lambda seq_len=seq_len: seq_len,
        )
        model.bind_req_manager = lambda req_manager, model=model: setattr(model, "req_manager", req_manager)
        model.prepare_attention_states = lambda model=model: prepare(model)
        model.init_runtime = lambda model=model: warmup(model)
        models.append(model)

    if fail_runtime:
        with pytest.raises(RuntimeError, match="warmup failed"):
            initialize_model_group(models[0], models[1:])
        assert not ready and warmed == models[:-1]
    else:
        initialize_model_group(models[0], models[1:])
        assert warmed == models and ready == [True]
    create_requests.assert_called_once_with(max_request_num=2, max_sequence_length=48)
    with pytest.raises(AssertionError, match="resources already initialized"):
        initialize_model_group(models[0], models[1:])
    assert create_requests.call_count == 1
