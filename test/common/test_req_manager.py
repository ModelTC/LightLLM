import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.linear_att_cache_manager.config_objs import get_linear_att_state_mtp_size


def test_linear_att_state_buffer_log_reports_shape_and_memory():
    source = Path("lightllm/common/req_manager.py").read_text()
    module = ast.parse(source)

    class_node = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ReqManagerForMamba"
    )
    init_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    init_source = ast.unparse(init_node)

    assert "logger.info" in init_source
    assert "conv_state shape=" in init_source
    assert "ssm_state shape=" in init_source
    assert "total memory=" in init_source
    assert "_format_nbytes(conv_nbytes)" in init_source
    assert "_format_nbytes(ssm_nbytes)" in init_source
    assert "linear_att_state_mtp_size" in init_source


@pytest.mark.parametrize(
    ("run_mode", "mtp_step", "expected"),
    [
        ("prefill", 3, 1),
        ("decode", 3, 4),
        ("normal", 3, 4),
        ("prefill", 0, 1),
        ("normal", 0, 1),
    ],
)
def test_linear_att_state_mtp_size(run_mode, mtp_step, expected):
    args = SimpleNamespace(run_mode=run_mode, mtp_step=mtp_step)
    assert get_linear_att_state_mtp_size(args) == expected


def test_qwen_prefill_uses_compact_linear_att_state_indexes(monkeypatch):
    from lightllm.models.qwen3next import infer_struct

    monkeypatch.setattr(infer_struct, "get_env_start_args", lambda: SimpleNamespace(mtp_step=3))
    state = SimpleNamespace(
        b_seq_len=torch.tensor([8, 16], dtype=torch.int32),
        b_req_idx=torch.tensor([1, 3], dtype=torch.int32),
        b_mtp_index=torch.zeros(2, dtype=torch.int32),
        is_prefill=True,
    )
    model = SimpleNamespace(req_manager=SimpleNamespace(linear_att_state_mtp_size=1))

    infer_struct.init_mtp_verify_extra_state(state, model)

    assert torch.equal(state.b_buffer_idx, torch.tensor([1, 3], dtype=torch.int32))
    assert not state.is_mtp_verify


def test_qwen_decode_keeps_speculative_linear_att_state_indexes(monkeypatch):
    from lightllm.models.qwen3next import infer_struct

    monkeypatch.setattr(infer_struct, "get_env_start_args", lambda: SimpleNamespace(mtp_step=3))
    state = SimpleNamespace(
        b_seq_len=torch.ones(8, dtype=torch.int32),
        b_req_idx=torch.tensor([1, 1, 1, 1, 3, 3, 3, 3], dtype=torch.int32),
        b_mtp_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32),
        is_prefill=False,
    )
    model = SimpleNamespace(
        req_manager=SimpleNamespace(
            linear_att_state_mtp_size=4,
            req_to_accept_len=torch.ones(4, dtype=torch.int32),
        )
    )

    infer_struct.init_mtp_verify_extra_state(state, model)

    assert torch.equal(state.b_buffer_idx, torch.tensor([4, 5, 6, 7, 12, 13, 14, 15], dtype=torch.int32))
    assert torch.equal(
        state.b_ssm_index_rows,
        torch.tensor([[4, 5, 6, 7], [12, 13, 14, 15]], dtype=torch.int32),
    )
    assert torch.equal(state.b_conv_buffer_idx, torch.tensor([1, 3], dtype=torch.int32))


def test_qwen_decode_rejects_compact_linear_att_state_layout(monkeypatch):
    from lightllm.models.qwen3next import infer_struct

    monkeypatch.setattr(infer_struct, "get_env_start_args", lambda: SimpleNamespace(mtp_step=3))
    state = SimpleNamespace(
        b_seq_len=torch.ones(4, dtype=torch.int32),
        b_req_idx=torch.ones(4, dtype=torch.int32),
        b_mtp_index=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        is_prefill=False,
    )
    model = SimpleNamespace(req_manager=SimpleNamespace(linear_att_state_mtp_size=1))

    with pytest.raises(AssertionError, match="MTP decode requires"):
        infer_struct.init_mtp_verify_extra_state(state, model)


@pytest.mark.parametrize(
    ("state_mtp_size", "req_idx", "expected"),
    [
        (1, 3, (3, 3)),
        (4, 3, (3, 12)),
    ],
)
def test_pd_linear_att_state_page_indexes_follow_physical_layout(state_mtp_size, req_idx, expected):
    from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper

    helper = object.__new__(Qwen3NextLinearAttPageHelper)
    helper.mem_manager = SimpleNamespace(linear_att_state_mtp_size=state_mtp_size)

    assert helper._get_req_state_indexes(req_idx) == expected
