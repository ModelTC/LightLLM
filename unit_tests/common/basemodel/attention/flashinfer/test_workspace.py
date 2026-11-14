from types import SimpleNamespace

import torch

from lightllm.common.basemodel.attention import ModelAttention
from lightllm.platform import get_device_runtime


def test_workspace_isolated_by_model_group_and_slot():
    model = ModelAttention.__new__(ModelAttention)
    model.workspaces, model.runtime = {}, get_device_runtime("cpu")
    other = ModelAttention.__new__(ModelAttention)
    other.workspaces, other.runtime = {}, get_device_runtime("cpu")
    primary = model.workspace(0, 0)
    assert primary is model.workspace(0, 0)
    assert primary is not model.workspace(1, 0)
    assert primary is not model.workspace(0, 1)
    assert primary is not other.workspace(0, 0)


def test_workspace_growth_keeps_captured_references_alive():
    model = ModelAttention.__new__(ModelAttention)
    model.workspaces, model.runtime = {}, get_device_runtime("cpu")
    workspace = model.workspace(0, 0)
    captured = workspace.scratch("indices", 8, torch.int32)
    captured.fill_(7)
    enlarged = workspace.scratch("indices", 32, torch.int32)
    assert enlarged.data_ptr() != captured.data_ptr()
    assert captured.tolist() == [7] * 8
    assert workspace.scratch("indices", 4, torch.int32).data_ptr() == enlarged.data_ptr()
