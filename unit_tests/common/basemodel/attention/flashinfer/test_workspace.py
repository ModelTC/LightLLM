from types import SimpleNamespace

import torch

from lightllm.common.basemodel.attention import get_attention_workspace
from lightllm.platform import get_device_runtime


def test_workspace_isolated_by_model_group_and_slot():
    model = SimpleNamespace(_attention_workspaces={}, device_runtime=get_device_runtime("cpu"))
    other = SimpleNamespace(_attention_workspaces={}, device_runtime=get_device_runtime("cpu"))
    primary = get_attention_workspace(model, 0, 0, "cpu")
    assert primary is get_attention_workspace(model, 0, 0, "cpu")
    assert primary is not get_attention_workspace(model, 1, 0, "cpu")
    assert primary is not get_attention_workspace(model, 0, 1, "cpu")
    assert primary is not get_attention_workspace(other, 0, 0, "cpu")


def test_workspace_growth_keeps_captured_references_alive():
    model = SimpleNamespace(_attention_workspaces={}, device_runtime=get_device_runtime("cpu"))
    workspace = get_attention_workspace(model, 0, 0, "cpu")
    captured = workspace.get("indices", 8, torch.int32)
    captured.fill_(7)
    enlarged = workspace.get("indices", 32, torch.int32)
    assert enlarged.data_ptr() != captured.data_ptr()
    assert captured.tolist() == [7] * 8
    assert workspace.get("indices", 4, torch.int32).data_ptr() == enlarged.data_ptr()
