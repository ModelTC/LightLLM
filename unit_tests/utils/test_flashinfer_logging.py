from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lightllm.distributed import flashinfer_all_reduce as module


def test_nonleader_workspace_failure_remains_visible(monkeypatch):
    workspace = module.FlashInferAllReduce.__new__(module.FlashInferAllReduce)
    workspace._workspace = None
    workspace.max_workspace_size = 4096
    workspace.rank = 1
    workspace.world_size = 2
    workspace.group = None
    logger = Mock()
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module, "TorchDistBackend", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "flashinfer_comm",
        SimpleNamespace(create_allreduce_fusion_workspace=Mock(side_effect=RuntimeError("local allocation failed"))),
    )
    assert workspace._ensure_workspace(128, torch.float16) is False
    assert workspace.disabled is True
    logger.warning.assert_called_once()
    assert "local allocation failed" in logger.warning.call_args.args[1]
    logger.debug.assert_called_once_with("FlashInfer workspace initialization diagnostics", exc_info=True)
