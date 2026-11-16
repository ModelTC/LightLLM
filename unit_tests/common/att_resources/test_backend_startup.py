from __future__ import annotations


import pytest


import torch


from lightllm.server.core.objs.start_args_type import StartArgs


from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_backend_rejects_more_than_36_drafts_before_loading():
    backend = ModeBackend.__new__(ModeBackend)
    backend.args = StartArgs(mtp_mode="vanilla_with_att", mtp_step=37)
    with pytest.raises(AssertionError, match="at most 36"):
        backend.init_mtp_draft_model({})
    assert backend.draft_models == []
