from types import SimpleNamespace

import torch

from lightllm.common.attention.backend import AttentionBuildResources
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo


def make_resources(**overrides) -> AttentionBuildResources:
    """Model-wide resources for setting up a backend outside a model, on the current CUDA device.

    get_workspace returns the same tensor for the same arguments, like the serving workspace store.
    """
    workspaces = {}

    def get_workspace(name, numel, dtype):
        key = (name, numel, dtype)
        if key not in workspaces:
            workspaces[key] = torch.empty(numel, dtype=dtype, device="cuda")
        return workspaces[key]

    fields = dict(
        device=torch.cuda.current_device(),
        page_size=1,
        max_seq_len=512,
        max_cuda_graph_batch_size=8,
        max_decode_rows=8,
        microbatch_count=2,
        workspace_bytes=32 * 1024 * 1024,
        get_workspace=get_workspace,
    )
    return AttentionBuildResources(**{**fields, **overrides})


def make_infer_state(
    *, req_to_token_indexs=None, pad_req_idx=None, cuda_graph_max_kv_seq_len=None, max_q_seq_len=1, **fields
) -> InferStateInfo:
    state = InferStateInfo()
    state.req_manager = SimpleNamespace(req_to_token_indexs=req_to_token_indexs, HOLD_REQUEST_ID=pad_req_idx)
    state.graph_mode = GraphMode.CAPTURE if cuda_graph_max_kv_seq_len is not None else GraphMode.EAGER
    state.cuda_graph_max_kv_seq_len = cuda_graph_max_kv_seq_len
    state.max_q_seq_len = max_q_seq_len
    for name, value in fields.items():
        assert hasattr(state, name), name
        setattr(state, name, value)
    return state
