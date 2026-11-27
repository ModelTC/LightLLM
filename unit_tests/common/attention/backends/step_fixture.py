from types import SimpleNamespace

import torch

from lightllm.common.basemodel.attention_backend_manager import AttentionBackendManager
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo


def make_infer_state(
    *, req_to_token_indexs=None, pad_req_idx=None, cuda_graph_max_kv_seq_len=None, max_q_seq_len=1, **fields
) -> InferStateInfo:
    state = InferStateInfo()
    state.page_size = 1
    state.model_max_seq_len = 512
    state.max_cuda_graph_batch_size = 8
    state.max_decode_rows = 8
    state.attention_backend_manager = AttentionBackendManager()
    state.graph_mode = GraphMode.CAPTURE if cuda_graph_max_kv_seq_len is not None else GraphMode.EAGER
    state.cuda_graph_max_kv_seq_len = cuda_graph_max_kv_seq_len
    state.max_q_seq_len = max_q_seq_len
    state.req_manager = SimpleNamespace(req_to_token_indexs=req_to_token_indexs, HOLD_REQUEST_ID=pad_req_idx)
    return configure_infer_state(state, **fields)


def configure_infer_state(state: InferStateInfo, **fields) -> InferStateInfo:
    for name, value in fields.items():
        assert hasattr(state, name), name
        setattr(state, name, value)
    return state
