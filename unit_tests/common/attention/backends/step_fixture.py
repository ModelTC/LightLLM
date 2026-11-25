from types import SimpleNamespace

from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo


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
