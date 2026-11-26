from lightllm.common.basemodel.infer_struct import GraphMode


def should_init_decode_wrapper(infer_state) -> bool:
    """Replay reuses the wrapper held by the captured graph."""
    return infer_state.graph_mode is not GraphMode.REPLAY
