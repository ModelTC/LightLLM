def should_init_decode_wrapper(model, infer_state) -> bool:
    graph = getattr(model, "graph", None)
    if graph is None:
        # Cuda graph is disabled, so this state owns a normal decode wrapper.
        return True

    if infer_state.is_cuda_graph:
        # This is the captured graph state; it must create the wrapper captured by replay.
        return True

    if not graph.can_run(infer_state.batch_size, infer_state.max_kv_seq_len):
        # Cuda graph is enabled, but this input falls outside graph limits and runs normally.
        return True

    # This is a temporary replay state. Its tensors are copied into the captured graph state.
    return False
