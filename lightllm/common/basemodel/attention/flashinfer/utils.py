def should_init_decode_wrapper(infer_state) -> bool:
    match (infer_state.is_cuda_graph_capture, infer_state.is_cuda_graph_replay):
        case (False, False):
            # eager 推理（包括 graph 不可用时的回退）会直接执行 attention，需要创建 wrapper。
            return True
        case (True, False):
            # capture 需要创建 wrapper，并随捕获状态保存在图缓存中，供后续 replay 使用。
            return True
        case (False, True):
            # replay 的临时状态只准备待复制的 metadata，实际执行使用图缓存中的 wrapper。
            return False
        case _:
            # 两个标记必须是互斥的布尔值，不能同时处于 capture 和 replay 状态。
            raise ValueError(
                f"Invalid CUDA Graph flags: capture={infer_state.is_cuda_graph_capture}, "
                f"replay={infer_state.is_cuda_graph_replay}"
            )
