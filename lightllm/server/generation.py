from .httpserver.stop_seq_match import match_stop_sequence_stream


def generate_with_stop(manager, prompt, sampling_params, multimodal_params, request):
    """应用侧统一包装生成流；底层 manager 只负责生成和停止指定请求。"""
    # RL 和最终 metadata 路径保留后端原始 token，不做应用侧截断或提前结束。
    needs_final_metadata = (
        getattr(manager.args, "enable_rl", False)
        or getattr(manager.args, "enable_return_routed_experts", False)
        or getattr(sampling_params, "prompt_logprobs", -1) >= 0
    )
    if needs_final_metadata:
        assert sampling_params.stop_match_mode.is_backend_all(), (
            "RL and final token metadata require stop_match_mode=BACKEND_ALL"
        )
    generator = manager.generate(prompt, sampling_params, multimodal_params, request=request)
    if not sampling_params.stop_sequences.size or needs_final_metadata:
        return generator
    return match_stop_sequence_stream(
        generator,
        sampling_params,
        manager.tokenizer,
        prompt,
        manager.args.reasoning_parser,
        on_stop=manager.stop_request,
    )
