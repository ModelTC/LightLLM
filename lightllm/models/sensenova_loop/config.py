"""SenseNova Loop execution layout."""


def get_loop_layer_indices(settings: dict, num_layers: int) -> tuple[int, ...]:
    """Map execution depth to checkpoint layers; caches use execution depth."""
    start, end, count = settings["start"], settings["end"], settings["eval_count"]
    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))


def get_loop_layer_types(settings: dict, text_config: dict) -> tuple[str, ...]:
    num_layers = text_config["num_hidden_layers"]
    indices = get_loop_layer_indices(settings, num_layers)
    layer_types = text_config["layer_types"]
    return tuple(layer_types[i] for i in indices)


def validate_inline_vision(args, model_type):
    """Keep the first inline implementation within a serialized, single-GPU service."""
    if not args.visual_infer_in_llm:
        return
    if model_type != "sensenova_loop":
        raise ValueError("--visual_infer_in_llm currently supports SenseNova Loop only")
    if args.disable_vision or not args.disable_audio or args.visual_use_proxy_mode:
        raise ValueError("Inline vision requires local vision and disabled audio")
    if (args.tp, args.dp, args.nnodes, args.visual_tp, args.visual_dp, args.running_max_req_size) != (1, 1, 1, 1, 1, 1):
        raise ValueError("Inline vision requires one GPU, one vision worker and --running_max_req_size 1")
    if getattr(args, "visual_gpu_ids", None) not in (None, [0]):
        raise ValueError("Inline vision uses the LLM worker's GPU; --visual_gpu_ids must be 0")
    if args.run_mode != "normal" or args.mtp_mode is not None or args.enable_cpu_cache:
        raise ValueError("Inline vision currently requires normal mode without MTP or CPU KV offload")
    if args.enable_prefill_cudagraph or args.enable_prefill_microbatch_overlap or args.enable_decode_microbatch_overlap:
        raise ValueError(
            "Inline vision requires eager prefill without microbatch overlap; decode CUDA Graph is supported"
        )
