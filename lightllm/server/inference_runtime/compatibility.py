from __future__ import annotations

import torch.multiprocessing as mp


def configure_vla_runtime(args, *, model_type: str) -> bool:
    """Normalize and validate the first-phase Pi action compatibility matrix.

    This helper is intentionally separate from backend selection.  A Pi
    checkpoint still selects the same normal/DP/PD/constraint backend as any
    other model; unsupported first-phase combinations fail here instead of
    being hidden behind an exclusive VLA backend.
    """

    enabled = model_type in {"pi0", "pi05"}
    args.enable_vla = enabled
    if not enabled:
        return False

    if args.run_mode != "normal":
        raise ValueError("Pi action inference does not yet support PD-separated run modes")

    unsupported = []
    if args.dp != 1:
        unsupported.append("DP")
    if args.nnodes != 1:
        unsupported.append("multi-node TP")
    if getattr(args, "enable_cpu_cache", False) or getattr(args, "enable_disk_cache", False):
        unsupported.append("multi-level KV cache")
    if getattr(args, "enable_dp_prompt_cache_fetch", False):
        unsupported.append("DP prompt-cache fetch")
    if getattr(args, "enable_tpsp_mix_mode", False):
        unsupported.append("TPSP mix mode")
    if getattr(args, "enable_prefill_microbatch_overlap", False):
        unsupported.append("prefill microbatch overlap")
    if getattr(args, "enable_decode_microbatch_overlap", False):
        unsupported.append("decode microbatch overlap")
    if getattr(args, "enable_prefill_decode_mixed", False):
        unsupported.append("mixed prefill/decode batching")
    if getattr(args, "output_constraint_mode", "none") != "none":
        unsupported.append("constraint decoding")
    if getattr(args, "token_healing_mode", False):
        unsupported.append("token healing")
    if getattr(args, "diverse_mode", False):
        unsupported.append("diverse generation")
    if getattr(args, "use_reward_model", False):
        unsupported.append("reward-model mode")
    if getattr(args, "return_all_prompt_logprobs", False):
        unsupported.append("all-prompt-logprob mode")
    if unsupported:
        raise ValueError("Pi action inference has not been validated with: " + ", ".join(unsupported))

    # Action ranks attach matching target KV buffers through CUDA IPC.
    mp.set_start_method("spawn", force=True)
    args.action_tp = args.tp
    args.action_gpu_ids = list(range(args.tp))

    from lightllm.models.pi0.config import Pi0VLAConfig

    config = Pi0VLAConfig.from_start_args(args)
    if args.max_req_total_len is None:
        args.max_req_total_len = args.vla_max_prefix_tokens + config.action_horizon + 1

    # Pi prefixes are bidirectional, so causal radix reuse and partial-prefix
    # chunking are not semantically valid.  This does not alter those features
    # for ordinary causal checkpoints.
    args.disable_dynamic_prompt_cache = True
    args.disable_chunked_prefill = True
    args.disable_audio = True
    args.disable_vision = False
    return True
