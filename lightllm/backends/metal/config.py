"""Supported first-stage Metal serving configuration."""
import json
from pathlib import Path


def configure_server(args):
    config = json.loads((Path(args.model_dir) / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Metal currently supports dense Qwen3.5 text inference")
    if args.tp != 1 or args.dp != 1 or args.nnodes != 1 or args.run_mode != "normal":
        raise ValueError("Metal currently requires --tp 1 --dp 1 --nnodes 1 --run_mode normal")
    for name in (
        "enable_cpu_cache",
        "enable_disk_cache",
        "enable_dp_prompt_cache_fetch",
        "enable_ep_moe",
        "enable_tpsp_mix_mode",
        "enable_prefill_microbatch_overlap",
        "enable_decode_microbatch_overlap",
        "enable_prefill_cudagraph",
        "enable_torch_memory_saver",
        "enable_rl",
        "enable_mps",
        "enable_prompt_logprobs",
        "use_reward_model",
        "diverse_mode",
    ):
        if getattr(args, name):
            raise ValueError(f"Metal does not support --{name}")
    if args.quant_type != "none" or args.quant_cfg or args.expert_dtype or config.get("quantization_config"):
        raise ValueError("Metal currently requires original unquantized Qwen3.5 weights")
    if args.llm_kv_type != "None" or args.mtp_mode or args.mtp_step:
        raise ValueError("Metal currently requires unquantized KV and no MTP")
    if args.data_type not in ("fp16", "float16", "fp32", "float32"):
        raise ValueError("Metal currently supports --data_type fp16 or fp32")
    if args.disable_vision is False or args.disable_audio is False:
        raise ValueError("Metal currently supports text input only")
    if args.output_constraint_mode != "none":
        raise ValueError("Metal currently requires --output_constraint_mode none")
    if args.penalty_counter_mode not in ("gpu_counter", "cpu_counter"):
        raise ValueError("Metal supports gpu_counter and cpu_counter penalty modes")
    if any(name != "auto" for name in args.llm_prefill_att_backend + args.llm_decode_att_backend):
        raise ValueError("Metal requires automatic attention backend selection")
    if args.target_vocab_topk_sampling or args.draft_vocab_topk_sampling:
        raise ValueError("Metal does not support vocabulary candidate pruning")
    if args.enable_return_routed_experts or args.enable_profiling:
        raise ValueError("Metal currently does not support expert capture or profiling")
    if args.max_total_token_num is None:
        raise ValueError("Set --max_total_token_num explicitly to bound unified memory use on Metal")
    args.disable_vision = True
    args.disable_audio = True
    args.disable_cudagraph = True
    args.disable_symm_mem_allreduce = True
    args.disable_flashinfer_allreduce = True
    args.linear_att_ssm_data_type = "float32"
    # This existing sampling path is Torch sort/multinomial despite its legacy
    # name. Penalties and token counters resolve through the operator backend.
    args.sampling_backend = "triton"
