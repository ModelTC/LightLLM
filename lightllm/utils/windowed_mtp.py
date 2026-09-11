"""Opt-in bounded draft KV policy, independent of the proposal algorithm."""
import json
from pathlib import Path

from lightllm.common.build_utils import repair_config


def window_capacity(args):
    if args.mtp_draft_window <= 0 or args.mtp_draft_sinks < 0:
        raise ValueError("mtp_draft_window must be positive and mtp_draft_sinks nonnegative")
    return args.mtp_draft_window + args.mtp_draft_sinks


def window_kv_pool_bytes(args, element_size):
    config = json.loads((Path(args.mtp_draft_model_dir[0]) / "config.json").read_text())
    # Match the existing draft initialization order: base aliases, Llama KV
    # defaults, then the overrides applied by Qwen3.5 adapters.
    # Do not normalize again after merging: runtime also retains those aliases.
    merge_nested = config.get("model_type") in ("qwen3_5", "qwen3_5_text")
    repair_config(config, same_names=["num_attention_heads", "n_head"])
    repair_config(config, same_names=["hidden_size", "n_embd", "n_embed"])
    repair_config(config, same_names=["num_hidden_layers", "n_layer"])
    config.setdefault("num_key_value_heads", config["num_attention_heads"])
    if merge_nested:
        config.update(config.get("dflash_config", {}))
    layers = config["n_layer"]
    heads = config["num_key_value_heads"] // args.tp
    dim = config.get("head_dim", config["n_embed"] // config["num_attention_heads"])
    return (args.running_max_req_size + 1) * (layers * window_capacity(args) * 2 * heads * dim * element_size + 12)


def validate_windowed_mtp(args):
    window_capacity(args)
    if args.mtp_mode not in ("dflash", "dspark"):
        raise ValueError("windowed draft KV requires dflash or dspark")
    if args.run_mode != "normal" or args.dp != 1:
        raise ValueError("windowed draft KV currently requires normal deployment and dp=1")
    if not args.disable_dynamic_prompt_cache:
        raise ValueError("windowed draft KV requires --disable_dynamic_prompt_cache")
    if args.llm_kv_type not in (None, "None"):
        raise ValueError("windowed draft KV currently requires unquantized KV")
    for option in (
        "enable_cpu_cache",
        "enable_dp_prompt_cache_fetch",
        "diverse_mode",
        "enable_tpsp_mix_mode",
        "enable_prefill_decode_mixed",
        "enable_decode_microbatch_overlap",
        "enable_prefill_microbatch_overlap",
        "mtp_dynamic_verify",
    ):
        if getattr(args, option, False):
            raise ValueError(f"windowed draft KV does not yet support --{option}")
