"""Autonomous VLA protocol, server, backend, and TP workers."""

from .io import load_action_config


def configure_defaults(args) -> None:
    """Apply defaults required by a Prefix-KV action branch."""

    config = load_action_config(args)
    # This flag selects VLAServer. Checkpoint registration selects the concrete
    # action model implementation once when TP workers start.
    args.generation_backend = "action"
    args.generation_runtime_config = config
    if args.max_req_total_len is None:
        args.max_req_total_len = (
            args.vla_max_prefix_tokens + config["action_horizon"] + 1
        )

    # The sidecar lease must reference one complete, committed target prefix.
    args.disable_dynamic_prompt_cache = True
    args.disable_chunked_prefill = True
    args.disable_audio = True
    args.disable_vision = False
    args.schedule_time_interval = min(args.schedule_time_interval, 0.001)

    if args.data_type in {"fp32", "float32"}:
        args.llm_prefill_att_backend = ["triton"]
        args.llm_decode_att_backend = ["triton"]


def create_client(args):
    from .client import ActionGenerationClient

    return ActionGenerationClient(args)


def get_start_process():
    from .server import start_vla_process

    return start_vla_process
