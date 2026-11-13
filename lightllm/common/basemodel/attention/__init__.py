from lightllm.common.attention.api import AttentionPhase, ExecutionRequirements

from .base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState
from .create_utils import (
    get_decode_att_backend_class,
    get_mla_decode_att_backend_class,
    get_mla_prefill_att_backend_class,
    get_neo_prefill_att_backend_class,
    get_nsa_decode_att_backend_class,
    get_nsa_prefill_att_backend_class,
    get_prefill_att_backend_class,
)


def describe_execution_requirements(model):
    args = model.args
    phases = set()
    if not args.disable_cudagraph:
        phases.add(AttentionPhase.DECODE)
    if args.enable_prefill_cudagraph:
        phases.add(AttentionPhase.PREFILL)
    transfers = set()
    if args.enable_cpu_cache:
        transfers.add("cpu")
    if args.enable_dp_prompt_cache_fetch:
        transfers.add("dp")
    if args.run_mode in ("prefill", "decode"):
        transfers.add("pd")
    return ExecutionRequirements(
        frozenset(phases),
        2 if args.enable_decode_microbatch_overlap or args.enable_prefill_microbatch_overlap else 1,
        getattr(model, "max_req_num", 1),
        max(getattr(model, "max_seq_length", 1), getattr(model, "graph_max_len_in_batch", 1)),
        args.enable_ep_moe,
        frozenset(transfers),
    )
