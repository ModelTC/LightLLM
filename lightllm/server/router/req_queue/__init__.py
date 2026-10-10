from .chunked_prefill.impl import ChunkedPrefillQueue
from .chunked_prefill.beam_impl import ChunkedBeamContinuesBatchQueue
from .chunked_prefill.impl_for_pd_prefill import PDPrefillQueue
from .chunked_prefill.impl_for_pd_decode import PDDecodeQueue
from .dp_base_queue import DpQueue


def _get_req_queue_class(args, router, dp_size_in_node: int):
    if args.run_mode == "prefill":
        return PDPrefillQueue
    if args.run_mode == "decode":
        return PDDecodeQueue

    if args.diverse_mode:
        return ChunkedBeamContinuesBatchQueue
    # 禁用 chunked prefill 时，chunked_prefill_size 会设为 max_req_total_len，仍复用同一队列。
    return ChunkedPrefillQueue


def build_req_queue(args, router, dp_size_in_node: int):
    queue_class = _get_req_queue_class(args, router, dp_size_in_node)

    if dp_size_in_node == 1:
        return queue_class(args, router, 0, dp_size_in_node)
    else:
        return DpQueue(args, router, queue_class, dp_size_in_node)
