from .chunked_prefill.impl_for_pd_decode import QueueForPDDecode
from .chunked_prefill.impl import ChunkedPrefillQueue
from .chunked_prefill.beam_impl import ChunkedBeamContinuesBatchQueue
from .chunked_prefill.impl_for_nixl_pd import NIXLPDQueue
from .chunked_prefill.impl_past_future import PastFutureQueue
from .dp_base_queue import DpQueue


def _get_req_queue_class(args, router, dp_size_in_node: int):
    if args.past_future_scheduler:
        if args.diverse_mode:
            raise ValueError("Diverse mode is not supported with past future scheduler yet")
        chunked_prefill_queue_impl = PastFutureQueue
    else:
        chunked_prefill_queue_impl = ChunkedPrefillQueue

    if args.diverse_mode:
        return ChunkedBeamContinuesBatchQueue
    if args.token_healing_mode:
        return chunked_prefill_queue_impl
    if args.output_constraint_mode != "none":
        return chunked_prefill_queue_impl
    if args.first_token_constraint_mode:
        return chunked_prefill_queue_impl
    if args.run_mode in ["decode"]:
        return QueueForPDDecode
    if args.run_mode in ["prefill"]:
        return chunked_prefill_queue_impl
    if args.run_mode in ["nixl_prefill", "nixl_decode"]:
        return NIXLPDQueue

    if args.disable_chunked_prefill:
        # 虽然也使用chuncked prefill queue 但是由于 args.chunked_prefill_size = args.max_req_total_len
        # 所以调度的实际行为类似过去的 continues batch 调度，所以将两种调度的实现统一为一种实现，减少代码重复。
        return chunked_prefill_queue_impl
    else:
        return chunked_prefill_queue_impl


def build_req_queue(args, router, dp_size_in_node: int):
    queue_class = _get_req_queue_class(args, router, dp_size_in_node)

    if dp_size_in_node == 1:
        return queue_class(args, router, 0, dp_size_in_node)
    else:
        return DpQueue(args, router, queue_class, dp_size_in_node)
