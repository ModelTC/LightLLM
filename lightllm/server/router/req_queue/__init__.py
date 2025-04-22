from .continues_batch.impl import ContinuesBatchQueue
from .continues_batch.impl_for_pd_decode import QueueForPDDecode
from .chunked_prefill.impl_for_pd_prefill import QueueForPDChunkedPrefill
from .chunked_prefill.impl import ChunkedPrefillQueue
from .chunked_prefill.beam_impl import ChunkedBeamContinuesBatchQueue
from .dp_base_queue import DpQueue


def _get_req_queue_class(args, router, dp_size_in_node: int):
    if args.diverse_mode:
        return ChunkedBeamContinuesBatchQueue
    if args.token_healing_mode:
        return ChunkedPrefillQueue
    if args.output_constraint_mode != "none":
        return ChunkedPrefillQueue
    if args.first_token_constraint_mode:
        return ChunkedPrefillQueue
    if args.run_mode == "decode":
        return QueueForPDDecode
    if args.run_mode == "prefill":
        return QueueForPDChunkedPrefill

    if args.disable_chunked_prefill:
        return ContinuesBatchQueue
    else:
        return ChunkedPrefillQueue


def build_req_queue(args, router, dp_size_in_node: int):
    queue_class = _get_req_queue_class(args, router, dp_size_in_node)

    if dp_size_in_node == 1:
        return queue_class(args, router, 0, dp_size_in_node)
    else:
        return DpQueue(args, router, queue_class, dp_size_in_node)
