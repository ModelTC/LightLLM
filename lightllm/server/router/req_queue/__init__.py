from .continues_batch.impl import ContinuesBatchQueue
from .continues_batch.beam_impl import BeamContinuesBatchQueue
from .continues_batch.pd_decode_impl import ContinuesBatchQueueForPDDecode
from .chunked_prefill.impl import ChunkedPrefillQueue
from .dp_base_queue import DpQueue


def build_req_queue(args, router, dp_size_in_node: int):
    queue_class = None
    if args.run_mode == "decode":
        queue_class = ContinuesBatchQueueForPDDecode
    if args.diverse_mode:
        queue_class = BeamContinuesBatchQueue
    if args.enable_chunked_prefill:
        queue_class = ChunkedPrefillQueue
    if args.token_healing_mode:
        queue_class = ContinuesBatchQueue
    if args.output_constraint_mode != "none":
        queue_class = ContinuesBatchQueue
    if args.first_token_constraint_mode:
        queue_class = ContinuesBatchQueue
    if queue_class is None:
        queue_class = ContinuesBatchQueue

    if dp_size_in_node == 1:
        return queue_class(args, router, 0, dp_size_in_node)
    else:
        return DpQueue(args, router, queue_class, dp_size_in_node)
