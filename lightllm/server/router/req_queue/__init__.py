from .continues_batch.impl import ContinuesBatchQueue
from .continues_batch.beam_impl import BeamContinuesBatchQueue
from .splitfuse.impl import SplitFuseQueue
from .continues_batch.pd_decode_impl import ContinuesBatchQueueForPDDecode
from .dp_base_queue import DpQueue


def build_req_queue(args, router, dp_size: int):
    queue_class = None
    if args.run_mode == "decode":
        queue_class = ContinuesBatchQueueForPDDecode
    if args.splitfuse_mode:
        queue_class = SplitFuseQueue
    if args.beam_mode:
        queue_class = BeamContinuesBatchQueue
    if args.diverse_mode:
        queue_class = BeamContinuesBatchQueue
    if args.token_healing_mode:
        queue_class = ContinuesBatchQueue
    if args.simple_constraint_mode:
        queue_class = ContinuesBatchQueue
    if args.first_token_constraint_mode:
        queue_class = ContinuesBatchQueue
    queue_class = ContinuesBatchQueue

    if dp_size == 1:
        return queue_class(args, router, 0, dp_size)
    else:
        return DpQueue(args, router, queue_class, dp_size)
