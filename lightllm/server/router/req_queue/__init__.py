from .continues_batch.impl import ContinuesBatchQueue
from .continues_batch.beam_impl import BeamContinuesBatchQueue
from .splitfuse.impl import SplitFuseQueue


def build_req_queue(args, router):
    if args.splitfuse_mode:
        return SplitFuseQueue(args, router)
    if args.beam_mode:
        return BeamContinuesBatchQueue(args, router)
    if args.diverse_mode:
        return BeamContinuesBatchQueue(args, router)
    return ContinuesBatchQueue(args, router)
