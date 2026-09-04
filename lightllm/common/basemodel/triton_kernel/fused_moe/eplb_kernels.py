import triton
import triton.language as tl


@triton.jit
def eplb_replica_index(token_index, logical_id, replica_count):
    """Choose a replica with independent phases for a token's top-k experts."""
    token_hash = token_index.to(tl.uint32) * 2654435769
    expert_hash = logical_id.to(tl.uint32) * 2246822519
    return (token_hash + expert_hash) % replica_count.to(tl.uint32)
