from lightllm.common.basemodel.layer_weights.meta_weights.row_sharded_embedding_weight import RowShardedEmbeddingWeight
from lightllm.common.basemodel.layer_weights.meta_weights.pinned_embedding_weight import PinnedRowShardedEmbeddingWeight
from lightllm.utils.envs_utils import get_env_start_args


def ngram_embedding_weight(dim, rows, name, dtype, shard_count):
    args = get_env_start_args()
    if not args.get("offload_ngram_embedding", False):
        return RowShardedEmbeddingWeight(dim, rows, name, dtype, shard_count)
    if args.get("hardware_platform", "cuda") != "cuda":
        raise ValueError("N-gram embedding offload requires CUDA Triton lookup")
    from ..pinned_memory import pinned_zeros

    return PinnedRowShardedEmbeddingWeight(dim, rows, name, dtype, shard_count, cpu_allocator=pinned_zeros)
