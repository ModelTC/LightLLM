from lightllm.common.basemodel.triton_kernel.embedding import embedding
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.platform.ops.embedding import EmbeddingOps
from lightllm.platform.plugin.ops import register_ops
from lightllm.server.embed_cache.copy_to_cache import offload_embed_tensor_to_cache

register_ops(
    EmbeddingOps,
    platforms=("cuda", "musa"),
    embedding=embedding,
    multimodal_emb=multimodal_emb,
    offload_embed_tensor_to_cache=offload_embed_tensor_to_cache,
)
