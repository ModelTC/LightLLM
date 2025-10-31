#pragma once

#include "mycuda.cuh"
#include "mytorch.cuh"

namespace atex::ops {

/**
 * @brief Host function for scattering cache data with 16-byte alignment.
 *
 * This function launches a CUDA kernel that scatters data from a contiguous block
 * back to the KV cache. The destination locations in the KV cache are determined
 * by the provided page index array.
 *
 * @tparam BlockSize Block size in bytes.
 * @tparam PageSize Page size in bytes.
 * @tparam LaunchBlockCount Number of blocks launched in the kernel
 *
 * @param kvcache Pointer to the KV cache data in GPU memory (destination).
 * @param block Source buffer in GPU memory (contiguous block data).
 * @param kv_page_indexer Array of page indices that specify where in kvcache to scatter data.
 * @param num_of_page Number of pages to process.
 * @param num_of_layer Number of layers.
 * @param layer_stride Stride between layers in bytes.
 * @param stream CUDA stream for asynchronous execution.
 */
void cache_scatter_align16(char *kvcache, const char *block, const int32_t *kv_page_indexer, int64_t num_of_page,
                           int64_t num_of_layer, int64_t page_size, int64_t layer_stride, int64_t page_stride,
                           int64_t block_size,
                           cudaStream_t stream);

} // namespace atex::ops
