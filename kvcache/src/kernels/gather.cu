#include "config.h"
#include "kernels/gather.h"

#include <stdexcept>

namespace atex::ops {

/**
 * @brief CUDA kernel for aligned gathering of cache data.
 *
 * @tparam TPB Threads per block.
 * @tparam VPT Number of elements processed per thread.
 */
template <int64_t TPB, int64_t VPT>
__global__ void device_cache_gather_align16(const int32_t *__restrict__ src, int32_t *__restrict__ dst,
                                            const int32_t *__restrict__ page_idx_array, const int64_t num_of_page,
                                            const int64_t num_of_layer, const int64_t page_size,
                                            const int64_t layer_stride, const int64_t page_stride) {

  for (int64_t page_idx = blockIdx.x; page_idx < num_of_page; page_idx += gridDim.x) {
    const int64_t src_page_offset = page_idx_array[page_idx] * page_stride;
    const int64_t dst_page_offset = page_idx * page_size;

    for (int64_t offset = threadIdx.x * VPT; offset < (page_size * num_of_layer); offset += TPB * VPT) {
      const int64_t src_layer_offset = (offset / page_size) * layer_stride;
      const int64_t dst_layer_offset = (offset / page_size) * num_of_page * page_size;
      const int64_t page_offset = offset % page_size;

      vec_copy<sizeof(int32_t) * VPT>(src + src_page_offset + src_layer_offset + page_offset, // Read from KV cache
                                      dst + dst_page_offset + dst_layer_offset + page_offset  // Store in cache block
      );
    }
  }
}

/**
 * @brief Host function for gathering cache data with 16-byte alignment.
 *
 * @tparam BlockSize Block size in bytes.
 * @tparam PageSize Page size in bytes.
 * @tparam Number of blocks launched in the kernel
 *
 * @param kvcache Pointer to the KV cache data in GPU memory.
 * @param block Output buffer in GPU memory.
 * @param kv_page_indexer Array of page indices.
 * @param num_of_page Number of pages to process.
 * @param num_of_layer Number of layers.
 * @param layer_stride Stride between layers in bytes.
 * @param stream CUDA stream for asynchronous execution.
 */
void cache_gather_align16(const char *kvcache, char *block, const int32_t *kv_page_indexer, const int64_t num_of_page,
                          const int64_t num_of_layer, const int64_t page_size, const int64_t layer_stride,
                          const int64_t page_stride, const int64_t block_size, cudaStream_t stream) {
  constexpr int64_t LaunchBlockCount = ACS_KernelBlocks;

  if (page_size % 16 != 0) {
    throw std::runtime_error("Page size must be aligned to 16 bytes.");
  }
  if (block_size % 16 != 0) {
    throw std::runtime_error("Block size must be aligned to 16 bytes.");
  }
  if (block_size < page_size) {
    throw std::runtime_error("Block size must be greater than or equal to page size.");
  }
  if (page_size % sizeof(int32_t) != 0 || layer_stride % sizeof(int32_t) != 0) {
    throw std::runtime_error("Page size and layer stride must be multiples of sizeof(int32_t).");
  }
  if (page_stride % sizeof(int32_t) != 0) {
    throw std::runtime_error("Page stride must be a multiple of sizeof(int32_t).");
  }
  if (num_of_page * page_size > block_size) {
    throw std::runtime_error("Too many pages: total memory exceeds block size.");
  }

  const int64_t page_elems = page_size / sizeof(int32_t);
  const int64_t layer_stride_elems = layer_stride / sizeof(int32_t);
  const int64_t page_stride_elems = page_stride / sizeof(int32_t);

  if (page_size >= 2048) {
    constexpr int64_t TPB = 1024;
    constexpr int64_t VPT = 4;
    device_cache_gather_align16<TPB, VPT><<<LaunchBlockCount, TPB, 0, stream>>>(
        reinterpret_cast<const int32_t *>(kvcache), reinterpret_cast<int32_t *>(block), kv_page_indexer, num_of_page,
        num_of_layer, page_elems, layer_stride_elems, page_stride_elems);
    return;
  }

  switch (page_size) {
  case 64: { // Default small-page configuration
    constexpr int64_t TPB = 128;
    constexpr int64_t VPT = 4;
    device_cache_gather_align16<TPB, VPT><<<LaunchBlockCount, TPB, 0, stream>>>(
        reinterpret_cast<const int32_t *>(kvcache), reinterpret_cast<int32_t *>(block), kv_page_indexer, num_of_page,
        num_of_layer, page_elems, layer_stride_elems, page_stride_elems);
    return;
  }
  case 128: {
    constexpr int64_t TPB = 256;
    constexpr int64_t VPT = 4;
    device_cache_gather_align16<TPB, VPT><<<LaunchBlockCount, TPB, 0, stream>>>(
        reinterpret_cast<const int32_t *>(kvcache), reinterpret_cast<int32_t *>(block), kv_page_indexer, num_of_page,
        num_of_layer, page_elems, layer_stride_elems, page_stride_elems);
    return;
  }
  case 256: {
    constexpr int64_t TPB = 256;
    constexpr int64_t VPT = 4;
    device_cache_gather_align16<TPB, VPT><<<LaunchBlockCount, TPB, 0, stream>>>(
        reinterpret_cast<const int32_t *>(kvcache), reinterpret_cast<int32_t *>(block), kv_page_indexer, num_of_page,
        num_of_layer, page_elems, layer_stride_elems, page_stride_elems);
    return;
  }
  case 512: {
    constexpr int64_t TPB = 512;
    constexpr int64_t VPT = 4;
    device_cache_gather_align16<TPB, VPT><<<LaunchBlockCount, TPB, 0, stream>>>(
        reinterpret_cast<const int32_t *>(kvcache), reinterpret_cast<int32_t *>(block), kv_page_indexer, num_of_page,
        num_of_layer, page_elems, layer_stride_elems, page_stride_elems);
    return;
  }
  case 1024: {
    constexpr int64_t TPB = 512;
    constexpr int64_t VPT = 4;
    device_cache_gather_align16<TPB, VPT><<<LaunchBlockCount, TPB, 0, stream>>>(
        reinterpret_cast<const int32_t *>(kvcache), reinterpret_cast<int32_t *>(block), kv_page_indexer, num_of_page,
        num_of_layer, page_elems, layer_stride_elems, page_stride_elems);
    return;
  }
  case 1152: { // Legacy DeepSeek V3/R1 models
    constexpr int64_t TPB = 576;
    constexpr int64_t VPT = 4;
    device_cache_gather_align16<TPB, VPT><<<LaunchBlockCount, TPB, 0, stream>>>(
        reinterpret_cast<const int32_t *>(kvcache), reinterpret_cast<int32_t *>(block), kv_page_indexer, num_of_page,
        num_of_layer, page_elems, layer_stride_elems, page_stride_elems);
    return;
  }
  default: {
    constexpr int64_t TPB = 1024;
    constexpr int64_t VPT = 4;
    device_cache_gather_align16<TPB, VPT><<<LaunchBlockCount, TPB, 0, stream>>>(
        reinterpret_cast<const int32_t *>(kvcache), reinterpret_cast<int32_t *>(block), kv_page_indexer, num_of_page,
        num_of_layer, page_elems, layer_stride_elems, page_stride_elems);
    return;
  }
  }
}

} // namespace atex::ops