/*
 * NVFP4 small-query tile staging, using FlashInfer's FA2 device implementation.
 * Derived from FlashInfer, Copyright (c) 2023 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 * https://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once
#include <flashinfer/attention/prefill.cuh>

namespace flashinfer {
namespace lightllm_internal {
template <class Base, class T, unsigned KV, unsigned D>
struct RepackStorage : Base {
  alignas(16) T repack[KV * D];
  __device__ __forceinline__ T* kv_smem_repack_ptr() { return repack; }
};

template <class Base>
struct RepackTraits : Base {
  static constexpr bool USE_KV_REPACK = true;
  using SharedStoragePaged = RepackStorage<typename Base::SharedStoragePaged, typename Base::DTypeQ,
                                           Base::CTA_TILE_KV, Base::HEAD_DIM_QK>;
};

} // namespace lightllm_internal

template <bool SAME, unsigned CTA, unsigned DQ, unsigned DV, flashinfer::PosEncodingMode POS,
          bool REDUCE_FP16, flashinfer::MaskMode MASK, class Variant, class Params>
cudaError_t LightLLMBatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl, cudaStream_t stream) {
  using namespace flashinfer;
  if constexpr (CTA != 16 || DQ != DV || (DQ != 128 && DQ != 256) ||
                POS != PosEncodingMode::kNone || !is_fp4_type_v<typename Params::DTypeKV>) {
    return flashinfer::BatchPrefillWithPagedKVCacheDispatched<SAME, CTA, DQ, DV, POS, REDUCE_FP16,
                                                              MASK, Variant>(params, tmp_v, tmp_s, enable_pdl, stream);
  } else {
    using Q = typename Params::DTypeQ;
    using KV = typename Params::DTypeKV;
    using O = typename Params::DTypeO;
    using Acc = std::conditional_t<REDUCE_FP16 && std::is_same_v<Q, half>, half, float>;
    constexpr unsigned MMA_KV = DQ == 128 ? 2 : 1;
    using Original = KernelTraits<MASK, CTA, 1, MMA_KV, DQ / 16, DV / 16, 1, 4, POS,
                                  Q, KV, O, Acc, typename Params::IdType, Variant, true>;
    using Traits = lightllm_internal::RepackTraits<Original>;
    static_assert(!Traits::IsInvalid());
    if (params.padded_batch_size == 0) return cudaSuccess;
    dim3 grid(params.padded_batch_size, 1, params.paged_kv.num_heads);
    dim3 block(32, 1, 4);
    auto kernel = BatchPrefillWithPagedKVCacheKernel<SAME, Traits, Params>;
    constexpr size_t smem = sizeof(typename Traits::SharedStoragePaged);
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    auto output = params.o;
    auto lse = params.lse;
    params.partition_kv = tmp_v != nullptr;
    if (tmp_v != nullptr) { params.o = tmp_v; params.lse = tmp_s; }
    if (enable_pdl) {
      cudaLaunchAttribute attribute;
      attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute.val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t config{};
      config.attrs = &attribute; config.numAttrs = 1;
      config.gridDim = grid; config.blockDim = block; config.dynamicSmemBytes = smem; config.stream = stream;
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
    } else {
      void* args[] = {(void*)&params};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, smem, stream));
    }
    if (tmp_v != nullptr) {
      if constexpr (Variant::use_softmax) {
        FLASHINFER_CUDA_CALL(VariableLengthMergeStates(tmp_v, tmp_s, params.merge_indptr, output, lse,
              params.max_total_num_rows, params.total_num_rows, params.num_qo_heads, DV, enable_pdl, stream));
      } else {
        FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(tmp_v, params.merge_indptr, output,
              params.max_total_num_rows, params.total_num_rows, params.num_qo_heads, DV, enable_pdl, stream));
      }
    }
    return cudaSuccess;
  }
}
} // namespace flashinfer

// Only this JIT module's generated instantiations select the small-tile launcher.
// The upstream headers and every other FlashInfer module remain untouched.
#define BatchPrefillWithPagedKVCacheDispatched LightLLMBatchPrefillWithPagedKVCacheDispatched
