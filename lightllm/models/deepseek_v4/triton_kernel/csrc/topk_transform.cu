// Copyright 2023-2024 SGLang Team
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4 c4-indexer top-k selection + page-translate.
//
// Adapted from SGLang commit 8cea0473ea5299bc04885f8f6ba71269415a39b5,
// python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh. LightLLM replaces
// the tvm::ffi TensorView / TensorMatcher binding with a torch cpp_extension
// launcher while preserving the original Hopper PDL protocol.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr uint32_t kTopK = 512;
constexpr uint32_t kTopKBlockSize = 512;
constexpr uint32_t kSMEM = 16 * 1024 * sizeof(uint32_t);  // 64KB (bytes)

template <bool UsePDL>
__device__ __forceinline__ void pdl_wait_primary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (UsePDL) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

template <bool UsePDL>
__device__ __forceinline__ void pdl_trigger_secondary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (UsePDL) asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i,
                                                    uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

__device__ void naive_transform(const int32_t* __restrict__ page_table, int32_t* __restrict__ indices,
                                 int32_t* __restrict__ raw_indices, const uint32_t length,
                                 const uint32_t page_bits) {
  if (const auto tx = threadIdx.x; tx < length) {
    indices[tx] = page_to_indices(page_table, tx, page_bits);
    if (raw_indices != nullptr) raw_indices[tx] = tx;
  } else if (tx < kTopK) {
    indices[tx] = -1;  // fill invalid indices to -1
    if (raw_indices != nullptr) raw_indices[tx] = -1;
  }
}

__device__ void radix_topk(const float* __restrict__ input, int32_t* __restrict__ output,
                           const uint32_t length) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = kTopKBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;

  extern __shared__ uint32_t s_input_idx[][kSMEM / (2 * sizeof(int32_t))];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = kTopK;
  auto& s_histogram = _s_histogram_buf[0];

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = _s_histogram_buf[k][tx];
        if (tx + j < RADIX) {
          value += _s_histogram_buf[k][tx + j];
        }
        _s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  remain_topk -= s_histogram[threshold_bin + 1];
  if (remain_topk == 0) {
    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const uint32_t bin = convert_to_uint8(input[idx]);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw_input = input[idx];
      const uint32_t bin = convert_to_uint8(raw_input);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (pos < SMEM_INPUT_SIZE) {
          s_input_idx[0][pos] = idx;
          const auto sbin = convert_to_uint32(raw_input);
          const auto sub_bin = (sbin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r_idx = round % 2;

    const auto raw_num_input = s_num_input[r_idx];
    const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = remain_topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin2 = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin2 + 1];

    if (remain_topk == 0) {
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin2) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin2) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        } else if (bin == threshold_bin2) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              output[kTopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < SMEM_INPUT_SIZE) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto sbin = convert_to_uint32(raw_input);
              const auto sub_bin = (sbin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

struct TopKParams {
  const float* scores;
  const int32_t* seq_lens;
  const int32_t* page_table;
  int32_t* page_indices;
  int32_t* raw_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t page_bits;
};

template <bool UsePDL>
__global__ void topk_transform_kernel(const __grid_constant__ TopKParams params) {
  const uint32_t work_id = blockIdx.x;
  const uint32_t seq_len = params.seq_lens[work_id];
  const auto score_ptr = params.scores + work_id * params.score_stride;
  const auto page_ptr = params.page_table + work_id * params.page_table_stride;
  const auto indices_ptr = params.page_indices + work_id * kTopK;
  const auto raw_indices_ptr = params.raw_indices != nullptr ? params.raw_indices + work_id * kTopK : nullptr;
  const uint32_t page_bits = params.page_bits;

  pdl_wait_primary<UsePDL>();

  if (seq_len <= kTopK) {
    naive_transform(page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits);
  } else {
    __shared__ int32_t s_topk_indices[kTopK];
    radix_topk(score_ptr, s_topk_indices, seq_len);
    const auto tx = threadIdx.x;
    indices_ptr[tx] = page_to_indices(page_ptr, s_topk_indices[tx], page_bits);
    if (raw_indices_ptr != nullptr) raw_indices_ptr[tx] = s_topk_indices[tx];
  }

  pdl_trigger_secondary<UsePDL>();
}

template <bool UsePDL>
void launch_topk(const TopKParams& params, uint32_t batch_size, cudaStream_t stream) {
  constexpr uint32_t smem = kSMEM + sizeof(int32_t);
  static const cudaError_t smem_result = cudaFuncSetAttribute(
      topk_transform_kernel<UsePDL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  C10_CUDA_CHECK(smem_result);

  cudaLaunchConfig_t config{};
  config.gridDim = batch_size;
  config.blockDim = kTopKBlockSize;
  config.dynamicSmemBytes = smem;
  config.stream = stream;

  cudaLaunchAttribute attribute{};
  if constexpr (UsePDL) {
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = true;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, topk_transform_kernel<UsePDL>, params));
}

}  // namespace

void topk_transform_512_cuda(at::Tensor scores, at::Tensor seq_lens, at::Tensor page_table,
                             at::Tensor page_indices, int64_t page_size, c10::optional<at::Tensor> raw_indices) {
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(
      seq_lens.is_cuda() && page_table.is_cuda() && page_indices.is_cuda() &&
          seq_lens.get_device() == scores.get_device() && page_table.get_device() == scores.get_device() &&
          page_indices.get_device() == scores.get_device(),
      "all tensors must be on the same CUDA device");
  TORCH_CHECK(scores.dim() == 2 && scores.dtype() == at::kFloat, "scores must be [B, S] float32");
  TORCH_CHECK(
      seq_lens.dim() == 1 && seq_lens.size(0) == scores.size(0) && seq_lens.dtype() == at::kInt &&
          seq_lens.is_contiguous(),
      "seq_lens must be [B] int32 contiguous");
  TORCH_CHECK(
      page_table.dim() == 2 && page_table.size(0) == scores.size(0) && page_table.dtype() == at::kInt,
      "page_table must be [B, P] int32");
  TORCH_CHECK(page_indices.dim() == 2 && page_indices.dtype() == at::kInt && page_indices.is_contiguous(),
              "page_indices must be [B, 512] int32 contiguous");
  TORCH_CHECK(page_indices.size(0) == scores.size(0), "page_indices first dim must match scores");
  TORCH_CHECK(page_indices.size(1) == (int64_t)kTopK, "page_indices second dim must be 512");
  TORCH_CHECK(scores.stride(1) == 1 && page_table.stride(1) == 1, "scores/page_table last dim must be contiguous");
  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size must be a power of two");

  const uint32_t batch_size = scores.size(0);
  if (batch_size == 0) return;
  const uint32_t page_bits = __builtin_ctzll(page_size);

  int32_t* raw_ptr = nullptr;
  if (raw_indices.has_value()) {
    auto& r = raw_indices.value();
    TORCH_CHECK(
        r.is_cuda() && r.get_device() == scores.get_device() && r.dim() == 2 && r.size(0) == scores.size(0) &&
            r.size(1) == (int64_t)kTopK && r.dtype() == at::kInt && r.is_contiguous(),
        "raw_indices must be [B, 512] int32 contiguous on the same CUDA device");
    raw_ptr = r.data_ptr<int32_t>();
  }

  c10::cuda::CUDAGuard guard(scores.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const TopKParams params{
      scores.data_ptr<float>(),
      seq_lens.data_ptr<int32_t>(),
      page_table.data_ptr<int32_t>(),
      page_indices.data_ptr<int32_t>(),
      raw_ptr,
      scores.stride(0),
      page_table.stride(0),
      page_bits,
  };
  if (at::cuda::getCurrentDeviceProperties()->major >= 9) {
    launch_topk<true>(params, batch_size, stream);
  } else {
    launch_topk<false>(params, batch_size, stream);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_transform_512", &topk_transform_512_cuda, "DeepSeek-V4 c4 indexer top-512 + page translate");
}
