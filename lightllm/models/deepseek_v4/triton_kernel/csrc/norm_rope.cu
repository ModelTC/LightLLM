// Copyright 2023-2024 SGLang Team
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4 main Q/K and indexer-Q fused kernels.
//
// Adapted from SGLang commit 8cea0473ea5299bc04885f8f6ba71269415a39b5,
// python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh.  This local
// port keeps the device math, warp mapping, launch bounds, and Hopper PDL
// protocol, while replacing tvm::ffi and the generic SGLang headers with a
// small torch cpp_extension binding specialized for LightLLM's DSV4 shapes.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

constexpr uint32_t kWarpThreads = 32;
constexpr uint32_t kFusedQBlockSize = 128;
constexpr uint32_t kFusedQNumWarps = kFusedQBlockSize / kWarpThreads;
constexpr uint32_t kFusedKBlockSize = 256;
constexpr uint32_t kFusedKNumWarps = kFusedKBlockSize / kWarpThreads;

constexpr int64_t kMainHeadDim = 512;
constexpr int64_t kMainRopeDim = 64;
constexpr int64_t kMainNopeDim = kMainHeadDim - kMainRopeDim;
constexpr int64_t kIndexerHeadDim = 128;
constexpr int64_t kIndexerRopeDim = 64;
constexpr uint32_t kFlashMLAPageSize = 128;
constexpr int32_t kFlashMLAPageBits = 7;
constexpr int64_t kFlashMLADataBytes = 576;
constexpr int64_t kFlashMLAScaleBytes = 8;
constexpr int64_t kFlashMLABytesPerToken = kFlashMLADataBytes + kFlashMLAScaleBytes;
constexpr int64_t kFlashMLAPageBytes =
    ((kFlashMLABytesPerToken * kFlashMLAPageSize + kFlashMLADataBytes - 1) / kFlashMLADataBytes) *
    kFlashMLADataBytes;

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
  T data[N];

  __device__ T& operator[](int i) { return data[i]; }
  __device__ const T& operator[](int i) const { return data[i]; }
};

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
  }
  return value;
}

template <int NumThreads, typename T>
__device__ __forceinline__ T warp_reduce_sum_width(T value) {
  static_assert(NumThreads > 0 && NumThreads <= 32 && (NumThreads & (NumThreads - 1)) == 0);
#pragma unroll
  for (int mask = NumThreads / 2; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
  }
  return value;
}

template <bool UsePDL>
__device__ __forceinline__ void pdl_wait_primary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (UsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

template <bool UsePDL>
__device__ __forceinline__ void pdl_trigger_secondary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (UsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}

__device__ __forceinline__ int32_t cast_to_ue8m0(float x) {
  const uint32_t u = __float_as_uint(x);
  const int32_t exp = static_cast<int32_t>((u >> 23) & 0xffu);
  const uint32_t mantissa = u & 0x7fffffu;
  return exp + (mantissa != 0);
}

__device__ __forceinline__ float inv_scale_ue8m0(int32_t exp) {
  return __uint_as_float(static_cast<uint32_t>(127 + 127 - exp) << 23);
}

__device__ __forceinline__ uint16_t pack_fp8(float x, float y) {
  constexpr float kFp8Max = 448.0f;
  const float2 values = {
      fmaxf(fminf(x, kFp8Max), -kFp8Max),
      fmaxf(fminf(y, kFp8Max), -kFp8Max),
  };
  return __nv_cvt_float2_to_fp8x2(values, __NV_SATFINITE, __NV_E4M3);
}

template <typename Kernel, typename... Args>
void launch_kernel(Kernel kernel, dim3 grid, dim3 block, cudaStream_t stream, bool enable_pdl, Args... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;

  cudaLaunchAttribute attribute{};
  if (enable_pdl) {
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = true;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }

  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

bool device_supports_pdl() {
  return at::cuda::getCurrentDeviceProperties()->major >= 9;
}

void check_cuda_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_same_device(const at::Tensor& reference, const at::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  TORCH_CHECK(tensor.get_device() == reference.get_device(), name, " must be on the same CUDA device as the input");
}

struct FusedQNormRopeParams {
  const __nv_bfloat16* q_input;
  __nv_bfloat16* q_output;
  const float* freqs_cis;
  const void* positions;
  int64_t q_input_stride_batch;
  int64_t q_output_stride_batch;
  uint32_t batch_size;
  uint32_t num_q_heads;
  float eps;
};

template <typename PosT, bool UsePDL>
__global__ __launch_bounds__(kFusedQBlockSize, 16) void fused_q_norm_rope_kernel(
    const __grid_constant__ FusedQNormRopeParams params) {
  constexpr int64_t kVecSize = 8;
  constexpr int64_t kLocalSize = kMainHeadDim / (kWarpThreads * kVecSize);
  constexpr uint32_t kRopeVecs = kMainRopeDim / kVecSize;
  using Storage = AlignedVector<__nv_bfloat16, kVecSize>;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t work_id = blockIdx.x * kFusedQNumWarps + warp_id;
  const uint32_t total_works = params.batch_size * params.num_q_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_q_heads;
  const uint32_t head_id = work_id % params.num_q_heads;
  const auto* input_ptr =
      params.q_input + batch_id * params.q_input_stride_batch + head_id * kMainHeadDim;
  auto* output_ptr =
      params.q_output + batch_id * params.q_output_stride_batch + head_id * kMainHeadDim;
  const int32_t position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);

  __shared__ Storage rope_storage[kFusedQNumWarps][kRopeVecs];

  pdl_wait_primary<UsePDL>();

  Storage input_vec[kLocalSize];
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
    input_vec[i] = reinterpret_cast<const Storage*>(input_ptr)[i * kWarpThreads + lane_id];
  }

  const float2 freq = reinterpret_cast<const float2*>(params.freqs_cis + position * kMainRopeDim)[lane_id];

  float sum_of_squares = 0.0f;
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const float x = __bfloat162float(input_vec[i][j]);
      sum_of_squares += x * x;
    }
  }
  sum_of_squares = warp_reduce_sum(sum_of_squares);
  const float norm_factor = rsqrtf(sum_of_squares / static_cast<float>(kMainHeadDim) + params.eps);

#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      input_vec[i][j] = __float2bfloat16_rn(__bfloat162float(input_vec[i][j]) * norm_factor);
    }
  }

  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeVecs;
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
    if (i == kLocalSize - 1 && is_rope_lane) {
      rope_storage[warp_id][lane_id - (kWarpThreads - kRopeVecs)] = input_vec[i];
    } else {
      reinterpret_cast<Storage*>(output_ptr)[i * kWarpThreads + lane_id] = input_vec[i];
    }
  }
  __syncwarp();

  pdl_trigger_secondary<UsePDL>();

  const auto elem = reinterpret_cast<const __nv_bfloat162*>(rope_storage[warp_id])[lane_id];
  const float2 x = __bfloat1622float2(elem);
  const float out_real = x.x * freq.x - x.y * freq.y;
  const float out_imag = x.x * freq.y + x.y * freq.x;
  reinterpret_cast<__nv_bfloat162*>(output_ptr + kMainNopeDim)[lane_id] =
      __floats2bfloat162_rn(out_real, out_imag);
}

struct FusedKNormRopeFlashMLAParams {
  const __nv_bfloat16* kv;
  const __nv_bfloat16* kv_weight;
  const float* freqs_cis;
  const void* positions;
  const int32_t* out_loc;
  uint8_t* kvcache;
  int64_t kv_stride_batch;
  uint32_t batch_size;
  float eps;
};

template <typename PosT, bool UsePDL>
__global__ __launch_bounds__(kFusedKBlockSize, 8) void fused_k_norm_rope_flashmla_kernel(
    const __grid_constant__ FusedKNormRopeFlashMLAParams params) {
  using Storage = AlignedVector<__nv_bfloat16, 2>;

  const uint32_t tx = threadIdx.x;
  const uint32_t warp_id = tx / kWarpThreads;
  const uint32_t lane_id = tx % kWarpThreads;
  const uint32_t work_id = blockIdx.x;
  if (work_id >= params.batch_size) return;

  const auto* input_ptr = params.kv + work_id * params.kv_stride_batch;
  const int32_t position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const int32_t out_loc = params.out_loc[work_id];
  const float* freqs_cis = params.freqs_cis + position * kMainRopeDim;

  pdl_wait_primary<UsePDL>();

  const Storage input_vec = reinterpret_cast<const Storage*>(input_ptr)[tx];
  const Storage weight_vec = reinterpret_cast<const Storage*>(params.kv_weight)[tx];
  float2 data;
  float2 freq{};
  if (warp_id == kFusedKNumWarps - 1) {
    freq = reinterpret_cast<const float2*>(freqs_cis)[lane_id];
  }

  float sum_of_squares = 0.0f;
  const float input_x = __bfloat162float(input_vec[0]);
  const float input_y = __bfloat162float(input_vec[1]);
  sum_of_squares += input_x * input_x;
  sum_of_squares += input_y * input_y;
  const float warp_sum = warp_reduce_sum(sum_of_squares);

  __shared__ float partial_sums[kFusedKNumWarps];
  if (lane_id == 0) partial_sums[warp_id] = warp_sum;
  __syncthreads();
  sum_of_squares = warp_reduce_sum_width<kFusedKNumWarps>(partial_sums[lane_id % kFusedKNumWarps]);
  const float norm_factor = rsqrtf(sum_of_squares / static_cast<float>(kMainHeadDim) + params.eps);
  data.x = input_x * norm_factor * __bfloat162float(weight_vec[0]);
  data.y = input_y * norm_factor * __bfloat162float(weight_vec[1]);

  const int32_t page = out_loc >> kFlashMLAPageBits;
  const int32_t offset = out_loc & (kFlashMLAPageSize - 1);
  auto* page_ptr = params.kvcache + static_cast<int64_t>(page) * kFlashMLAPageBytes;
  auto* value_ptr = page_ptr + static_cast<int64_t>(offset) * kFlashMLADataBytes;

  pdl_trigger_secondary<UsePDL>();

  if (warp_id == kFusedKNumWarps - 1) {
    const float out_real = data.x * freq.x - data.y * freq.y;
    const float out_imag = data.x * freq.y + data.y * freq.x;
    reinterpret_cast<__nv_bfloat162*>(value_ptr + kMainNopeDim)[lane_id] =
        __floats2bfloat162_rn(out_real, out_imag);
  } else {
    const float abs_max = warp_reduce_max(fmaxf(fabsf(data.x), fabsf(data.y)));
    const float scale_raw = fmaxf(1.0e-4f, abs_max) / 448.0f;
    const int32_t scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const float inv_scale = inv_scale_ue8m0(scale_ue8m0);
    reinterpret_cast<uint16_t*>(value_ptr)[tx] = pack_fp8(data.x * inv_scale, data.y * inv_scale);
    if (lane_id == 0) {
      auto* scale_ptr = page_ptr + kFlashMLAPageSize * kFlashMLADataBytes + offset * kFlashMLAScaleBytes;
      scale_ptr[warp_id] = static_cast<uint8_t>(scale_ue8m0);
    }
  }
}

struct FusedQIndexerParams {
  const __nv_bfloat16* q_input;
  uint8_t* q_fp8;
  const __nv_bfloat16* weight;
  float* weights_out;
  float weight_scale;
  const float* freqs_cis;
  const void* positions;
  uint32_t batch_size;
  uint32_t num_heads;
};

template <typename PosT, bool UsePDL>
__global__ __launch_bounds__(kFusedQBlockSize, 16) void fused_q_indexer_rope_hadamard_quant_kernel(
    const __grid_constant__ FusedQIndexerParams params) {
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeVecs = kIndexerRopeDim / kVecSize;
  using Storage = AlignedVector<__nv_bfloat16, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t work_id = blockIdx.x * kFusedQNumWarps + warp_id;
  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const int32_t position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
  const auto* input_ptr = params.q_input + static_cast<int64_t>(work_id) * kIndexerHeadDim;
  const float* freqs_cis = params.freqs_cis + position * kIndexerRopeDim;
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeVecs;

  pdl_wait_primary<UsePDL>();

  const float weight_value = __bfloat162float(params.weight[work_id]);
  const Storage input_vec = reinterpret_cast<const Storage*>(input_ptr)[lane_id];
  Float4 data;
  Float4 freq{};
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) data[i] = __bfloat162float(input_vec[i]);
  if (is_rope_lane) {
    freq = reinterpret_cast<const Float4*>(freqs_cis)[lane_id - (kWarpThreads - kRopeVecs)];
    const float x_real = data[0];
    const float x_imag = data[1];
    const float y_real = data[2];
    const float y_imag = data[3];
    data[0] = x_real * freq[0] - x_imag * freq[1];
    data[1] = x_real * freq[1] + x_imag * freq[0];
    data[2] = y_real * freq[2] - y_imag * freq[3];
    data[3] = y_real * freq[3] + y_imag * freq[2];
  }

  pdl_trigger_secondary<UsePDL>();

  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a1;
    data[1] = a0 - a1;
    data[2] = a2 + a3;
    data[3] = a2 - a3;
  }
  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a2;
    data[1] = a1 + a3;
    data[2] = a0 - a2;
    data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float other = __shfl_xor_sync(0xffffffffu, data[i], mask, kWarpThreads);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
  constexpr float kHadamardScale = 0.08838834764831845f;  // 1 / sqrt(128)
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) data[i] *= kHadamardScale;

  float local_max = fabsf(data[0]);
#pragma unroll
  for (int i = 1; i < kVecSize; ++i) local_max = fmaxf(local_max, fabsf(data[i]));
  const float abs_max = warp_reduce_max(local_max);
  const float scale = fmaxf(1.0e-4f, abs_max) / 448.0f;
  const float inv_scale = 1.0f / scale;
  AlignedVector<uint16_t, 2> result;
  result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
  result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
  auto* output = reinterpret_cast<AlignedVector<uint16_t, 2>*>(
      params.q_fp8 + static_cast<int64_t>(work_id) * kIndexerHeadDim);
  output[lane_id] = result;
  params.weights_out[work_id] = weight_value * params.weight_scale * scale;
}

template <typename PosT, bool UsePDL>
void launch_q_norm(const FusedQNormRopeParams& params, cudaStream_t stream) {
  const uint32_t works = params.batch_size * params.num_q_heads;
  const dim3 grid((works + kFusedQNumWarps - 1) / kFusedQNumWarps);
  launch_kernel(fused_q_norm_rope_kernel<PosT, UsePDL>, grid, kFusedQBlockSize, stream, UsePDL, params);
}

template <typename PosT, bool UsePDL>
void launch_k_norm(const FusedKNormRopeFlashMLAParams& params, cudaStream_t stream) {
  launch_kernel(
      fused_k_norm_rope_flashmla_kernel<PosT, UsePDL>, params.batch_size, kFusedKBlockSize, stream, UsePDL, params);
}

template <typename PosT, bool UsePDL>
void launch_indexer(const FusedQIndexerParams& params, cudaStream_t stream) {
  const uint32_t works = params.batch_size * params.num_heads;
  const dim3 grid((works + kFusedQNumWarps - 1) / kFusedQNumWarps);
  launch_kernel(
      fused_q_indexer_rope_hadamard_quant_kernel<PosT, UsePDL>, grid, kFusedQBlockSize, stream, UsePDL, params);
}

void fused_q_norm_rope_cuda(
    const at::Tensor& q_input,
    const at::Tensor& q_output,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions,
    double eps) {
  check_cuda_tensor(q_input, "q_input");
  check_same_device(q_input, q_output, "q_output");
  check_same_device(q_input, freqs_cis, "freqs_cis");
  check_same_device(q_input, positions, "positions");
  TORCH_CHECK(q_input.scalar_type() == at::kBFloat16, "q_input must be bfloat16");
  TORCH_CHECK(q_output.scalar_type() == at::kBFloat16, "q_output must be bfloat16");
  TORCH_CHECK(q_input.sizes() == q_output.sizes(), "q_input and q_output shapes must match");
  TORCH_CHECK(q_input.dim() == 3 && q_input.size(2) == kMainHeadDim, "q_input must be [B, H, 512]");
  TORCH_CHECK(q_input.stride(2) == 1 && q_input.stride(1) == kMainHeadDim, "q_input head rows must be contiguous");
  TORCH_CHECK(
      q_output.stride(2) == 1 && q_output.stride(1) == kMainHeadDim, "q_output head rows must be contiguous");
  TORCH_CHECK(
      freqs_cis.dim() == 2 && freqs_cis.size(1) == kMainRopeDim && freqs_cis.scalar_type() == at::kFloat &&
          freqs_cis.is_contiguous(),
      "freqs_cis must be contiguous [max_pos, 64] float32");
  TORCH_CHECK(
      positions.dim() == 1 && positions.size(0) == q_input.size(0) && positions.is_contiguous(),
      "positions must be contiguous [B]");
  TORCH_CHECK(
      positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong,
      "positions must be int32 or int64");
  if (q_input.size(0) == 0) return;

  c10::cuda::CUDAGuard guard(q_input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const FusedQNormRopeParams params{
      reinterpret_cast<const __nv_bfloat16*>(q_input.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q_output.data_ptr()),
      freqs_cis.data_ptr<float>(),
      positions.data_ptr(),
      q_input.stride(0),
      q_output.stride(0),
      static_cast<uint32_t>(q_input.size(0)),
      static_cast<uint32_t>(q_input.size(1)),
      static_cast<float>(eps),
  };
  const bool use_pdl = device_supports_pdl();
  if (positions.scalar_type() == at::kInt) {
    use_pdl ? launch_q_norm<int32_t, true>(params, stream) : launch_q_norm<int32_t, false>(params, stream);
  } else {
    use_pdl ? launch_q_norm<int64_t, true>(params, stream) : launch_q_norm<int64_t, false>(params, stream);
  }
}

void fused_k_norm_rope_flashmla_cuda(
    const at::Tensor& kv,
    const at::Tensor& kv_weight,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions,
    const at::Tensor& out_loc,
    const at::Tensor& kvcache,
    double eps,
    int64_t page_size) {
  check_cuda_tensor(kv, "kv");
  check_same_device(kv, kv_weight, "kv_weight");
  check_same_device(kv, freqs_cis, "freqs_cis");
  check_same_device(kv, positions, "positions");
  check_same_device(kv, out_loc, "out_loc");
  check_same_device(kv, kvcache, "kvcache");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == kMainHeadDim && kv.stride(1) == 1, "kv must be [B, 512]");
  TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bfloat16");
  TORCH_CHECK(
      kv_weight.dim() == 1 && kv_weight.size(0) == kMainHeadDim && kv_weight.scalar_type() == at::kBFloat16 &&
          kv_weight.is_contiguous(),
      "kv_weight must be contiguous [512] bfloat16");
  TORCH_CHECK(
      freqs_cis.dim() == 2 && freqs_cis.size(1) == kMainRopeDim && freqs_cis.scalar_type() == at::kFloat &&
          freqs_cis.is_contiguous(),
      "freqs_cis must be contiguous [max_pos, 64] float32");
  TORCH_CHECK(
      positions.dim() == 1 && positions.size(0) == kv.size(0) && positions.is_contiguous(),
      "positions must be contiguous [B]");
  TORCH_CHECK(
      positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong,
      "positions must be int32 or int64");
  TORCH_CHECK(
      out_loc.dim() == 1 && out_loc.size(0) == kv.size(0) && out_loc.scalar_type() == at::kInt &&
          out_loc.is_contiguous(),
      "out_loc must be contiguous [B] int32");
  TORCH_CHECK(
      kvcache.dim() == 2 && kvcache.scalar_type() == at::kByte && kvcache.is_contiguous() &&
          kvcache.size(1) == kFlashMLAPageBytes,
      "kvcache must be contiguous [num_pages, 74880] uint8");
  TORCH_CHECK(page_size == kFlashMLAPageSize, "DSV4 main K CUDA kernel requires page_size=128");
  if (kv.size(0) == 0) return;

  c10::cuda::CUDAGuard guard(kv.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const FusedKNormRopeFlashMLAParams params{
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kv_weight.data_ptr()),
      freqs_cis.data_ptr<float>(),
      positions.data_ptr(),
      out_loc.data_ptr<int32_t>(),
      kvcache.data_ptr<uint8_t>(),
      kv.stride(0),
      static_cast<uint32_t>(kv.size(0)),
      static_cast<float>(eps),
  };
  const bool use_pdl = device_supports_pdl();
  if (positions.scalar_type() == at::kInt) {
    use_pdl ? launch_k_norm<int32_t, true>(params, stream) : launch_k_norm<int32_t, false>(params, stream);
  } else {
    use_pdl ? launch_k_norm<int64_t, true>(params, stream) : launch_k_norm<int64_t, false>(params, stream);
  }
}

void fused_q_indexer_rope_hadamard_quant_cuda(
    const at::Tensor& q_input,
    const at::Tensor& q_fp8,
    const at::Tensor& weight,
    const at::Tensor& weights_out,
    double weight_scale,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions) {
  check_cuda_tensor(q_input, "q_input");
  check_same_device(q_input, q_fp8, "q_fp8");
  check_same_device(q_input, weight, "weight");
  check_same_device(q_input, weights_out, "weights_out");
  check_same_device(q_input, freqs_cis, "freqs_cis");
  check_same_device(q_input, positions, "positions");
  TORCH_CHECK(
      q_input.dim() == 3 && q_input.size(2) == kIndexerHeadDim && q_input.scalar_type() == at::kBFloat16 &&
          q_input.is_contiguous(),
      "q_input must be contiguous [B, H, 128] bfloat16");
  TORCH_CHECK(
      q_fp8.sizes() == q_input.sizes() && q_fp8.scalar_type() == at::kFloat8_e4m3fn && q_fp8.is_contiguous(),
      "q_fp8 must be contiguous [B, H, 128] float8_e4m3fn");
  TORCH_CHECK(
      weight.dim() == 2 && weight.size(0) == q_input.size(0) && weight.size(1) == q_input.size(1) &&
          weight.scalar_type() == at::kBFloat16 && weight.is_contiguous(),
      "weight must be contiguous [B, H] bfloat16");
  TORCH_CHECK(
      weights_out.dim() == 3 && weights_out.size(0) == q_input.size(0) &&
          weights_out.size(1) == q_input.size(1) && weights_out.size(2) == 1 &&
          weights_out.scalar_type() == at::kFloat && weights_out.is_contiguous(),
      "weights_out must be contiguous [B, H, 1] float32");
  TORCH_CHECK(
      freqs_cis.dim() == 2 && freqs_cis.size(1) == kIndexerRopeDim && freqs_cis.scalar_type() == at::kFloat &&
          freqs_cis.is_contiguous(),
      "freqs_cis must be contiguous [max_pos, 64] float32");
  TORCH_CHECK(
      positions.dim() == 1 && positions.size(0) == q_input.size(0) && positions.is_contiguous(),
      "positions must be contiguous [B]");
  TORCH_CHECK(
      positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong,
      "positions must be int32 or int64");
  if (q_input.size(0) == 0) return;

  c10::cuda::CUDAGuard guard(q_input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const FusedQIndexerParams params{
      reinterpret_cast<const __nv_bfloat16*>(q_input.data_ptr()),
      reinterpret_cast<uint8_t*>(q_fp8.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      weights_out.data_ptr<float>(),
      static_cast<float>(weight_scale),
      freqs_cis.data_ptr<float>(),
      positions.data_ptr(),
      static_cast<uint32_t>(q_input.size(0)),
      static_cast<uint32_t>(q_input.size(1)),
  };
  const bool use_pdl = device_supports_pdl();
  if (positions.scalar_type() == at::kInt) {
    use_pdl ? launch_indexer<int32_t, true>(params, stream) : launch_indexer<int32_t, false>(params, stream);
  } else {
    use_pdl ? launch_indexer<int64_t, true>(params, stream) : launch_indexer<int64_t, false>(params, stream);
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_q_norm_rope", &fused_q_norm_rope_cuda, "DSV4 fused main-Q RMSNorm + RoPE");
  module.def(
      "fused_k_norm_rope_flashmla",
      &fused_k_norm_rope_flashmla_cuda,
      "DSV4 fused main-K RMSNorm + RoPE + FlashMLA cache write");
  module.def(
      "fused_q_indexer_rope_hadamard_quant",
      &fused_q_indexer_rope_hadamard_quant_cuda,
      "DSV4 fused indexer-Q RoPE + Hadamard + FP8 quant");
}
