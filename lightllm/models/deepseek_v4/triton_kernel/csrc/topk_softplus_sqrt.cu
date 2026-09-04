/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Adapted for LightLLM from vLLM commit
// 36d2a5086bae12d0c0b311607373e4fc2d1036aa:
// csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu.
// LightLLM-local CUDA binding for DeepSeek-V4's fixed top-6 router. The
// vLLM warp kernel is shared by hash and bias routing here; image tokens use
// LightLLM's multimodal-ID contract (all IDs >= image_token_start).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kTopK = 6;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;
constexpr int kMaxExpertsPerLane = 12;

__device__ __forceinline__ float softplus_sqrt(float value) {
  float score = sqrtf(fmaxf(value, 0.0f) + __logf(1.0f + __expf(-fabsf(value))));
  return isnan(score) ? 0.0f : score;
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

template <bool UsePDL>
__launch_bounds__(kThreadsPerBlock) __global__ void topk_softplus_sqrt_kernel(
    const float* logits,
    float* output_weights,
    int64_t* output_indices,
    int num_rows,
    int num_experts,
    float routed_scaling_factor,
    const float* correction_bias,
    const int64_t* input_ids,
    const int64_t* tid2eid,
    const float* bias_vl,
    int64_t image_token_start) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (row >= num_rows) return;

  pdl_wait_primary<UsePDL>();

  const int64_t token_id = input_ids == nullptr ? 0 : input_ids[row];
  const bool is_image = bias_vl != nullptr && token_id >= image_token_start;
  const bool use_hash = tid2eid != nullptr && !is_image;

  int expert = 0;
  float weight = 0.0f;
  if (use_hash) {
    if (lane < kTopK) {
      expert = static_cast<int>(tid2eid[token_id * kTopK + lane]);
      weight = softplus_sqrt(logits[row * num_experts + expert]);
    }
  } else {
    const float* selection_bias = is_image ? bias_vl : correction_bias;
    const int experts_per_lane = num_experts / 32;
    float scores[kMaxExpertsPerLane];
    float selection_scores[kMaxExpertsPerLane];

#pragma unroll
    for (int i = 0; i < kMaxExpertsPerLane; ++i) {
      if (i < experts_per_lane) {
        const int expert_id = lane + 32 * i;
        const float score = softplus_sqrt(logits[row * num_experts + expert_id]);
        scores[i] = score;
        selection_scores[i] = score + selection_bias[expert_id];
      } else {
        scores[i] = 0.0f;
        selection_scores[i] = -INFINITY;
      }
    }

    int selected_experts[kTopK];
    float selected_weights[kTopK];
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      float best_score = -INFINITY;
      int best_expert = num_experts;
#pragma unroll
      for (int i = 0; i < kMaxExpertsPerLane; ++i) {
        const int expert_id = lane + 32 * i;
        if (selection_scores[i] > best_score) {
          best_score = selection_scores[i];
          best_expert = expert_id;
        }
      }
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
        const int other_expert = __shfl_xor_sync(0xffffffffu, best_expert, mask);
        if (other_score > best_score || (other_score == best_score && other_expert < best_expert)) {
          best_score = other_score;
          best_expert = other_expert;
        }
      }

      float selected_weight = 0.0f;
#pragma unroll
      for (int i = 0; i < kMaxExpertsPerLane; ++i) {
        if (lane + 32 * i == best_expert) {
          selected_weight = scores[i];
          selection_scores[i] = -INFINITY;
        }
      }
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        selected_weight += __shfl_xor_sync(0xffffffffu, selected_weight, mask);
      }
      selected_experts[slot] = best_expert;
      selected_weights[slot] = selected_weight;
    }

    if (lane < kTopK) {
      expert = selected_experts[lane];
      weight = selected_weights[lane];
    }
  }

  float weight_sum = weight;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    weight_sum += __shfl_xor_sync(0xffffffffu, weight_sum, mask);
  }

  pdl_trigger_secondary<UsePDL>();

  if (lane < kTopK) {
    const int offset = row * kTopK + lane;
    output_weights[offset] = weight * routed_scaling_factor / (weight_sum > 0.0f ? weight_sum : 1.0f);
    output_indices[offset] = static_cast<int64_t>(expert);
  }
}

template <int NumExperts, bool UsePDL>
__launch_bounds__(kThreadsPerBlock) __global__ void topk_bias_softplus_sqrt_kernel(
    const float* logits,
    float* output_weights,
    int64_t* output_indices,
    int num_rows,
    float routed_scaling_factor,
    const float* correction_bias,
    const int64_t* input_ids,
    const float* bias_vl,
    int64_t image_token_start) {
  constexpr int kValuesPerThread = NumExperts / 32;
  constexpr int kVectorsPerThread = kValuesPerThread / 4;
  const int row = blockIdx.x * kWarpsPerBlock + threadIdx.y;
  const int lane = threadIdx.x;
  if (row >= num_rows) return;

  pdl_wait_primary<UsePDL>();

  const int64_t token_id = input_ids == nullptr ? 0 : input_ids[row];
  const bool is_image = bias_vl != nullptr && token_id >= image_token_start;
  const float* selection_bias = is_image ? bias_vl : correction_bias;
  const float4* row_logits = reinterpret_cast<const float4*>(logits + row * NumExperts);
  float values[kValuesPerThread];

#pragma unroll
  for (int vector = 0; vector < kVectorsPerThread; ++vector) {
    const float4 packed = row_logits[vector * 32 + lane];
    const float packed_values[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int local = vector * 4 + i;
      const int expert_id = vector * 128 + lane * 4 + i;
      values[local] = softplus_sqrt(packed_values[i]) + selection_bias[expert_id];
    }
  }

  float selected_sum = 0.0f;
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    float best_score = values[0];
    int best_expert = lane * 4;
#pragma unroll
    for (int local = 1; local < kValuesPerThread; ++local) {
      const int vector = local / 4;
      const int element = local % 4;
      const int expert_id = vector * 128 + lane * 4 + element;
      if (values[local] > best_score) {
        best_score = values[local];
        best_expert = expert_id;
      }
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
      const int other_expert = __shfl_xor_sync(0xffffffffu, best_expert, mask);
      if (other_score > best_score || (other_score == best_score && other_expert < best_expert)) {
        best_score = other_score;
        best_expert = other_expert;
      }
    }

    if (lane == 0) {
      const float selected_weight = best_score - selection_bias[best_expert];
      const int offset = row * kTopK + slot;
      output_weights[offset] = selected_weight;
      output_indices[offset] = static_cast<int64_t>(best_expert);
      selected_sum += selected_weight;
    }

    if (slot + 1 < kTopK) {
      const int vector = best_expert / 128;
      const int owner_lane = (best_expert / 4) % 32;
      if (lane == owner_lane) {
        values[vector * 4 + best_expert % 4] = -INFINITY;
      }
    }
  }

  if (lane == 0) {
    const float scale = routed_scaling_factor / (selected_sum > 0.0f ? selected_sum : 1.0f);
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      output_weights[row * kTopK + slot] *= scale;
    }
  }

  pdl_trigger_secondary<UsePDL>();
}

template <int NumExperts, bool UsePDL>
void launch_topk_bias_softplus_sqrt(
    const at::Tensor& topk_weights,
    const at::Tensor& topk_indices,
    const at::Tensor& logits,
    double routed_scaling_factor,
    const float* correction_bias,
    const int64_t* input_ids,
    const float* bias_vl,
    int64_t image_token_start,
    cudaStream_t stream) {
  cudaLaunchConfig_t config{};
  config.gridDim = (logits.size(0) + kWarpsPerBlock - 1) / kWarpsPerBlock;
  config.blockDim = dim3(32, kWarpsPerBlock);
  config.stream = stream;

  cudaLaunchAttribute attribute{};
  if constexpr (UsePDL) {
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = true;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }

  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &config,
      topk_bias_softplus_sqrt_kernel<NumExperts, UsePDL>,
      logits.data_ptr<float>(),
      topk_weights.data_ptr<float>(),
      topk_indices.data_ptr<int64_t>(),
      static_cast<int>(logits.size(0)),
      static_cast<float>(routed_scaling_factor),
      correction_bias,
      input_ids,
      bias_vl,
      image_token_start));
}

template <bool UsePDL>
void launch_topk_softplus_sqrt(
    const at::Tensor& topk_weights,
    const at::Tensor& topk_indices,
    const at::Tensor& logits,
    double routed_scaling_factor,
    const float* correction_bias,
    const int64_t* input_ids,
    const int64_t* tid2eid,
    const float* bias_vl,
    int64_t image_token_start,
    cudaStream_t stream) {
  cudaLaunchConfig_t config{};
  config.gridDim = (logits.size(0) + kWarpsPerBlock - 1) / kWarpsPerBlock;
  config.blockDim = kThreadsPerBlock;
  config.stream = stream;

  cudaLaunchAttribute attribute{};
  if constexpr (UsePDL) {
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = true;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }

  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &config,
      topk_softplus_sqrt_kernel<UsePDL>,
      logits.data_ptr<float>(),
      topk_weights.data_ptr<float>(),
      topk_indices.data_ptr<int64_t>(),
      static_cast<int>(logits.size(0)),
      static_cast<int>(logits.size(1)),
      static_cast<float>(routed_scaling_factor),
      correction_bias,
      input_ids,
      tid2eid,
      bias_vl,
      image_token_start));
}

void dispatch_pdl(
    const at::Tensor& topk_weights,
    const at::Tensor& topk_indices,
    const at::Tensor& logits,
    double routed_scaling_factor,
    const float* correction_bias,
    const int64_t* input_ids,
    const int64_t* tid2eid,
    const float* bias_vl,
    int64_t image_token_start,
    cudaStream_t stream) {
  if (tid2eid == nullptr) {
    if (logits.size(1) == 256) {
      if (at::cuda::getCurrentDeviceProperties()->major >= 9) {
        launch_topk_bias_softplus_sqrt<256, true>(
            topk_weights,
            topk_indices,
            logits,
            routed_scaling_factor,
            correction_bias,
            input_ids,
            bias_vl,
            image_token_start,
            stream);
      } else {
        launch_topk_bias_softplus_sqrt<256, false>(
            topk_weights,
            topk_indices,
            logits,
            routed_scaling_factor,
            correction_bias,
            input_ids,
            bias_vl,
            image_token_start,
            stream);
      }
    } else if (at::cuda::getCurrentDeviceProperties()->major >= 9) {
      launch_topk_bias_softplus_sqrt<384, true>(
          topk_weights,
          topk_indices,
          logits,
          routed_scaling_factor,
          correction_bias,
          input_ids,
          bias_vl,
          image_token_start,
          stream);
    } else {
      launch_topk_bias_softplus_sqrt<384, false>(
          topk_weights,
          topk_indices,
          logits,
          routed_scaling_factor,
          correction_bias,
          input_ids,
          bias_vl,
          image_token_start,
          stream);
    }
    return;
  }

  if (at::cuda::getCurrentDeviceProperties()->major >= 9) {
    launch_topk_softplus_sqrt<true>(
        topk_weights,
        topk_indices,
        logits,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
        bias_vl,
        image_token_start,
        stream);
  } else {
    launch_topk_softplus_sqrt<false>(
        topk_weights,
        topk_indices,
        logits,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
        bias_vl,
        image_token_start,
        stream);
  }
}

void check_optional_vector(
    const at::Tensor& logits,
    const c10::optional<at::Tensor>& tensor,
    at::ScalarType dtype,
    int64_t length,
    const char* name) {
  if (!tensor.has_value()) return;
  TORCH_CHECK(
      tensor->is_cuda() && tensor->get_device() == logits.get_device() && tensor->scalar_type() == dtype &&
          tensor->dim() == 1 && tensor->size(0) == length && tensor->is_contiguous(),
      name,
      " must be a contiguous vector on the logits device");
}

}  // namespace

void topk_softplus_sqrt_cuda(
    const at::Tensor& topk_weights,
    const at::Tensor& topk_indices,
    const at::Tensor& logits,
    double routed_scaling_factor,
    const c10::optional<at::Tensor>& correction_bias,
    const c10::optional<at::Tensor>& input_ids,
    const c10::optional<at::Tensor>& tid2eid,
    const c10::optional<at::Tensor>& bias_vl,
    int64_t image_token_start) {
  TORCH_CHECK(
      logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.is_contiguous(),
      "logits must be contiguous [num_tokens, num_experts] float32 CUDA");
  TORCH_CHECK(logits.size(1) == 256 || logits.size(1) == 384, "DSV4 router requires 256 or 384 experts");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.get_device() == logits.get_device() &&
          topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous() &&
          topk_weights.sizes() == at::IntArrayRef({logits.size(0), kTopK}),
      "topk_weights must be contiguous [num_tokens, 6] float32 on the logits device");
  TORCH_CHECK(
      topk_indices.is_cuda() && topk_indices.get_device() == logits.get_device() && topk_indices.is_contiguous() &&
          topk_indices.scalar_type() == at::kLong &&
          topk_indices.sizes() == at::IntArrayRef({logits.size(0), kTopK}),
      "topk_indices must be contiguous [num_tokens, 6] int64 on the logits device");

  check_optional_vector(logits, correction_bias, at::kFloat, logits.size(1), "correction_bias");
  check_optional_vector(logits, bias_vl, at::kFloat, logits.size(1), "bias_vl");
  if (input_ids.has_value()) {
    TORCH_CHECK(
        input_ids->is_cuda() && input_ids->get_device() == logits.get_device() && input_ids->dim() == 1 &&
            input_ids->size(0) == logits.size(0) && input_ids->is_contiguous() && input_ids->scalar_type() == at::kLong,
        "input_ids must be contiguous [num_tokens] int64 on the logits device");
  }
  if (tid2eid.has_value()) {
    TORCH_CHECK(input_ids.has_value(), "input_ids is required for hash routing");
    TORCH_CHECK(
        tid2eid->is_cuda() && tid2eid->get_device() == logits.get_device() && tid2eid->dim() == 2 &&
            tid2eid->size(1) == kTopK && tid2eid->is_contiguous() && tid2eid->scalar_type() == at::kLong,
        "tid2eid must be contiguous [vocab_size, 6] int64 on the logits device");
  } else {
    TORCH_CHECK(correction_bias.has_value(), "correction_bias is required for non-hash routing");
  }
  if (bias_vl.has_value()) {
    TORCH_CHECK(input_ids.has_value() && image_token_start > 0, "vision routing requires input_ids and image_token_start");
  }
  if (logits.size(0) == 0) return;

  c10::cuda::CUDAGuard guard(logits.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const float* correction_bias_ptr =
      correction_bias.has_value() ? correction_bias->data_ptr<float>() : nullptr;
  const float* bias_vl_ptr = bias_vl.has_value() ? bias_vl->data_ptr<float>() : nullptr;
  dispatch_pdl(
      topk_weights,
      topk_indices,
      logits,
      routed_scaling_factor,
      correction_bias_ptr,
      input_ids.has_value() ? input_ids->data_ptr<int64_t>() : nullptr,
      tid2eid.has_value() ? tid2eid->data_ptr<int64_t>() : nullptr,
      bias_vl_ptr,
      image_token_start,
      stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("topk_softplus_sqrt", &topk_softplus_sqrt_cuda, "DeepSeek-V4 fused top-6 sqrt-softplus router");
}
