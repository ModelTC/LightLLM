// Copyright 2026 LightLLM Team
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4 EPLB top-k. The warp-per-row selection is specialized to the
// fp32, 256-expert router. Hash-routed text uses tid2eid; vision tokens use
// the same bias_vl route selection as the non-EPLB CUDA router.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kExperts = 256;
constexpr int kWarps = 4;
constexpr unsigned kMask = 0xffffffffu;

__device__ __forceinline__ float score(float x) {
  return sqrtf(fmaxf(x, 0.f) + log1pf(expf(-fabsf(x))));
}

__device__ __forceinline__ bool better(float value, int id, float other_value, int other_id) {
  return value > other_value || (value == other_value && id < other_id);
}

template <bool IsHash, bool RecordLoad, bool ReturnLogical, bool SingleToken>
__global__ void moe_topk_eplb_kernel(
    const float* __restrict__ logits, const float* __restrict__ bias,
    const int64_t* __restrict__ input_ids, const int64_t* __restrict__ tid2eid,
    const float* __restrict__ bias_vl, int64_t image_token_start,
    float* __restrict__ weights, int64_t* __restrict__ physical_ids,
    int64_t* __restrict__ logical_ids, const int32_t* __restrict__ logical_to_physical,
    const int32_t* __restrict__ replica_count, int64_t* __restrict__ counter,
    int64_t sample_index, int tokens, int topk, int map_slots, int counter_experts,
    float routed_scaling_factor) {
  const int lane = threadIdx.x & 31;
  const int token = blockIdx.x * kWarps + threadIdx.x / 32;
  if (token >= tokens) return;
  const float* row = logits + token * kExperts;
  const bool is_image = bias_vl != nullptr && input_ids[token] >= image_token_start;
  int selected[8];
  float selected_score[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    selected[i] = -1;
    selected_score[i] = 0.f;
  }

  if (IsHash && !is_image) {
    if (lane == 0) {
      const int64_t input = input_ids[token];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if (k >= topk) break;
        const int id = static_cast<int>(tid2eid[input * topk + k]);
        selected[k] = id;
        selected_score[k] = score(row[id]);
      }
    }
  } else {
    const float* selection_bias = is_image ? bias_vl : bias;
    float local_choice[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int id = lane + j * 32;
      local_choice[j] = score(row[id]) + selection_bias[id];
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (k >= topk) break;
      float best = -INFINITY;
      int best_id = kExperts;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int id = lane + j * 32;
        bool used = false;
#pragma unroll
        for (int prev = 0; prev < 8; ++prev) used |= prev < k && selected[prev] == id;
        if (!used && better(local_choice[j], id, best, best_id)) { best = local_choice[j]; best_id = id; }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(kMask, best, offset);
        const int other_id = __shfl_down_sync(kMask, best_id, offset);
        if (lane + offset < 32 && better(other, other_id, best, best_id)) { best = other; best_id = other_id; }
      }
      best_id = __shfl_sync(kMask, best_id, 0);
      selected[k] = best_id;
      selected_score[k] = score(row[best_id]);
    }
  }
  if (lane == 0) {
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) if (k < topk) sum += selected_score[k];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (k >= topk) break;
      const int id = selected[k];
      uint32_t replica = 0;
      if constexpr (!SingleToken) {
        uint32_t value = static_cast<uint32_t>(token) ^ ((static_cast<uint32_t>(id) + 1u) * 0x9E3779B9u);
        value = (value ^ (value >> 16)) * 0x7FEB352Du;
        value = (value ^ (value >> 15)) * 0x846CA68Bu;
        value ^= value >> 16;
        replica = value % static_cast<uint32_t>(replica_count[id]);
      }
      weights[token * topk + k] = selected_score[k] / fmaxf(sum, 1e-20f) * routed_scaling_factor;
      physical_ids[token * topk + k] = logical_to_physical[id * map_slots + replica];
      if constexpr (ReturnLogical) logical_ids[token * topk + k] = id;
      if constexpr (RecordLoad) {
        atomicAdd(reinterpret_cast<unsigned long long*>(counter + sample_index * counter_experts + id), 1ULL);
      }
    }
  }
}

template <bool IsHash, bool RecordLoad, bool ReturnLogical, bool SingleToken>
void launch(const torch::Tensor& logits, const c10::optional<torch::Tensor>& bias,
            const c10::optional<torch::Tensor>& input_ids, const c10::optional<torch::Tensor>& tid2eid,
            const c10::optional<torch::Tensor>& bias_vl, int64_t image_token_start,
            const torch::Tensor& weights, const torch::Tensor& physical_ids, const torch::Tensor& logical_ids,
            const torch::Tensor& logical_to_physical, const torch::Tensor& replica_count,
            const torch::Tensor& counter, int64_t sample_index, float routed_scaling_factor) {
  const int tokens = logits.size(0), topk = weights.size(1);
  moe_topk_eplb_kernel<IsHash, RecordLoad, ReturnLogical, SingleToken>
      <<< (tokens + kWarps - 1) / kWarps, kWarps * 32, 0, at::cuda::getCurrentCUDAStream() >>>(
          logits.data_ptr<float>(), bias.has_value() ? bias->data_ptr<float>() : nullptr,
          input_ids.has_value() ? input_ids->data_ptr<int64_t>() : nullptr,
          tid2eid.has_value() ? tid2eid->data_ptr<int64_t>() : nullptr,
          bias_vl.has_value() ? bias_vl->data_ptr<float>() : nullptr, image_token_start,
          weights.data_ptr<float>(), physical_ids.data_ptr<int64_t>(), logical_ids.data_ptr<int64_t>(),
          logical_to_physical.data_ptr<int32_t>(), replica_count.data_ptr<int32_t>(), counter.data_ptr<int64_t>(),
          sample_index, tokens, topk, logical_to_physical.size(1), counter.size(1), routed_scaling_factor);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool RecordLoad, bool ReturnLogical>
void dispatch(const torch::Tensor& logits, const c10::optional<torch::Tensor>& bias,
              const c10::optional<torch::Tensor>& input_ids, const c10::optional<torch::Tensor>& tid2eid,
              const c10::optional<torch::Tensor>& bias_vl, int64_t image_token_start,
              const torch::Tensor& weights, const torch::Tensor& physical_ids, const torch::Tensor& logical_ids,
              const torch::Tensor& logical_to_physical, const torch::Tensor& replica_count,
              const torch::Tensor& counter, int64_t sample_index, bool is_hash, bool single_token,
              float routed_scaling_factor) {
#define DSV4_EPLB_LAUNCH(HASH, SINGLE) \
  launch<HASH, RecordLoad, ReturnLogical, SINGLE>(logits, bias, input_ids, tid2eid, bias_vl, image_token_start, weights, physical_ids, logical_ids, logical_to_physical, replica_count, counter, sample_index, routed_scaling_factor)
  if (is_hash && single_token) DSV4_EPLB_LAUNCH(true, true);
  else if (is_hash) DSV4_EPLB_LAUNCH(true, false);
  else if (single_token) DSV4_EPLB_LAUNCH(false, true);
  else DSV4_EPLB_LAUNCH(false, false);
#undef DSV4_EPLB_LAUNCH
}

void check_optional(const torch::Tensor& logits, const c10::optional<torch::Tensor>& tensor,
                    const char* name, torch::ScalarType dtype, int64_t dim, int64_t length) {
  if (!tensor.has_value()) return;
  TORCH_CHECK(tensor->is_cuda() && tensor->device() == logits.device() && tensor->is_contiguous() &&
                  tensor->scalar_type() == dtype && tensor->dim() == dim && tensor->size(dim - 1) == length,
              name, " must be contiguous on the logits CUDA device with the expected shape and dtype");
}
}  // namespace

void moe_topk_eplb(torch::Tensor logits, c10::optional<torch::Tensor> bias,
                   c10::optional<torch::Tensor> input_ids, c10::optional<torch::Tensor> tid2eid,
                   c10::optional<torch::Tensor> bias_vl, int64_t image_token_start, torch::Tensor weights,
                   torch::Tensor physical_ids,
                   torch::Tensor logical_ids, torch::Tensor logical_to_physical, torch::Tensor replica_count,
                   torch::Tensor counter, int64_t sample_index, bool record_load, bool return_logical_ids,
                   float routed_scaling_factor) {
  const auto check = [&](const torch::Tensor& tensor, const char* name, torch::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.device() == logits.device(), name, " must share logits device");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
  };
  check(logits, "logits", torch::kFloat);
  check(weights, "weights", torch::kFloat);
  check(physical_ids, "physical_ids", torch::kLong);
  check(logical_ids, "logical_ids", torch::kLong);
  check(logical_to_physical, "logical_to_physical", torch::kInt);
  check(replica_count, "replica_count", torch::kInt);
  check(counter, "counter", torch::kLong);
  TORCH_CHECK(logits.dim() == 2 && logits.size(1) == kExperts, "logits must be [M, 256]");
  const int64_t tokens = logits.size(0), topk = weights.size(1);
  TORCH_CHECK(topk >= 1 && topk <= 8, "topk must be in [1, 8]");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == tokens, "weights must be [M, K]");
  TORCH_CHECK(physical_ids.sizes() == weights.sizes(), "physical_ids must be [M, K]");
  TORCH_CHECK(logical_ids.sizes() == weights.sizes(), "logical_ids must be [M, K]");
  TORCH_CHECK(logical_to_physical.dim() == 2 && logical_to_physical.size(0) == kExperts &&
                  logical_to_physical.size(1) > 0, "logical_to_physical must be [256, map_slots]");
  TORCH_CHECK(replica_count.dim() == 1 && replica_count.size(0) == kExperts, "replica_count must be [256]");
  TORCH_CHECK(counter.dim() == 2 && counter.size(0) > 0 && counter.size(1) == kExperts,
              "counter must be [rows, 256]");
  TORCH_CHECK(sample_index >= 0 && sample_index < counter.size(0), "sample_index outside counter rows");
  const bool is_hash = tid2eid.has_value();
  check_optional(logits, bias, "bias", torch::kFloat, 1, kExperts);
  check_optional(logits, bias_vl, "bias_vl", torch::kFloat, 1, kExperts);
  if (input_ids.has_value()) check_optional(logits, input_ids, "input_ids", torch::kLong, 1, tokens);
  if (is_hash) {
    TORCH_CHECK(input_ids.has_value(), "input_ids is required for hash routing");
    TORCH_CHECK(tid2eid->is_cuda() && tid2eid->device() == logits.device() && tid2eid->is_contiguous() &&
                    tid2eid->scalar_type() == torch::kLong && tid2eid->dim() == 2 && tid2eid->size(1) == topk,
                "tid2eid must be contiguous [vocab, K] int64 on the logits device");
  } else {
    TORCH_CHECK(bias.has_value(), "bias is required for non-hash routing");
  }
  if (bias_vl.has_value()) {
    TORCH_CHECK(input_ids.has_value() && image_token_start > 0,
                "vision routing requires input_ids and image_token_start > 0");
  }
  if (tokens == 0) return;
  c10::cuda::CUDAGuard guard(logits.device());
  const bool single = tokens == 1;
  if (record_load && return_logical_ids) {
    dispatch<true, true>(logits, bias, input_ids, tid2eid, bias_vl, image_token_start, weights, physical_ids,
                         logical_ids, logical_to_physical, replica_count, counter, sample_index, is_hash, single,
                         routed_scaling_factor);
  } else if (record_load) {
    dispatch<true, false>(logits, bias, input_ids, tid2eid, bias_vl, image_token_start, weights, physical_ids,
                          logical_ids, logical_to_physical, replica_count, counter, sample_index, is_hash, single,
                          routed_scaling_factor);
  } else if (return_logical_ids) {
    dispatch<false, true>(logits, bias, input_ids, tid2eid, bias_vl, image_token_start, weights, physical_ids,
                          logical_ids, logical_to_physical, replica_count, counter, sample_index, is_hash, single,
                          routed_scaling_factor);
  } else {
    dispatch<false, false>(logits, bias, input_ids, tid2eid, bias_vl, image_token_start, weights, physical_ids,
                           logical_ids, logical_to_physical, replica_count, counter, sample_index, is_hash, single,
                           routed_scaling_factor);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_topk_eplb", &moe_topk_eplb); }
