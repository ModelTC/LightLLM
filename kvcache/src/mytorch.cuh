#pragma once

#include "mycuda.cuh"

#include <torch/torch.h>

namespace atex {
template <typename T> __host__ inline T *PTR(torch::Tensor t) { return reinterpret_cast<T *>(t.data_ptr()); }

template <> __host__ inline fp16_t *PTR(torch::Tensor t) { return reinterpret_cast<fp16_t *>(t.data_ptr()); }

template <> __host__ inline fp16x2_t *PTR(torch::Tensor t) { return reinterpret_cast<fp16x2_t *>(t.data_ptr()); }

template <> __host__ inline int8x4_t *PTR(torch::Tensor t) { return reinterpret_cast<int8x4_t *>(t.data_ptr()); }

template <> __host__ inline int8x2_t *PTR(torch::Tensor t) { return reinterpret_cast<int8x2_t *>(t.data_ptr()); }

template <> __host__ inline int8_t *PTR(torch::Tensor t) { return reinterpret_cast<int8_t *>(t.data_ptr()); }

template <> __host__ inline char *PTR(torch::Tensor t) { return reinterpret_cast<char *>(t.data_ptr()); }

template <> __host__ inline uint16_t *PTR(torch::Tensor t) { return reinterpret_cast<uint16_t *>(t.data_ptr()); }

template <> __host__ inline uint32_t *PTR(torch::Tensor t) { return reinterpret_cast<uint32_t *>(t.data_ptr()); }

template <> __host__ inline void *PTR(torch::Tensor t) { return reinterpret_cast<void *>(t.data_ptr()); }

} // namespace atex
