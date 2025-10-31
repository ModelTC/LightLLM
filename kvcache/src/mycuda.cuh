#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace atex {
// type definitions
using fp16_t = __half;
using fp16x2_t = __half2;
using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;

using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
using fp8x4_e4m3_t = __nv_fp8x4_e4m3;

using fp32_t = float;
using fp32x2_t = float2;
using fp32x4_t = float4;

using int32x4_t = int4;
using int32x2_t = int2;

using int8x2_t = int16_t;
using int8x4_t = int32_t;
using int8x8_t = int64_t;

using vec_type = int4;

// convert fp16_t to fp32_t
__device__ inline fp32_t cvt_f16_f32(const fp16_t x) { return __half2float(x); }

__device__ inline fp16_t cvt_f32_f16(const fp32_t x) { return __float2half(x); }

template <typename T> __host__ __device__ T Cdiv(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename T> __host__ __device__ T Adiv(T value, T alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

__device__ inline fp32x2_t operator+(const fp32x2_t &a, const fp32x2_t &b) { return {a.x + b.x, a.y + b.y}; }

__device__ inline fp16_t abs(const fp16_t &x) { return __habs(x); }

__device__ inline bool operator>(const fp16_t &a, const fp16_t &b) { return __hgt(a, b); }

__device__ inline fp16_t operator+(const fp16_t &a, const fp16_t &b) { return __hadd(a, b); }

__device__ inline fp16_t operator-(const fp16_t &a, const fp16_t &b) { return __hsub(a, b); }

__device__ inline fp16_t operator*(const fp16_t &a, const fp16_t &b) { return __hmul(a, b); }

__device__ inline fp16_t operator/(const fp16_t &a, const fp16_t &b) { return __hdiv(a, b); }

__device__ inline fp16_t &operator+=(fp16_t &a, const fp16_t &b) {
  a = __hadd(a, b);
  return a;
}

__device__ inline fp16_t &operator-=(fp16_t &a, const fp16_t &b) {
  a = __hsub(a, b);
  return a;
}

__device__ inline fp16_t &operator*=(fp16_t &a, const fp16_t &b) {
  a = __hmul(a, b);
  return a;
}

__device__ inline fp16_t &operator/=(fp16_t &a, const fp16_t &b) {
  a = __hdiv(a, b);
  return a;
}

__device__ inline fp16x2_t operator+(const fp16x2_t &a, const fp16x2_t &b) { return __hadd2(a, b); }

template <int VPT> struct BytesToType;

template <> struct BytesToType<2> { using type = uint16_t; };
template <> struct BytesToType<4> { using type = uint32_t; };
template <> struct BytesToType<8> { using type = uint64_t; };
template <> struct BytesToType<16> { using type = float4; };

template <int Bytes> __device__ inline void vec_copy(const void *src, void *dest) {
  using T = typename BytesToType<Bytes>::type;

  const T *in = static_cast<const T *>(src);
  T *out = static_cast<T *>(dest);
  *out = *in;
}

template <int32_t divisor> __device__ inline int32x2_t divmod(int32_t x);

template <> __device__ inline int32x2_t divmod<128>(int32_t x) { return {x >> 7, x & 0x7F}; }

template <> __device__ inline int32x2_t divmod<64>(int32_t x) { return {x >> 6, x & 0x3F}; }

template <> __device__ inline int32x2_t divmod<32>(int32_t x) { return {x >> 5, x & 0x1F}; }

template <> __device__ inline int32x2_t divmod<16>(int32_t x) { return {x >> 4, x & 0x0F}; }

template <> __device__ inline int32x2_t divmod<8>(int32_t x) { return {x >> 3, x & 0x07}; }

template <> __device__ inline int32x2_t divmod<4>(int32_t x) { return {x >> 2, x & 0x03}; }

template <> __device__ inline int32x2_t divmod<2>(int32_t x) { return {x >> 1, x & 0x01}; }

} // namespace atex
