#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <array>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>
#include "custom_all_reduce.cuh"

// #define CUDACHECK(cmd)                                              \
//   do {                                                              \
//     cudaError_t e = cmd;                                            \
//     if (e != cudaSuccess) {                                         \
//       printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
//              cudaGetErrorString(e));                                \
//       exit(EXIT_FAILURE);                                           \
//     }                                                               \
//   } while (0)

namespace vllm {

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct gather_packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
};

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    custom_all_gather_kernel(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename gather_packed_t<T>::P;
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = tid; idx < size; idx += stride) {
    #pragma unroll
      for (int step = 0; step < ngpus; step ++) {
          int src_rank = (rank - step + ngpus) % ngpus;  // 当前步骤中数据来源的进程
          P* ptr = (P*)_dp->ptrs[src_rank];
          int dst_offset = src_rank * size;         // 数据在 recv_buf 中的存储位置
          // 从 src_rank 的 handle 中读取数据，并存储到 recv_buf
          int dst_idx = dst_offset + idx;
          ((P*)result)[dst_idx] = ptr[idx];
      }
  }
  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);

}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllgather {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  RankSignals sg_;
  // Stores an map from a pointer to its peer pointters from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph capture
  // time. Therefore, during capture, we increment the rank data pointer and use
  // that as the argument to the kernel. The kernel arguments are stored in
  // graph_unreg_buffers_. The actual peer pointers will be filled in at the
  // memory pointed to by the pointers in graph_unreg_buffers_ when
  // the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allgather synchronization, and the second section
   * is for storing the intermediate results required by some allgather algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllgather(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool full_nvlink = true)
      : rank_(rank),
        world_size_(world_size),
        full_nvlink_(full_nvlink),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr,
                                CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the remote
  // possibility of different allocation patterns between ranks. For example,
  // rank 1 may get the same input address for the second allgather, but rank 2
  // got a different address. IPC handles have internal reference counting
  // mechanism so overhead should be small.
  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allgather, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search. Using
   * 36 blocks give the best or close to the best runtime on the devices I
   * tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also only
   * take a small amount of SMs. Not quite sure the underlying reason, but my
   * guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allgather(cudaStream_t stream, T* input, T* output, int size,
                 int threads = 512, int block_limit = 36) {
    auto d = gather_packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allgather currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }
    size /= d;
    // auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
    // TODO(hanzhi713): Threshold is different for A100 and H100.
    // Add per device threshold.
#define REDUCE_CASE(ngpus)                            \
  case ngpus: {                                       \
    KL(ngpus, custom_all_gather_kernel);        \
    break;                                            \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allgather only supports num gpus in (2,4,6,8). Actual num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  ~CustomAllgather() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};
/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and add
 a template instantiation:
 * template void vllm::CustomAllgather::allgather<half>(cudaStream_t, half *,
 half *, int, int, int);
*/
}  // namespace vllm
