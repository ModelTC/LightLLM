/**
 * This is a standalone test for custom allgather.
 * To compile, make sure you have MPI and NCCL installed in your system.
 * export MPI_HOME=xxx
 * nvcc -O2 -arch=native -std=c++17 custom_all_gather_test.cu -o
 * custom_all_gather_test -lnccl -I${MPI_HOME} -lmpi
 *
 * Warning: this C++ test is not designed to be very readable and was used
 * during the rapid prototyping process.
 *
 * To run:
 * mpirun --allow-run-as-root -np 8 ./custom_all_gather_test
 */
#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include <limits>
#include <vector>

#include "cuda_profiler_api.h"
#include "custom_all_gather.cuh"
#include "mpi.h"
#include "nccl.h"

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

__global__ void dummy_kernel() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  for (int i = 0; i < 100; i++) __nanosleep(1000000);  // 100ms
#else
  for (int i = 0; i < 100; i++) {
    long long int start = clock64();
    while (clock64() - start < 150000000);  // approximately 98.4ms on P40
  }
#endif
}

template <typename T>
__global__ void convert_data(const T* data1, const T* data2, double* fdata1,
                             double* fdata2, int size) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    fdata1[idx] = data1[idx];
    fdata2[idx] = data2[idx];
  }
}

__global__ void init_rand(curandState_t* state, int size, int nRanks) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    for (int i = 0; i < nRanks; i++) {
      curand_init(i + 1, idx, 0, &state[idx * nRanks + i]);
    }
  }
}

template <typename T>
void run(int myRank, int nRanks, ncclComm_t& comm, int threads, int block_limit,
         int data_size, bool performance_test) {
  T* result;
  T* result_nccl;
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECK(cudaMalloc(&result, nRanks * data_size * sizeof(T)));
  CUDACHECK(cudaMemset(result, 0, nRanks * data_size * sizeof(T)));

  CUDACHECK(cudaMalloc(&result_nccl, nRanks * data_size * sizeof(T)));
  CUDACHECK(cudaMemset(result_nccl, 0, nRanks * data_size * sizeof(T)));

  cudaIpcMemHandle_t self_data_handle;
  cudaIpcMemHandle_t data_handles[8];
  vllm::Signal* buffer;
  T* self_data_copy;
  /**
   * Allocate IPC buffer
   *
   * The first section is a temporary buffer for storing intermediate allgather
   * results, if a particular algorithm requires it. The second section is for
   * the input to the allgather. The actual API takes the input pointer as an
   * argument (that is, they can and usually should be allocated separately).
   * But since the input pointers and the temporary buffer all require IPC
   * registration, they are allocated and registered together in the test for
   * convenience.
   */
  CUDACHECK(
      cudaMalloc(&buffer, data_size * sizeof(T) + sizeof(vllm::Signal)));
  CUDACHECK(
      cudaMemset(buffer, 0, data_size * sizeof(T) + sizeof(vllm::Signal)));
  CUDACHECK(cudaMalloc(&self_data_copy, data_size * sizeof(T)));
  CUDACHECK(cudaIpcGetMemHandle(&self_data_handle, buffer));

  MPICHECK(MPI_Allgather(&self_data_handle, sizeof(cudaIpcMemHandle_t),
                         MPI_BYTE, data_handles, sizeof(cudaIpcMemHandle_t),
                         MPI_BYTE, MPI_COMM_WORLD));

  void* rank_data;
  size_t rank_data_sz = 16 * 1024 * 1024;
  CUDACHECK(cudaMalloc(&rank_data, rank_data_sz));
  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < nRanks; i++) {
    if (i == myRank)
      ipc_ptrs[i] = buffer;
    else
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptrs[i], data_handles[i],
                                     cudaIpcMemLazyEnablePeerAccess));
  }
  vllm::CustomAllgather fa(ipc_ptrs, rank_data, rank_data_sz, myRank, nRanks);
  auto* self_data =
      reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) +
                           sizeof(vllm::Signal));
  // hack buffer registration
  {
    void* data[8];
    for (int i = 0; i < nRanks; i++) {
      data[i] =
          ((char*)ipc_ptrs[i]) + sizeof(vllm::Signal);
    }
    fa.register_buffer(data);
  }

  double* ground_truth;
  CUDACHECK(cudaMallocHost(&ground_truth, nRanks * data_size * sizeof(double)));
  curandState_t* states;
  CUDACHECK(cudaMalloc(&states, sizeof(curandState_t) * nRanks * data_size));
  init_rand<<<108, 1024, 0, stream>>>(states, data_size, nRanks);
  CUDACHECK(cudaMemcpyAsync(self_data_copy, self_data, data_size * sizeof(T),
                            cudaMemcpyDeviceToDevice, stream));
  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  ncclDataType_t ncclDtype;
  if (std::is_same<T, half>::value) {
    ncclDtype = ncclFloat16;
  } else if (std::is_same<T, nv_bfloat16>::value) {
    ncclDtype = ncclBfloat16;
  } else {
    ncclDtype = ncclFloat;
  }
  double *nccl_result, *my_result;
  CUDACHECK(cudaMallocHost(&nccl_result, nRanks * data_size * sizeof(double)));
  CUDACHECK(cudaMallocHost(&my_result, nRanks * data_size * sizeof(double)));
  if (performance_test) {
    dummy_kernel<<<1, 1, 0, stream>>>();
    constexpr int warmup_iters = 5;
    constexpr int num_iters = 100;
    // warmup
    for (int i = 0; i < warmup_iters; i++) {
      NCCLCHECK(ncclAllGather(self_data_copy, result_nccl, data_size, ncclDtype,
                              comm, stream));
    }
    CUDACHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iters; i++) {
      NCCLCHECK(ncclAllGather(self_data_copy, result_nccl, data_size, ncclDtype,
                              comm, stream));
    }
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    float allgather_ms = 0;
    cudaEventElapsedTime(&allgather_ms, start, stop);

    dummy_kernel<<<1, 1, 0, stream>>>();
    // warm up
    for (int i = 0; i < warmup_iters; i++) {
      fa.allgather<T>(stream, self_data, result, data_size, threads,
                      block_limit);
    }
    CUDACHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iters; i++) {
      fa.allgather<T>(stream, self_data, result, data_size, threads,
                      block_limit);
    }
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    float duration_ms = 0;
    cudaEventElapsedTime(&duration_ms, start, stop);
    if (myRank == 0)
      printf(
          "Rank %d done, nGPUs:%d, sz (kb): %d, %d, %d, my time:%.2fus, nccl "
          "time:%.2fus\n",
          myRank, nRanks, data_size * sizeof(T) / 1024, threads, block_limit,
          duration_ms * 1e3 / num_iters, allgather_ms * 1e3 / num_iters);

    // And wait for all the queued up work to complete
    CUDACHECK(cudaStreamSynchronize(stream));

    NCCLCHECK(ncclAllGather(self_data_copy, result_nccl, data_size, ncclDtype,
                            comm, stream));

    convert_data<T><<<108, 1024, 0, stream>>>(result_nccl, result, nccl_result,
                                              my_result, nRanks * data_size);
    CUDACHECK(cudaStreamSynchronize(stream));

    for (unsigned long j = 0; j < data_size; j++) {
      auto diff = abs(nccl_result[j] - my_result[j]);
      if (diff >= 4e-2) {
        printf("Rank %d: Verification mismatch at %lld: %f != (my) %f, gt=%f\n",
               myRank, j, nccl_result[j], my_result[j], ground_truth[j]);
        break;
      }
    }
    long double avg_diffs = 0.0;
    for (int j = 0; j < nRanks * data_size; j++) {
      avg_diffs += abs(nccl_result[j] - my_result[j]);
    }
    if (myRank == 0)
      std::cout << "average abs diffs: nccl: " << avg_diffs / data_size << std::endl;
      
  } else {
    for (int i = 0; i < 1; i++) {
      fa.allgather<T>(stream, self_data, result, data_size, threads,
                      block_limit);
      CUDACHECK(cudaStreamSynchronize(stream));
      NCCLCHECK(ncclAllGather(self_data_copy, result_nccl, data_size, ncclDtype,
                              comm, stream));
      convert_data<T><<<108, 1024, 0, stream>>>(
          result_nccl, result, nccl_result, my_result, data_size * nRanks);
      CUDACHECK(cudaStreamSynchronize(stream));

      for (unsigned long j = 0; j < data_size * nRanks; j++) {
        auto diff = abs(nccl_result[j] - my_result[j]);
        if (diff >= 4e-2) {
          printf(
              "Rank %d: Verification mismatch at %lld: %f != (my) %f, gt=%f\n",
              myRank, j, nccl_result[j], my_result[j], ground_truth[j]);
          break;
        }
      }
    }
    if (myRank == 0)
      printf("Test passed: nGPUs:%d, sz (kb): %d, %d, %d\n", nRanks,
             data_size * sizeof(T) / 1024, threads, block_limit);
    
    long double avg_diffs = 0.0;
    for (int j = 0; j < data_size * nRanks; j++) {
      avg_diffs += abs(nccl_result[j] - my_result[j]);
    }
    if (myRank == 0)
      std::cout << "average abs diffs: nccl: " << avg_diffs / data_size << std::endl;
  }

  CUDACHECK(cudaFree(result));
  CUDACHECK(cudaFree(self_data_copy));
  CUDACHECK(cudaFree(rank_data));
  CUDACHECK(cudaFree(buffer));
  CUDACHECK(cudaFree(states));
  CUDACHECK(cudaFree(result_nccl));
  CUDACHECK(cudaFreeHost(ground_truth));
  CUDACHECK(cudaFreeHost(nccl_result));
  CUDACHECK(cudaFreeHost(my_result));
  CUDACHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
  int nRanks, myRank;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  CUDACHECK(cudaSetDevice(myRank));
  ncclUniqueId id;
  ncclComm_t comm;
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast(static_cast<void*>(&id), sizeof(id), MPI_BYTE, 0,
                     MPI_COMM_WORLD));
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  bool performance_test = true;
  cudaProfilerStart();
  // Uncomment to scan through different block size configs.
  // for (int threads : {256, 512, 1024}) {
  //   for (int block_limit = 16; block_limit < 112; block_limit += 4) {
  //     run<half>(myRank, nRanks, comm, threads, block_limit, 1024 * 1024,
  //     performance_test);
  //   }
  // }
  // Scan through different sizes to test performance.
  for (int sz = 512; sz <= (8 << 20); sz *= 2) {
    run<half>(myRank, nRanks, comm, 512, 36, (sz + 8 * 47), performance_test);
  }

  cudaProfilerStop();
  MPICHECK(MPI_Finalize());
  return EXIT_SUCCESS;
}
