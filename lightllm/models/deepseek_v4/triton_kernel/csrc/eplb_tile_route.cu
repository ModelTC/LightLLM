#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDADeviceAssertion.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {
constexpr int kExperts = 256;
constexpr int kRanks = 8;
constexpr int kAlign = 128;

__global__ void count_kernel(const int64_t* ids, int n, int* hist, int* ordinal) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    const int64_t expert = ids[index];
    if (expert < 0 || expert >= kExperts) {
        ordinal[index] = -1;
        return;
    }
    ordinal[index] = atomicAdd(hist + expert, 1);
}

__device__ int solve_serial(
    const int* q,
    const int* masks,
    int* flow,
    int* load,
    int limit,
    int* augmentations) {
    int flex_ids[16];
    int flex_count = 0;
    int iteration_limit = 0;
    for (int expert = 0; expert < kExperts; ++expert) {
        const int replicas = __popc(masks[expert]);
        if (replicas > 1) {
            if (flex_count >= 16) return 6;
            flex_ids[flex_count++] = expert;
        }
        iteration_limit += q[expert];
    }
    // Singletons remain fixed: residual edges only represent flexible experts.
    for (int expert = 0; expert < kExperts; ++expert) {
        if (__popc(masks[expert]) != 1) continue;
        const int rank = __ffs(masks[expert]) - 1;
        if (load[rank] + q[expert] > limit) return 2;
        flow[expert * kRanks + rank] = q[expert];
        load[rank] += q[expert];
    }
    for (int flex_index = 0; flex_index < flex_count; ++flex_index) {
        const int expert = flex_ids[flex_index];
        int remaining = q[expert];
        while (remaining > 0) {
            int parent_rank[kRanks];
            int parent_expert[kRanks];
            int queue[kRanks];
            int head = 0;
            int tail = 0;
            for (int rank = 0; rank < kRanks; ++rank) {
                parent_rank[rank] = -2;
                parent_expert[rank] = -1;
            }
            for (int rank = 0; rank < kRanks; ++rank) {
                if (masks[expert] & (1 << rank)) {
                    parent_rank[rank] = -1;
                    queue[tail++] = rank;
                }
            }
            int sink = -1;
            while (head < tail && sink < 0) {
                const int from = queue[head++];
                if (load[from] < limit) {
                    sink = from;
                    break;
                }
                for (int prior_index = 0; prior_index < flex_index; ++prior_index) {
                    const int moved_expert = flex_ids[prior_index];
                    if (flow[moved_expert * kRanks + from] == 0) continue;
                    for (int to = 0; to < kRanks; ++to) {
                        if ((masks[moved_expert] & (1 << to)) && parent_rank[to] == -2) {
                            parent_rank[to] = from;
                            parent_expert[to] = moved_expert;
                            queue[tail++] = to;
                        }
                    }
                }
            }
            if (sink < 0) return 3;
            int delta = min(remaining, limit - load[sink]);
            for (int at = sink; parent_rank[at] != -1; at = parent_rank[at]) {
                const int from = parent_rank[at];
                const int moved_expert = parent_expert[at];
                delta = min(delta, flow[moved_expert * kRanks + from]);
            }
            if (delta <= 0) return 4;
            int root = sink;
            while (parent_rank[root] != -1) root = parent_rank[root];
            flow[expert * kRanks + root] += delta;
            for (int at = sink; parent_rank[at] != -1; at = parent_rank[at]) {
                const int from = parent_rank[at];
                const int moved_expert = parent_expert[at];
                flow[moved_expert * kRanks + from] -= delta;
                flow[moved_expert * kRanks + at] += delta;
            }
            load[sink] += delta;
            remaining -= delta;
            ++*augmentations;
            if (*augmentations > iteration_limit) return 7;
        }
    }
    for (int expert = 0; expert < kExperts; ++expert) {
        int assigned = 0;
        for (int rank = 0; rank < kRanks; ++rank) assigned += flow[expert * kRanks + rank];
        if (assigned != q[expert]) return 5;
    }
    return 0;
}

__global__ void solve_kernel(
    const int* counts,
    const int* logical_to_physical,
    const int* replica_counts,
    int map_slots,
    int physical_slots_per_rank,
    int* quota,
    int* status,
    int* max_tiles,
    int* augmentations) {
    __shared__ int q[kExperts];
    __shared__ int masks[kExperts];
    __shared__ int zeta[kExperts];
    __shared__ int flow[kExperts * kRanks];
    __shared__ int load[kRanks];
    __shared__ int bad;
    const int expert = threadIdx.x;

    if (expert == 0) {
        bad = 0;
        *status = 0;
        *max_tiles = 0;
        *augmentations = 0;
    }
    __syncthreads();
    int total = 0;
    for (int rank = 0; rank < kRanks; ++rank) total += counts[rank * kExperts + expert];
    q[expert] = (total + kAlign - 1) / kAlign;
    const int replicas = replica_counts[expert];
    int mask = 0;
    bool valid = replicas >= 1 && replicas <= map_slots;
    for (int slot = 0; slot < replicas && valid; ++slot) {
        const int physical = logical_to_physical[expert * map_slots + slot];
        if (physical < 0 || physical >= kRanks * physical_slots_per_rank) {
            valid = false;
            break;
        }
        const int rank = physical / physical_slots_per_rank;
        if (mask & (1 << rank)) {
            valid = false;
            break;
        }
        mask |= 1 << rank;
    }
    masks[expert] = mask;
    if (!valid || (q[expert] > 0 && mask == 0)) atomicExch(&bad, 1);
    zeta[expert] = 0;
    for (int rank = 0; rank < kRanks; ++rank) {
        flow[expert * kRanks + rank] = 0;
    }
    if (expert < kRanks) load[expert] = 0;
    __syncthreads();

    if (q[expert] != 0) atomicAdd(&zeta[masks[expert]], q[expert]);
    __syncthreads();
    for (int bit = 0; bit < kRanks; ++bit) {
        if (expert & (1 << bit)) zeta[expert] += zeta[expert ^ (1 << bit)];
        __syncthreads();
    }
    if (expert != 0) {
        const int bound = (zeta[expert] + __popc(expert) - 1) / __popc(expert);
        atomicMax(max_tiles, bound);
    }
    __syncthreads();

    if (expert == 0) {
        if (bad) {
            *status = 1;
            *max_tiles = 0;
        } else {
            *status = solve_serial(q, masks, flow, load, *max_tiles, augmentations);
        }
    }
    __syncthreads();
    // Flat indexing gives adjacent lanes adjacent quota elements. On failure,
    // every output remains defined even though callers must reject the status.
    for (int index = expert; index < kExperts * kRanks; index += blockDim.x) {
        quota[index] = *status == 0 ? flow[index] : 0;
    }
}

__global__ void remap_kernel(
    const int64_t* ids,
    const int* ordinal,
    int n,
    const int* counts,
    const int* quota,
    const int* logical_to_physical,
    const int* replica_counts,
    const int* status,
    int map_slots,
    int physical_slots_per_rank,
    int source_rank,
    int64_t* output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (status[0] != 0) {
        CUDA_KERNEL_ASSERT(false);
        return;
    }
    if (index >= n) return;
    const int64_t expert64 = ids[index];
    CUDA_KERNEL_ASSERT(expert64 >= 0 && expert64 < kExperts && ordinal[index] >= 0);
    if (expert64 < 0 || expert64 >= kExperts || ordinal[index] < 0) return;
    const int expert = static_cast<int>(expert64);
    int global_ordinal = ordinal[index];
    for (int rank = 0; rank < source_rank; ++rank) global_ordinal += counts[rank * kExperts + expert];
    const int tile = global_ordinal / kAlign;
    int accumulated = 0;
    int chosen_rank = -1;
    for (int rank = 0; rank < kRanks; ++rank) {
        accumulated += quota[expert * kRanks + rank];
        if (tile < accumulated) {
            chosen_rank = rank;
            break;
        }
    }
    CUDA_KERNEL_ASSERT(chosen_rank >= 0);
    if (chosen_rank < 0) return;
    const int replicas = replica_counts[expert];
    for (int slot = 0; slot < replicas; ++slot) {
        const int physical = logical_to_physical[expert * map_slots + slot];
        if (physical >= 0 && physical / physical_slots_per_rank == chosen_rank) {
            output[index] = physical;
            return;
        }
    }
    CUDA_KERNEL_ASSERT(false);
}

void check_cuda_tensor(const torch::Tensor& tensor, torch::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
}
}  // namespace

std::vector<torch::Tensor> count(torch::Tensor ids) {
    check_cuda_tensor(ids, torch::kInt64, "logical_ids");
    at::cuda::CUDAGuard guard(ids.device());
    auto histogram = torch::zeros({kExperts}, ids.options().dtype(torch::kInt));
    auto ordinal = torch::empty(ids.sizes(), ids.options().dtype(torch::kInt));
    const int n = ids.numel();
    if (n > 0) {
        count_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            ids.data_ptr<int64_t>(), n, histogram.data_ptr<int>(), ordinal.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {histogram, ordinal};
}

std::vector<torch::Tensor> solve(
    torch::Tensor counts,
    torch::Tensor logical_to_physical,
    torch::Tensor replica_counts,
    int64_t map_slots,
    int64_t physical_slots_per_rank) {
    check_cuda_tensor(counts, torch::kInt, "counts");
    check_cuda_tensor(logical_to_physical, torch::kInt, "logical_to_physical");
    check_cuda_tensor(replica_counts, torch::kInt, "replica_counts");
    TORCH_CHECK(counts.sizes() == torch::IntArrayRef({kRanks, kExperts}), "counts must be [8, 256]");
    TORCH_CHECK(logical_to_physical.dim() == 2 && logical_to_physical.size(0) == kExperts,
                "logical_to_physical must be [256, map_slots]");
    TORCH_CHECK(replica_counts.sizes() == torch::IntArrayRef({kExperts}), "replica_counts must be [256]");
    TORCH_CHECK(map_slots > 0 && logical_to_physical.size(1) == map_slots,
                "map_slots must match map width");
    TORCH_CHECK(physical_slots_per_rank > 0, "physical_slots_per_rank must be positive");
    TORCH_CHECK(counts.device() == logical_to_physical.device() && counts.device() == replica_counts.device(),
                "solver tensors must share one CUDA device");
    at::cuda::CUDAGuard guard(counts.device());
    auto quota = torch::empty({kExperts, kRanks}, counts.options());
    auto status = torch::empty({1}, counts.options());
    auto max_tiles = torch::empty({1}, counts.options());
    auto augmentations = torch::empty({1}, counts.options());
    solve_kernel<<<1, kExperts, 0, at::cuda::getCurrentCUDAStream()>>>(
        counts.data_ptr<int>(), logical_to_physical.data_ptr<int>(), replica_counts.data_ptr<int>(),
        static_cast<int>(map_slots), static_cast<int>(physical_slots_per_rank), quota.data_ptr<int>(),
        status.data_ptr<int>(),
        max_tiles.data_ptr<int>(), augmentations.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {quota, status, max_tiles, augmentations};
}

torch::Tensor remap(
    torch::Tensor ids,
    torch::Tensor ordinal,
    torch::Tensor counts,
    torch::Tensor quota,
    torch::Tensor status,
    torch::Tensor output,
    torch::Tensor logical_to_physical,
    torch::Tensor replica_counts,
    int64_t map_slots,
    int64_t physical_slots_per_rank,
    int64_t source_rank) {
    check_cuda_tensor(ids, torch::kInt64, "logical_ids");
    check_cuda_tensor(ordinal, torch::kInt, "ordinal");
    check_cuda_tensor(counts, torch::kInt, "counts");
    check_cuda_tensor(quota, torch::kInt, "quota");
    check_cuda_tensor(status, torch::kInt, "status");
    check_cuda_tensor(output, torch::kInt64, "physical_output");
    check_cuda_tensor(logical_to_physical, torch::kInt, "logical_to_physical");
    check_cuda_tensor(replica_counts, torch::kInt, "replica_counts");
    TORCH_CHECK(ordinal.sizes() == ids.sizes() && output.sizes() == ids.sizes(), "ordinal/output shape must match logical_ids");
    TORCH_CHECK(counts.sizes() == torch::IntArrayRef({kRanks, kExperts}), "counts must be [8, 256]");
    TORCH_CHECK(quota.sizes() == torch::IntArrayRef({kExperts, kRanks}), "quota must be [256, 8]");
    TORCH_CHECK(status.numel() == 1, "status must be scalar");
    TORCH_CHECK(logical_to_physical.dim() == 2 && logical_to_physical.size(0) == kExperts,
                "logical_to_physical must be [256, map_slots]");
    TORCH_CHECK(replica_counts.sizes() == torch::IntArrayRef({kExperts}), "replica_counts must be [256]");
    TORCH_CHECK(map_slots > 0 && logical_to_physical.size(1) == map_slots,
                "map_slots must match map width");
    TORCH_CHECK(physical_slots_per_rank > 0, "physical_slots_per_rank must be positive");
    TORCH_CHECK(source_rank >= 0 && source_rank < kRanks, "source_rank must be in [0, 7]");
    TORCH_CHECK(ids.device() == ordinal.device() && ids.device() == counts.device() &&
                    ids.device() == quota.device() && ids.device() == status.device() && ids.device() == output.device() && ids.device() == logical_to_physical.device() &&
                    ids.device() == replica_counts.device(),
                "remap tensors must share one CUDA device");
    at::cuda::CUDAGuard guard(ids.device());
    const int n = ids.numel();
    remap_kernel<<<std::max(1, (n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            ids.data_ptr<int64_t>(), ordinal.data_ptr<int>(), n, counts.data_ptr<int>(), quota.data_ptr<int>(),
            logical_to_physical.data_ptr<int>(), replica_counts.data_ptr<int>(), status.data_ptr<int>(), static_cast<int>(map_slots),
            static_cast<int>(physical_slots_per_rank), static_cast<int>(source_rank), output.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count", &count);
    m.def("solve", &solve);
    m.def("remap", &remap);
}
