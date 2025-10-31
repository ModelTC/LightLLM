#pragma once

#include "mycuda.cuh"
#include <map>                   // For std::map
#include <string>                // For std::string
#include <sstream>               // For std::stringstream
#include <stdexcept> // For std::runtime_error

namespace cache::transport {

torch::Tensor move_tensor_host_to_device(
    const torch::Tensor& t, cudaStream_t stream, const int32_t device_index
) {
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA not available");
    }
    if (!t.device().is_cpu()) {
        throw std::runtime_error("Input must be on CPU");
    }

    auto options = t.options().device(torch::Device(torch::kCUDA, device_index));
    torch::Tensor gpu_tensor = torch::empty_like(t, options);

    // Perform async H2D copy
    cudaMemcpyAsync(
        gpu_tensor.data_ptr(),
        t.data_ptr(),
        t.nbytes(),
        cudaMemcpyHostToDevice,
        stream
    );

    return gpu_tensor;
}

/**
 * @brief Asynchronously moves a torch::Tensor from a specified CUDA device (GPU) to host (CPU) memory using a specific CUDA stream.
 *
 * @param t The input tensor, which must be defined and located on the specified CUDA device.
 * @param stream The CUDA stream (cudaStream_t) to use for the memory transfer operation. This stream should be associated with device_index.
 * @param device_index The index (0-based) of the CUDA device where tensor t currently resides.
 * @return torch::Tensor A new tensor residing on the CPU. 
 * The copy operation is enqueued on the provided stream and might not be complete when the function returns.
 * @throws std::runtime_error if CUDA is not available, the input tensor is not defined, 
 * the input tensor is not on a CUDA device, or the input tensor's device does not match the provided device_index.
 */
torch::Tensor move_tensor_device_to_host(
    const torch::Tensor &t,
    cudaStream_t stream,
    const int32_t device_index
) {
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is not available.");
    }
    if (!t.is_cuda()) {
        throw std::runtime_error("Input tensor is not on CUDA.");
    }
    if (t.device().index() != device_index) {
        std::stringstream ss;
        ss << "Tensor is on CUDA device " << t.device().index()
           << " but provided device_index is " << device_index;
        throw std::runtime_error(ss.str());
    }

    auto options = t.options().device(torch::kCPU);
    torch::Tensor cpu_tensor = torch::empty_like(t, options);

    cudaError_t err = cudaMemcpyAsync(
        cpu_tensor.data_ptr(),
        t.data_ptr(),
        t.nbytes(),
        cudaMemcpyDeviceToHost,
        stream
    );

    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "cudaMemcpyAsync (DeviceToHost) failed: " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }

    return cpu_tensor;
}

/**
 * @brief Block buffer pool management class template.
 * @tparam BlockSize Size of each block buffer to allocate (in bytes).
 *
 * Provides simple interfaces for allocating and releasing fixed-size memory buffers,
 * associated with a string hash (presumably a task or block hash). Manages a pool
 * with a fixed capacity based on the number of blocks.
 * @note This class is basic and might need enhancements for production use
 * (e.g., more robust error handling, better memory management strategy, thread safety).
 * Currently, operations are not inherently thread-safe if accessed concurrently.
 */
template<int64_t BlockSize>
class TransportBuffer {
public:
    /**
     * @brief Constructor.
     * @param size Maximum number of blocks (buffers of BlockSize) the pool can allocate.
     */
    explicit TransportBuffer(int64_t size): size_(size) {}

    // Prevent copy and move operations to avoid issues with raw pointer ownership.
    TransportBuffer(const TransportBuffer &) = delete;
    TransportBuffer &operator=(const TransportBuffer &) = delete;
    TransportBuffer(TransportBuffer &&) = delete;
    TransportBuffer &operator=(TransportBuffer &&) = delete;

    /**
     * @brief Destructor. Releases all allocated buffers.
     */
    ~TransportBuffer() {
        for (auto const& [hash, buf] : buffer_) {
            delete[] buf; // Use delete[] for arrays allocated with new[]
        }
        buffer_.clear(); // Clear the map
    }

    /**
     * @brief Allocates a buffer associated with a given hash.
     * @param hash The string hash to associate with the allocated buffer. Typically a task or block hash.
     * @return char* Pointer to the newly allocated buffer of BlockSize bytes on success.
     * Returns nullptr if the pool is full. if a buffer
     * already exists for the given hash, or if memory allocation fails.
     * @note The caller does NOT own the returned buffer; it is managed by the TransportBuffer.
     * Call release() with the same hash to free the buffer.
     * This function is NOT thread-safe. Concurrent calls may lead to race conditions.
     * Potential exceptions: std::bad_alloc if `new char[BlockSize]` fails.
     */
    char* allocate(const std::string &hash) {
        // Check capacity and uniqueness
        if (buffer_.size() >= size_) {
            // Log error: Pool full
            return nullptr;
        }
        if (buffer_.count(hash)) { // Use count for checking existence efficiently
            // Log error: Buffer already exists for this hash
            return nullptr;
        }

        char *buf = nullptr;
        try {
            buf = new char[BlockSize]; // Allocate buffer
        } catch (const std::bad_alloc&) {
            // Log error: Memory allocation failed
            return nullptr; // Allocation failed
        }

        buffer_[hash] = buf; // Store the buffer pointer
        return buf;
    }


    /**
     * @brief Releases the buffer associated with the given hash.
     * @param hash The hash of the buffer to release.
     * @note If no buffer is found associated with the hash, the function does nothing.
     * This function is NOT thread-safe. Concurrent calls with allocate() or other
     * release() calls could lead to issues.
     */
    void release(const std::string &hash) {
        auto iter = buffer_.find(hash);
        if (iter == buffer_.end()) {
            // Log warning/info: Attempted to release non-existent buffer
            return; // Not found, nothing to do
        }

        delete[] iter->second; // Free the allocated memory (use delete[] for arrays)
        buffer_.erase(iter);   // Remove the entry from the map
        return;
    }

     /**
     * @brief Gets the maximum capacity of the buffer pool.
     * @return int64_t The maximum number of buffers that can be allocated.
     */
    int64_t getCapacity() const {
        return size_;
    }


private:
    ///< Maximum number of blocks (buffers) the pool can hold.
    int64_t size_;
    ///< Map storing pointers to allocated buffers, keyed by hash. Owns the buffer memory.
    std::map<std::string, char*> buffer_;
};

}; // end of namespace.