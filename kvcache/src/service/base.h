#pragma once

#include <torch/extension.h>

#include "config.h"
#include "core/error.h"
#include "core/queue.h"
#include "core/task.h"
#include "storage/local.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

int64_t get_env_int64(const char *name) {
  if (!name) {
    return 0;
  }
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return 0;
  }
  errno = 0;
  char *end = nullptr;
  long long parsed = std::strtoll(value, &end, 10);
  if (end == value || errno == ERANGE) {
    return 0;
  }
  return static_cast<int64_t>(parsed);
}

int64_t resolve_max_block_size_bytes() {
  const int64_t env_override = get_env_int64(ACS_MaxBlockSizeEnvVar);
  if (env_override > 0) {
    return env_override;
  }
  if (env_override == 0) {
    return ACS_DefaultMaxBlockSizeBytes;
  }
  return std::numeric_limits<int64_t>::max();
}

} // namespace

namespace cache::service {

struct alignas(8) CacheParam_t {
  char *base_ptr;
  int64_t page_size;
  int64_t num_of_layer;
  int64_t layer_stride;
  int64_t page_stride;
  int64_t num_of_page;

  CacheParam_t()
      : base_ptr(nullptr), page_size(0), num_of_layer(0), layer_stride(0), page_stride(0), num_of_page(0) {}
  CacheParam_t(char *ptr, int64_t page, int64_t layer, int64_t layer_strd, int64_t page_strd, int64_t pages)
      : base_ptr(ptr), page_size(page), num_of_layer(layer), layer_stride(layer_strd), page_stride(page_strd),
        num_of_page(pages) {}
};

// Generic Cache Service Base Class
class CacheService {
public:
  // Delete the default constructor
  CacheService() = delete;

  /**
   * Constructor
   * @param kvcache Reference to the CPU kv-cache tensor backing this service
   */
  explicit CacheService(const torch::Tensor &kvcache)
      : queue_(std::make_unique<cache::queue::TaskQueue>()), cache_info_(), block_size_(0) {

    if (kvcache.dim() != 3) {
      throw std::runtime_error(
          "The input kv-cache tensor has an incorrect dimension; it must be a 3D tensor of shape "
          "[num page, num layer, page size].");
    }

    if (kvcache.is_cuda()) {
      throw std::runtime_error("GPU kv-cache tensors are not supported. Please provide a CPU tensor.");
    }

    if (!kvcache.is_contiguous()) {
      throw std::runtime_error("The kv-cache tensor must be stored contiguously.");
    }

    const int64_t element_size = kvcache.element_size();
    const auto sizes = kvcache.sizes();
    const auto strides = kvcache.strides();

    int64_t num_layers = 0;
    int64_t num_pages = 0;
    int64_t layer_stride = 0;
    int64_t page_stride = 0;

    num_layers = sizes[1];
    num_pages = sizes[0];
    if (strides[2] != 1 || strides[1] != sizes[2]) {
      throw std::runtime_error("Page-major kv-cache tensor must have contiguous layer dimension.");
    }
    if (strides[0] != sizes[1] * sizes[2]) {
      throw std::runtime_error("Page-major kv-cache tensor must be contiguous along the page dimension.");
    }

    layer_stride = strides[1] * element_size;
    page_stride = strides[0] * element_size;

    const int64_t inferred_page_size = sizes[2] * element_size;
    cache_info_ = CacheParam_t((char *)kvcache.data_ptr(), inferred_page_size, num_layers, layer_stride,
                               page_stride, num_pages);

    const int64_t page_bytes = cache_info_.page_size * cache_info_.num_of_layer;
    const int64_t block_limit = resolve_max_block_size_bytes();

    int64_t pages_per_block = std::max<int64_t>(1, ACS_TokensPerBlock);
    if (block_limit > 0 && block_limit < std::numeric_limits<int64_t>::max()) {
      const int64_t max_pages_by_size = std::max<int64_t>(1, block_limit / std::max<int64_t>(page_bytes, int64_t(1)));
      pages_per_block = std::max<int64_t>(1, std::min<int64_t>(pages_per_block, max_pages_by_size));
    }
    block_size_ = page_bytes * pages_per_block;

    printf("CacheService created with following cache info: \n");
    printf("\tNum of layer: %lld \n", static_cast<long long>(cache_info_.num_of_layer));
    printf("\tNum of page: %lld \n", static_cast<long long>(cache_info_.num_of_page));
    printf("\tPage Size: %lld \n", static_cast<long long>(cache_info_.page_size));
    printf("\tLayer Stride: %lld \n", static_cast<long long>(cache_info_.layer_stride));
    printf("\tPage Stride: %lld \n", static_cast<long long>(cache_info_.page_stride));
    printf("\tPages Per Block: %lld \n", static_cast<long long>(pages_per_block));
    printf("\tBlock Size: %lld \n", static_cast<long long>(block_size_));

  }

  /**
   * The query function querys a given hash keys whether exist in current cache system.
   * A bool array will be returned as the query result,
   * where true means hash existence and false means not
   *
   * This function will throw no exception or error.
   */
  virtual std::vector<bool> query(const std::vector<std::string> &hashs) = 0;

  int64_t block_size() const;
  int64_t page_size() const;

  /**
   * The create function creates an asynchronous read/write task for kvcache.
   * All asynchronous read/write tasks are divided into a series of TaskBlocks,
  * each with a fixed size of block_size_.
   * This behavior is to facilitate efficient execution of subsequent transfers.
   * To create such a task, the user needs to provide the hash value corresponding
   * to each TaskBlock according to certain rules.
   * Asynchronous read requests will index the data based on these hash values.
   * The returned task object is jointly held by CacheService and Python.
   * The resources will only be destroyed when the task execution is completed
   * and the reference on the Python side is released.
   *
   * @param hashs Vector of hash strings to process
   * @param mode Operation mode of the cache task
   * @param kv_page_indexer A pytorch tensor of type int64 used to specify the kvcache
   * space corresponding to this task
   * @return Shared pointer to a newly created CacheTask
   * @exception If there is too many tasks(>)
   */
  std::shared_ptr<cache::task::CacheTask>
  create(const std::vector<std::string> &hashs, const torch::Tensor &kv_page_indexer_, const std::string &mode) {

    // Split task into blocks
    const int64_t page_size = cache_info_.page_size * cache_info_.num_of_layer;
    const int64_t page_per_block = (block_size_ + page_size - 1) / page_size;
    const int64_t num_of_pages = kv_page_indexer_.numel();
    const int64_t num_of_blocks = (num_of_pages + page_per_block - 1) / page_per_block;

    if (kv_page_indexer_.dtype() != torch::kInt32) {
      throw std::runtime_error("kv_page_indexer must be a tensor of type int32_t.");
    }

    if (!kv_page_indexer_.is_cpu()) {
      throw std::runtime_error("kv_page_indexer tensor must reside on CPU.");
    }

    auto kv_page_indexer = kv_page_indexer_;

    // Page mismatch
    if (num_of_blocks != hashs.size()) {
      throw std::runtime_error("The hash array length is incorrect, "
                               "want num of hash value: " +
                               std::to_string(num_of_blocks) + ", got: " + std::to_string(hashs.size()));
    }
    auto task = std::make_shared<cache::task::CacheTask>(hashs, kv_page_indexer, mode);

    // Submit task to the task queue
    if (queue_->submit(task) != cache::error::ACS_SUCCESS) {
      throw std::runtime_error("Failed to submit task to the queue. Too many tasks are currently queued.");
    }

    std::lock_guard<std::mutex> lock(lock_);
    // Push the task into taskpool to retain the reference to the task object on the C++ side.
    // This ensures that even if the task is released on the Python side,
    // the task object on the C++ side will not be immediately released.
    // All task releases should be performed after the task is released on the Python side
    // and the task is completed (or terminated) on the C++ side.
    taskpool_.push_back(task);
    return task;
  }

  /**
   * The az5 function notifies the system to immediately
   * abandon the subsequent execution of a task.
   * Since CacheService tasks are executed asynchronously in units of TaskBlocks,
   * some blocks in this task may have completed asynchronous access,
   * some may be running, and some may not have been launched yet.
   * The abort function will stop the launch of subsequent tasks
   * and immediately terminate the currently running blocks.
   * Blocks that have completed access are not affected.
   * Threads that are currently working may not be able to immediately end the task and respond,
   * and resources can only be reclaimed after the last thread completes execution.
   * After calling the abort function, the current task will no longer perform
   * read/write operations on the kvcache.
   */
  void az5(const std::shared_ptr<cache::task::CacheTask> &task) {
    for (cache::task::CacheBlock *block : task->blocks) {
      abort(block);
    }
  }

  /**
   * Notify the system to immediately abandon the subsequent execution of a Block.
   */
  void abort(cache::task::CacheBlock *block) {
    std::lock_guard<std::mutex> lock(block->task->lock);
    if (block->state == cache::task::State::Initial || block->state == cache::task::State::Working) {
      block->state = cache::task::State::Aborted;
      block->task->num_finished_blocks++;
    }
  }

  void deliver(cache::task::CacheBlock *block) {
    auto task = block->task;
    {
      std::lock_guard<std::mutex> lock(task->lock);

      if (block->state == cache::task::State::Working) {
        block->state = cache::task::State::Finished;
        task->num_finished_blocks++;
      }
    }

    // If all blocks of this task have completed execution,
    // remove the task from taskpool_
    if (task->ready()) {
      std::lock_guard<std::mutex> lock(lock_);

      int32_t i = 0;
      for (const std::shared_ptr<cache::task::CacheTask> &ptr : taskpool_) {
        if (ptr.get() == task) {
          break;
        } else {
          i++;
        }
      }
      if (i != taskpool_.size()) {
        taskpool_.erase(taskpool_.begin() + i);
      }
    }
  }

protected:
  std::mutex lock_;
  std::vector<std::shared_ptr<cache::task::CacheTask>> taskpool_;
  std::unique_ptr<cache::queue::TaskQueue> queue_;

  CacheParam_t cache_info_;

  int64_t block_size_;
};

inline int64_t CacheService::block_size() const {
  return block_size_;
}

inline int64_t CacheService::page_size() const {
  return cache_info_.page_size;
}

} // namespace cache::service
