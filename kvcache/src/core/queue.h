#pragma once

#include "config.h"
#include "core/error.h"
#include "core/task.h"

#include <boost/lockfree/queue.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace cache::queue {

/**
 * @brief Task queue class that stores all tasks & controls task state.
 *
 */
class TaskQueue {
public:
  TaskQueue() = default;
  TaskQueue(TaskQueue &&other) = delete;
  TaskQueue &operator=(TaskQueue &&other) = delete;

  cache::error::ACSError_t submit(const std::shared_ptr<cache::task::CacheTask> &task) {
    for (cache::task::CacheBlock *block : task->blocks) {
      blocks_.push(block);
    }
    return cache::error::ACS_SUCCESS;
  }

  cache::task::CacheBlock *claim() {
    cache::task::CacheBlock *block = nullptr;
    if (blocks_.pop(block)) {
      block->state = cache::task::State::Working;
      return block;
    }
    return nullptr;
  }

private:
  boost::lockfree::queue<cache::task::CacheBlock *> blocks_{
      ACS_QueueSize};   ///< Lock-free queue for cache blocks.
  int32_t max_depth_{}; ///< Maximum queue depth.
};

} // namespace cache::queue
