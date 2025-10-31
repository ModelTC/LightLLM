#pragma once

#include "config.h"

#include "core/error.h"
#include "core/queue.h"
#include "core/task.h"
#include "service/base.h"
#include "storage/local.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace cache::task;
using namespace cache::queue;
using namespace cache::storage;

namespace cache::service {

/**
 * @brief LocalCacheService is a local asynchronous caching system.
 *
 * This system is used for asynchronously storing and accessing large model KV caches,
 * supporting storage on local disks or distributed storage systems.
 */
class LocalCacheService : public CacheService {
public:
  LocalCacheService() = delete;

  /**
   * @brief Constructor
   * @param file Path to the local storage file
   * @param storage_size Size of the storage
   * @param num_shard Num of storage shard
  * @param kvcache KV cache tensor (CPU)
   * @param num_workers Number of worker threads
   */
  LocalCacheService(const string &file, size_t storage_size, size_t num_shard, const torch::Tensor &kvcache,
                    const size_t num_workers)
      : CacheService(kvcache), stop_(false), num_workers_(num_workers), block_size_(0) {
    block_size_ = static_cast<size_t>(this->block_size());

    if (storage_size < block_size_) {
      throw runtime_error("file size < blocksize is not allowed.");
    }

    ensure_disk_capacity(file, storage_size, num_shard);

    storage_ = make_unique<LocalStorageEngine>(file, storage_size, num_shard, block_size_);

    r_cpu_buffers_.resize(num_workers_);
    w_cpu_buffers_.resize(num_workers_);

    for (size_t i = 0; i < num_workers_; ++i) {
      r_cpu_buffers_[i] = new char[block_size_];
      w_cpu_buffers_[i] = new char[block_size_];
    }
  }

  /**
   * @brief Destructor, stops all worker threads and releases resources.
   */
  ~LocalCacheService() {
    stop_ = true;
    for (auto &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }

    for (auto &buffer : r_cpu_buffers_) {
      delete[] buffer;
    }
    for (auto &buffer : w_cpu_buffers_) {
      delete[] buffer;
    }

    r_cpu_buffers_.clear();
    w_cpu_buffers_.clear();
  }

  /**
   * The query function querys a given hash keys whether exist in current cache system.
   * A bool array will be returned as the query result,
   * where true means hash existence and false means not
   *
   * This function will throw no exception or error.
   */
  std::vector<bool> query(const std::vector<std::string> &hashs) override {
    std::vector<bool> ret(hashs.size());
    for (int32_t i = 0; i < hashs.size(); ++i) {
      ret[i] = storage_->query(hashs[i]);
    }
    return ret;
  }

  /**
   * @brief Runs worker threads.
   */
  void run() {
    stop_ = false;
    for (size_t i = 0; i < num_workers_; ++i) {
      workers_.emplace_back(&LocalCacheService::work, this, static_cast<int32_t>(i));
    }
  }

  std::shared_ptr<cache::storage::LocalStorageEngine::HashInfo> get_hash_info() {
    return storage_->getHashInfo();
  }

  void set_hash_info(const std::shared_ptr<cache::storage::LocalStorageEngine::HashInfo> &info) {
    storage_->setHashInfo(info);
  }

private:
  /**
   * @brief Core logic of the worker thread.
   * @param index Thread index
   */
  void work(int32_t index) {
    while (!stop_) {
      if (auto block = this->queue_->claim()) {
        if (block != nullptr) {
          CacheTask *task = block->task;
          char *cpu_buffer = (task->mode == Mode::Read) ? r_cpu_buffers_[index] : w_cpu_buffers_[index];
          processTask(block, cpu_buffer);
        }
      }
    }
  }

  void processTask(CacheBlock *block, char *cpu_buffer) {
    CacheTask *task = block->task;
    torch::Tensor page_tensor = task->page_indexer;
    auto bid = block->block_idx;

    const int64_t page_size = this->cache_info_.page_size * this->cache_info_.num_of_layer;
    const int64_t page_per_block = (static_cast<int64_t>(block_size_) + page_size - 1) / page_size;
    const int64_t remaining_pages = page_tensor.numel() - bid * page_per_block;
    if (remaining_pages <= 0) {
      this->abort(block);
      return;
    }
    const int64_t num_of_page = std::min(remaining_pages, page_per_block);
    int32_t *page_ptr = reinterpret_cast<int32_t *>(page_tensor.data_ptr()) + bid * page_per_block;

    if (!page_tensor.device().is_cpu()) {
      throw std::runtime_error("kv_page_indexer tensor must reside on CPU for CPU cache service.");
    }

    bool success = false;
    if (task->mode == Mode::Read) {
      success = handleReadCpu(block, cpu_buffer, page_ptr, num_of_page);
    } else {
      success = handleWriteCpu(block, cpu_buffer, page_ptr, num_of_page);
    }

    success ? this->deliver(block) : this->abort(block);
  }

  bool handleReadCpu(CacheBlock *block, char *cpu_buffer, int32_t *page_ptr, int64_t num_of_page) {
    if (storage_->read(cpu_buffer, block->hash) != block_size_) {
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(block->task->lock);
      if (block->state != cache::task::State::Working) {
        return false;
      }
    }

    cpu_scatter(this->cache_info_, cpu_buffer, page_ptr, num_of_page);
    return true;
  }

  bool handleWriteCpu(CacheBlock *block, char *cpu_buffer, int32_t *page_ptr, int64_t num_of_page) {
    {
      std::lock_guard<std::mutex> lock(block->task->lock);
      if (block->state != cache::task::State::Working) {
        return false;
      }
    }

    cpu_gather(this->cache_info_, cpu_buffer, page_ptr, num_of_page);
    return storage_->write(cpu_buffer, block->hash) == block_size_;
  }

  static void cpu_scatter(const CacheParam_t &info, const char *block, const int32_t *page_idx,
                          int64_t num_of_page) {
    const int64_t page_size = info.page_size;
    const int64_t num_of_layer = info.num_of_layer;
    const int64_t page_stride = info.page_stride;
    const int64_t total_pages = info.num_of_page;
    const int64_t page_bytes = page_size * num_of_layer;

    for (int64_t local_page = 0; local_page < num_of_page; ++local_page) {
      const int32_t dst_page = page_idx[local_page];
      if (dst_page < 0 || dst_page >= total_pages) {
        throw std::runtime_error("kv page index out of range in cpu_scatter.");
      }

      char *dst_page_ptr = info.base_ptr + static_cast<int64_t>(dst_page) * page_stride;
      const char *src_page_ptr = block + local_page * page_bytes;

      std::memcpy(dst_page_ptr, src_page_ptr, page_bytes);
    }
  }

  static void cpu_gather(const CacheParam_t &info, char *block, const int32_t *page_idx, int64_t num_of_page) {
    const int64_t page_size = info.page_size;
    const int64_t num_of_layer = info.num_of_layer;
    const int64_t page_stride = info.page_stride;
    const int64_t total_pages = info.num_of_page;
    const int64_t page_bytes = page_size * num_of_layer;

    for (int64_t local_page = 0; local_page < num_of_page; ++local_page) {
      const int32_t src_page = page_idx[local_page];
      if (src_page < 0 || src_page >= total_pages) {
        throw std::runtime_error("kv page index out of range in cpu_gather.");
      }

      const char *src_page_ptr = info.base_ptr + static_cast<int64_t>(src_page) * page_stride;
      char *dst_page_ptr = block + local_page * page_bytes;

      std::memcpy(dst_page_ptr, src_page_ptr, page_bytes);
    }
  }

  size_t block_size_;                                    ///< Block size
  unique_ptr<LocalStorageEngine> storage_;               ///< Local storage engine
  vector<thread> workers_;                               ///< Worker threads
  bool stop_;                                            ///< Thread stop flag
  size_t num_workers_;                                   ///< Number of worker threads
  vector<char*> r_cpu_buffers_;                          ///< CPU buffers for read worker
  vector<char*> w_cpu_buffers_;                          ///< CPU buffers for write worker

  // Ensure the backing storage path exposes enough disk capacity for the requested cache size.
  static void ensure_disk_capacity(const string &file, size_t storage_size, size_t num_shard) {
    namespace fs = std::filesystem;

    if (num_shard == 0) {
      throw std::runtime_error("num_shard must be greater than zero");
    }

    fs::path base_path(file);
    fs::path target_dir = base_path.parent_path();
    if (target_dir.empty()) {
      target_dir = fs::current_path();
    }

    std::error_code ec;
    fs::path probe_dir = target_dir;
    while (!probe_dir.empty() && !fs::exists(probe_dir, ec)) {
      ec.clear();
      probe_dir = probe_dir.parent_path();
    }

    if (probe_dir.empty()) {
      probe_dir = fs::current_path();
    }

    ec.clear();
    fs::space_info info = fs::space(probe_dir, ec);
    if (ec) {
      throw std::runtime_error("Failed to query available space for path: " + probe_dir.string() + ", reason: " +
                               ec.message());
    }

    uintmax_t reclaimable = 0;
    for (size_t i = 0; i < num_shard; ++i) {
      fs::path shard_path = base_path;
      shard_path += "_" + std::to_string(i);

      std::error_code exists_ec;
      if (!fs::exists(shard_path, exists_ec) || exists_ec) {
        continue;
      }

      std::error_code size_ec;
      const uintmax_t shard_size = fs::file_size(shard_path, size_ec);
      if (size_ec) {
        continue;
      }

      if (std::numeric_limits<uintmax_t>::max() - reclaimable < shard_size) {
        reclaimable = std::numeric_limits<uintmax_t>::max();
        break;
      }
      reclaimable += shard_size;
    }

    uintmax_t total_available = info.available;
    if (std::numeric_limits<uintmax_t>::max() - total_available < reclaimable) {
      total_available = std::numeric_limits<uintmax_t>::max();
    } else {
      total_available += reclaimable;
    }

    const uintmax_t required = static_cast<uintmax_t>(storage_size);
    if (total_available < required) {
      throw std::runtime_error("Insufficient disk space for local cache service. Required " +
                               std::to_string(required) + " bytes but only " +
                               std::to_string(total_available) + " bytes available including reclaimable shards.");
    }
  }
};

} // namespace cache::service