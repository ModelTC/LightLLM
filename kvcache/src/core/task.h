#pragma once

#include "mytorch.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cache::task {

enum State { Initial = 0, Working = 1, Finished = 2, Aborted = 3 };

enum Mode { Write = 1, Read = 2 };

class CacheTask; // 预声明 CacheTask 以便在 CacheBlock 中使用

/// @brief Cache block class, storing SHA-256 hash and GPU memory buffer pointer.
class CacheBlock {
public:
  /// @brief Constructor, calculates the SHA-256 hash of the data.
  /// @param hash Data hash.
  /// @param m Mode: Read or Write
  /// @param task Corresponding CacheTask
  CacheBlock(std::string hash_, const int64_t block_idx_, CacheTask *task_)
      : hash(std::move(hash_)), task(task_), block_idx(block_idx_) {}

  /// @brief 禁止拷贝和移动
  CacheBlock(CacheBlock &&other) = delete;
  CacheBlock &operator=(CacheBlock &&other) = delete;

  bool ready() const { return state == State::Finished; }

  int64_t block_idx;
  CacheTask *task; ///< Corresponding Task
  std::string hash;           ///< Hash of the block.
  State state{};              ///< Read/write state of the block.
};

/// @brief Task cache class, storing only cache blocks.
class CacheTask {
public:
  CacheTask() = delete;

  /// @brief Constructor, determines mode from user input ('r' or 'w').
  /// @param hashs Hash sequence.
  /// @param mode_str Mode string: "r" for Read, "w" for Write
  CacheTask(const std::vector<std::string> &hashs, torch::Tensor kv_page_indexer, const std::string &mode_str)
      : num_finished_blocks(0), page_indexer(std::move(kv_page_indexer)) {

    if (mode_str == "r") {
      mode = Mode::Read;
    } else if (mode_str == "w") {
      mode = Mode::Write;
    } else {
      throw std::invalid_argument("Invalid mode string. Use 'r' for Read or 'w' for Write.");
    }

    blocks.reserve(hashs.size());
    for (int64_t idx = 0; idx < hashs.size(); ++idx) {
      blocks.emplace_back(new CacheBlock(hashs[idx], idx, this));
    }
  }

  ~CacheTask() {
    for (auto block : blocks) {
      delete block;
    }
  }

  /// @brief 禁止拷贝和移动
  CacheTask(CacheTask &&other) = delete;
  CacheTask &operator=(CacheTask &&other) = delete;

  bool ready() const { return num_finished_blocks == blocks.size(); }

  std::vector<State> state() {
    auto ret = std::vector<State>(blocks.size());
    for (int32_t i = 0; i < blocks.size(); ++i) {
      ret[i] = blocks[i]->state;
    }
    return ret;
  }

  // 这个指针是用来标记 task 中的数据存取位置的
  torch::Tensor page_indexer;

  std::mutex lock;                             ///< Task state lock
  std::vector<CacheBlock *> blocks; ///< Blocks stored as shared_ptr
  int64_t num_finished_blocks;
  Mode mode; ///< Read/write mode of the task.
};

} // namespace cache::task
