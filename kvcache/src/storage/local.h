#pragma once

#include "core/error.h"
#include "storage/base.h"

#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cache {
namespace storage {

/**
 * @brief Cache index class
 */
class LocalCacheIndex {
public:
  LocalCacheIndex() = delete;
  /**
   * @brief Constructor
   * @param capacity Maximum number of hash values to store
   */
  LocalCacheIndex(size_t capacity) : capacity_(capacity) {
    for (size_t i = 0; i < capacity; i++) {
      empty_block_list_.push_back(i);
    }
  }

  /**
   * @brief Checks if a hash value exists
   *
   * If it exists, returns true and updates the LRU order (moves to the front);
   * If it doesn't exist, returns false.
   *
   * @param hash The hash value to search for
   * @return True if exists, false otherwise
   */
  bool exists(const std::string &hash) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = index_.find(hash);
    if (it == index_.end()) {
      return false;
    }
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
    return true;
  }

  /**
   * @brief Checkout a slot for a hash, allocating or reusing storage as needed.
   *
   * If the hash already exists, its offset is returned and the entry promoted to MRU.
   * If the hash is new and free space remains, a free offset is consumed.
   * Otherwise the LRU entry is evicted and its offset reused for the new hash.
   *
   * @param hash The hash value we wish to store.
   * @param offset Output parameter receiving the chosen file offset index.
   * @param evicted_hash Output parameter holding the evicted hash (empty if none).
   * @return True if the hash was newly inserted (with or without eviction), false if it already existed.
   */
  bool checkout_slot(const std::string &hash, size_t &offset, std::string &evicted_hash) {
    std::lock_guard<std::mutex> lock(lock_);

    auto existing = index_.find(hash);
    if (existing != index_.end()) {
      lru_list_.splice(lru_list_.begin(), lru_list_, existing->second.lru_iterator);
      offset = existing->second.foffset;
      evicted_hash.clear();
      return false;
    }

    if (!empty_block_list_.empty()) {
      offset = empty_block_list_.back();
      empty_block_list_.pop_back();
      evicted_hash.clear();
    } else {
      if (lru_list_.empty()) {
        throw std::runtime_error("LocalCacheIndex capacity exhausted with no entries to evict");
      }
      const std::string &victim_hash = lru_list_.back();
      auto victim_it = index_.find(victim_hash);
      if (victim_it == index_.end()) {
        throw std::runtime_error("LRU list and index inconsistent");
      }
      offset = victim_it->second.foffset;
      evicted_hash = victim_hash;
      lru_list_.pop_back();
      index_.erase(victim_it);
    }

    lru_list_.push_front(hash);
    index_[hash] = {lru_list_.begin(), offset};
    return true;
  }

  /**
   * @brief Gets the offset of a hash value
   *
   * @param hash The hash value to search for
   * @return The offset if exists, -1 otherwise
   */
  size_t get_offset(const std::string &hash) {
    std::lock_guard<std::mutex> lock(lock_);
    auto it = index_.find(hash);
    if (it == index_.end()) {
      return -1;
    } else {
      return it->second.foffset;
    }
  }

private:
  /**
   * @brief Internal structure to store LRU list iterator and exists call count
   */
  struct IndexEntry {
    std::list<std::string>::iterator lru_iterator;
    size_t foffset; // File pointer
  };

  size_t capacity_;                                   ///< Maximum number of hash values to store
  std::list<std::string> lru_list_;                   ///< LRU list, head is most recent, tail is least recently used
  std::list<size_t> empty_block_list_;                ///< List of free disk blocks
  std::unordered_map<std::string, IndexEntry> index_; ///< Map from hash value to IndexEntry
  std::mutex lock_;
};

class LocalStorageEngine : public StorageEngine {
public:
  struct HashInfo {
    std::vector<std::shared_ptr<LocalCacheIndex>> caches;
    std::vector<std::shared_ptr<std::mutex>> locks;

    HashInfo() = default;
  };

  /**
   * @brief Constructor for IO sharding.
   * @param filename Base filename; actual files will be filename_0, filename_1, ...
   * @param storage_size Total file size across all shards.
   * @param shard Number of shards.
   */
  LocalStorageEngine(const std::string &filename, const size_t storage_size, const size_t shard, const size_t block_size)
      : filename_(filename), storage_size_(storage_size), shard_(shard), block_size_(block_size) {

    // 每个 shard 分到的文件大小
    size_t shard_storage_size = storage_size_ / shard_;
    // 每个 shard 能存储的块数
    size_t shard_capacity = shard_storage_size / block_size;

    // 初始化缓存索引、锁和文件对象
    caches_.resize(shard_);
    locks_.resize(shard_);
    files_.resize(shard_);

    for (size_t i = 0; i < shard_; i++) {
      caches_[i] = std::make_shared<LocalCacheIndex>(shard_capacity);
      locks_[i] = std::make_shared<std::mutex>();
    }
    createOrOpenFiles(shard_storage_size);
  }

  ~LocalStorageEngine() override {
    for (size_t i = 0; i < shard_; i++) {
      if (files_[i].is_open()) {
        files_[i].close();
      }
    }
  }

  bool query(const std::string &hash) override {
    size_t shard_id = getShard(hash);
    return caches_[shard_id]->exists(hash);
  }

  size_t write(const char *buf, const std::string &hash) override {
    size_t shard_id = getShard(hash);
  size_t slot = 0;
  std::string evicted_hash;
  caches_[shard_id]->checkout_slot(hash, slot, evicted_hash);
  (void)evicted_hash;

    std::lock_guard<std::mutex> lock(*locks_[shard_id]);
    size_t offset = slot * block_size_;
    files_[shard_id].seekp(offset, std::ios::beg);
    files_[shard_id].write(buf, block_size_);
    files_[shard_id].flush();
    return block_size_;
  }

  size_t read(char *buf, const std::string &hash) override {
    size_t shard_id = getShard(hash);

    if (!caches_[shard_id]->exists(hash)) {
      return 0; // Hash does not exist
    }

    std::lock_guard<std::mutex> lock(*locks_[shard_id]);
    size_t offset = caches_[shard_id]->get_offset(hash) * block_size_;
    files_[shard_id].seekg(offset, std::ios::beg);
    files_[shard_id].read(buf, block_size_);
    return block_size_;
  }

  std::shared_ptr<HashInfo> getHashInfo() {
    auto info = std::make_shared<HashInfo>();
    info->caches = caches_;
    info->locks = locks_;
    return info;
  }

  void setHashInfo(const std::shared_ptr<HashInfo> &info) {
    if (!info) {
      throw std::runtime_error("HashInfo is null");
    }
    if (info->caches.size() != shard_ || info->locks.size() != shard_) {
      throw std::runtime_error("HashInfo shard size mismatch");
    }
    caches_ = info->caches;
    locks_ = info->locks;
  }

private:
  inline size_t getShard(const std::string &hash) { return std::hash<std::string>{}(hash) % shard_; }

  void createOrOpenFiles(size_t shard_storage_size) {
    for (size_t i = 0; i < shard_; i++) {
      std::stringstream ss;
      ss << filename_ << "_" << i;
      std::string shard_filename = ss.str();
      files_[i].open(shard_filename, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
      if (!files_[i].is_open()) {
        throw std::runtime_error("Failed to open file: " + shard_filename);
      }
      files_[i].seekp(shard_storage_size - 1, std::ios::beg);
      files_[i].write("", 1);
      files_[i].seekp(0, std::ios::beg);
    }
  }

  std::string filename_;
  size_t storage_size_;
  size_t shard_;
  size_t block_size_;
  std::vector<std::fstream> files_;
  std::vector<std::shared_ptr<std::mutex>> locks_;
  std::vector<std::shared_ptr<LocalCacheIndex>> caches_;
};

} // namespace storage
} // namespace cache
