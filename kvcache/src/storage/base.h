#pragma once

#include "core/error.h"
#include "core/task.h"

#include <string>

namespace cache::storage {

/**
 * @brief 存储引擎基类
 */
class StorageEngine {
public:
  virtual ~StorageEngine() = default;

  /**
   * @brief 查询存储引擎是否保存了给定的哈希值
   */
  virtual bool query(const std::string &hash) = 0;

  /**
   * @brief 写入给定的数据到存储引擎
   */
  virtual size_t write(const char *buf, const std::string &hash) = 0;

  /**
   * @brief 从存储引擎中读取数据，如果 hash 值不存在，返回0，否则返回读取的字节数
   */
  virtual size_t read(char *buf, const std::string &hash) = 0;

protected:
  // TODO 存储分页
  std::mutex lock_; // 存储锁
};

} // namespace cache::storage
