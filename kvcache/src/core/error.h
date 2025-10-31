#pragma once

namespace cache::error {

/**
 * @brief 错误码类型定义
 *
 * 下面的错误码用于指示 CacheService 的各种操作结果。
 */
enum ACSError_t {
  ACS_SUCCESS = 0,              // 操作成功
  ACS_MISSING_FILE = 1,         // 文件不存在
  ACS_FILE_EXISTS = 2,          // 文件已存在（冲突）
  ACS_FILE_CREATION_FAILED = 3, // 文件或目录创建失败
  ACS_TIMEOUT = 4,              // 操作超时
  ACS_CUDA_ERROR = 5,           // Cuda 操作失败
  ACS_QUEUE_LENGTH_EXCEED = 6,  // 任务队列超长
  ACS_HASH_ERROR = 7,           // 哈希值错误
  ACS_CACHE_ERROR = 8,          // 缓存分配错误
  ACS_TASK_MODE_ERROR = 9,      // 任务模式错误
  ACS_BUFFER_EXCEED = 10        // 缓存不够
};

} // namespace cache::error
