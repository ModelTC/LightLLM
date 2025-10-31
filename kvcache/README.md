# GPU 缓存服务

## 概述

本项目实现了一个异步 GPU 内存缓存系统，旨在高性能应用中高效管理和优化 GPU 内存使用。系统将 GPU 内存分割为固定大小的块，使用类似银行家算法的资源调度策略管理任务，并通过多线程并发处理缓存任务，从而提高整体吞吐量和响应能力。

## 特性

- **GPU 内存管理**  
  在连续的 GPU 内存区域中分配缓存，并将其划分为多个固定大小的块以供使用。
- **异步任务队列**  
  使用任务队列管理缓存任务，保证任务调度的高效性和资源分配的合理性。
- **读写模式支持**  
  支持读写两种操作模式，通过动态分配内存块满足不同任务的需求。
- **多线程并发处理**  
  利用多个工作线程异步领取缓存块并处理任务，充分发挥多核和 GPU 的性能优势。
- **健壮的错误处理**  
  采用详细的错误码体系，涵盖内存分配失败、队列溢出等多种异常情况，便于问题定位与调试。
- **本地存储集成**  
  集成本地存储引擎，实现数据的持久化存储，可用于本地磁盘或分布式存储系统。

## 代码结构

- **core/**
  - **task.h**  
    定义了 `CacheTask` 和 `CacheBlock` 类，用于描述任务和对应的内存块，管理任务状态与操作模式。

  - **buffer.h**  
    实现了 `CacheBuffer` 类，负责 GPU 内存的分配、切分以及空闲内存块的管理。

  - **queue.h**  
    包含了 `TaskQueue` 类，用于任务的调度与分发，管理可用的缓存块和活跃任务列表。

  - **error.h**  
    定义了错误码枚举 `ACSError_t`，用于指示各类操作的执行结果。

- **service.h**  
  提供了缓存服务的基类 `CacheService`，用于创建、提交和释放缓存任务。

- **LocalCacheService（服务实现部分）**  
  实现了基于本地存储的异步缓存服务，管理工作线程并负责调用任务队列处理缓存任务。

- **worker/** 与 **storage/**  
  分别包含工作线程的实现和本地存储引擎的相关代码，支持缓存任务的具体处理和数据持久化。

## 系统要求

- **编译器要求**  
  需要支持 C++11 或更高版本的编译器。

- **GPU 环境**  
  需要相应的 GPU 驱动和库，以支持 GPU 内存的分配和管理。

## 构建步骤

1. **克隆代码仓库：**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **编译（以 uv 为例）：**

    ```python
    uv build \
      -C override=cmake.options.Python_ROOT_DIR=$PWD/.venv \
      -C override=cmake.options.CMAKE_PREFIX_PATH=$PWD/.venv/lib/python3.10/site-packages/torch/share/cmake \
      -C override=cmake.options.CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    ```

## 创建与提交任务

- 使用 CacheService::create() 方法创建包含待处理 hash 序列和操作模式（读或写）的缓存任务。
- 通过 LocalCacheService::submit() 方法提交任务，任务队列会在接收到足够资源后开始处理。

## 工作线程处理

- 调用 LocalCacheService::run() 启动工作线程，每个线程会从任务队列中领取缓存块并执行异步处理。

## 资源释放

- 当任务完成或中止时，调用 CacheService::free() 释放相应的内存块和资源。

## 已知问题及未来改进

### GPU 性能优化

针对 GPU 相关操作进行专项优化和集成，进一步提升内存操作效率和系统响应速度。
