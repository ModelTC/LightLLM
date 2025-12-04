.. _multi_level_cache_deployment:

多级缓存部署指南
================

LightLLM 支持多级 KV Cache 缓存机制,通过 GPU (L1)、CPU (L2) 和磁盘 (L3) 三级缓存的组合,可以大幅降低部署成本并提升长文本场景下的吞吐能力。本文档详细介绍如何配置和使用多级缓存功能。

前置依赖
--------

使用三级缓存 (L1 + L2 + L3) 需要安装 **LightMem** 库。LightMem 是一个高性能的 KV Cache 磁盘管理库，专为大语言模型推理系统设计。

.. note::
   
   如果只使用二级缓存 (L1 + L2)，即 GPU + CPU 缓存，则不需要安装 LightMem。
   只有启用 ``--enable_disk_cache`` 参数时才需要 LightMem 支持。

安装 LightMem
~~~~~~~~~~~~~

**系统要求:**

- Python 3.10 或更高版本
- CMake 3.25 或更高版本
- C++17 兼容编译器
- PyTorch (CPU 版本即可)
- Boost C++ 库
- pybind11 (通过 pip 自动安装)

**平台支持:**

- **Linux**: 完全支持,包含 ``posix_fadvise`` 优化
- **macOS**: 支持,但不含 ``posix_fadvise`` 优化

**安装步骤:**

1. 安装系统依赖

   **Ubuntu/Debian:**
   
   .. code-block:: bash
   
       sudo apt-get update
       sudo apt-get install cmake build-essential libboost-all-dev

   **使用 Conda (推荐):**
   
   .. code-block:: bash
   
       conda install -c conda-forge cmake cxx-compiler boost libboost-devel

   **macOS:**
   
   .. code-block:: bash
   
       brew install cmake boost

2. 安装 PyTorch

   .. code-block:: bash
   
       pip install torch

3. 编译安装 LightMem

   LightMem 库位于 LightLLM 项目的 ``LightMem/`` 目录下:
   
   .. code-block:: bash
   
       # 进入 LightMem 目录
       cd /path/to/lightllm/LightMem
       
       # 使用 pip 安装 (推荐)
       pip install -v .
       
       # 或者手动构建 wheel 包
       python -m build --wheel
       pip install dist/*.whl

4. 验证安装

   .. code-block:: bash
   
       python -c "from light_mem import PyLocalCacheService; print('LightMem installed successfully!')"

**环境变量配置 (可选):**

LightMem 支持通过环境变量调整缓存块大小:

.. code-block:: bash

    # 设置缓存块大小 (单位: MB,默认 64MB)
    export LIGHTMEM_MAX_BLOCK_SIZE_MB=64

- **较大的块** (如 128MB): 减少开销,适合顺序访问,但小操作延迟会增加
- **较小的块** (如 32MB): 更细粒度的控制,适合随机访问,但每次操作开销更高
- 建议保持默认值 64MB,除非有特殊性能需求

多级缓存架构
------------

LightLLM 的多级缓存系统采用分层设计:

- **L1 Cache (GPU 显存)**: 最快速的缓存层,存储热点请求的 KV Cache,提供最低延迟
- **L2 Cache (CPU 内存)**: 中速缓存层,存储相对较冷的 KV Cache,成本低于 GPU
- **L3 Cache (磁盘存储)**: 最大容量缓存层,存储长期不活跃的 KV Cache,成本最低

**工作原理:**

1. 当请求的 KV Cache 无法全部保存在 GPU 时,系统会自动将部分 KV Cache 迁移到 CPU 内存
2. 当 CPU 内存也不足时,会进一步将最冷的数据迁移到磁盘
3. 当需要访问迁移的数据时,系统会自动从低层缓存加载回高层缓存
4. 这种分层机制对应用透明,无需修改代码

**适用场景:**

- 超长文本处理 (如百万 token 级别的上下文)
- 高并发对话场景 (需要缓存大量历史对话)
- 成本敏感的部署 (用更便宜的内存和磁盘替代昂贵的 GPU 显存)
- Prompt Cache 场景 (复用常见的 prompt 前缀)

部署方案
--------

1. L1 + L2 二级缓存 (GPU + CPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

适合大多数场景,在保持高性能的同时显著提升缓存容量。

**启动命令:**

.. code-block:: bash

    # 启用 GPU + CPU 二级缓存
    LOADWORKER=18 python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3-235B-A22B \
        --tp 8 \
        --graph_max_batch_size 500 \
        --enable_fa3 \
        --mem_fraction 0.88 \
        --enable_cpu_cache \
        --cpu_cache_storage_size 400 \
        --cpu_cache_token_page_size 64

**参数说明:**

基础参数
^^^^^^^^

- ``LOADWORKER=18``: 模型加载线程数,提高模型加载速度,建议设置为 CPU 核心数的一半
- ``--model_dir``: 模型文件路径,支持本地路径或 HuggingFace 模型名称
- ``--tp 8``: 张量并行度,使用 8 个 GPU 进行模型推理
- ``--graph_max_batch_size 500``: CUDA Graph 最大批次大小,影响吞吐量和显存占用
- ``--enable_fa3``: 启用 Flash Attention 3.0,提升注意力计算速度
- ``--mem_fraction 0.88``: GPU 显存使用比例,建议设置为 0.85-0.90

CPU 缓存参数
^^^^^^^^^^^^

- ``--enable_cpu_cache``: **启用 CPU 缓存** (L2 层),这是开启二级缓存的核心参数
- ``--cpu_cache_storage_size 400``: **CPU 缓存容量**,单位为 GB,此处设置为 400GB
  
  - 容量规划: 每 GB 大约可以缓存 ~13K tokens 的 KV Cache (取决于模型配置)
  - 建议设置为系统可用内存的 60%-80%
  - 对于 512GB 内存的机器,建议设置为 300-400GB

- ``--cpu_cache_token_page_size 64``: **CPU 缓存页大小**,单位为 token 数量
  
  - 默认值为 256,建议范围 64-512
  - 较小的页大小 (如 64) 适合细粒度的缓存管理,减少内存碎片
  - 较大的页大小 (如 256) 适合大批量数据迁移,提高传输效率
  - 该值需要权衡内存利用率和传输开销

**性能优化建议:**

1. **使用 Hugepages**: 启用大页内存可以显著提升 CPU 缓存性能

   .. code-block:: bash

       # 配置 Hugepages (需要 root 权限)
       echo 200000 > /proc/sys/vm/nr_hugepages
       
       # 验证配置
       cat /proc/meminfo | grep Huge

2. **NUMA 绑定**: 在多路服务器上,将进程绑定到特定 NUMA 节点以减少跨节点内存访问

   .. code-block:: bash

       numactl --cpubind=0 --membind=0 python -m lightllm.server.api_server ...

2. L1 + L2 + L3 三级缓存 (GPU + CPU + Disk)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

适合超长文本或极高并发场景,提供最大的缓存容量。

.. important::
   
   使用三级缓存需要先安装 **LightMem** 库,请参考上文"前置依赖"章节完成安装。
   如果未安装 LightMem,启动服务时会报错: ``ImportError: LightMem library is required for disk cache functionality``

**启动命令:**

.. code-block:: bash

    # 启用 GPU + CPU + Disk 三级缓存
    LOADWORKER=18 python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3-235B-A22B \
        --tp 8 \
        --graph_max_batch_size 500 \
        --enable_fa3 \
        --mem_fraction 0.88 \
        --enable_cpu_cache \
        --cpu_cache_storage_size 400 \
        --cpu_cache_token_page_size 256 \
        --enable_disk_cache \
        --disk_cache_storage_size 1000 \
        --disk_cache_dir /mnt/ssd/disk_cache_dir

**参数说明:**

磁盘缓存参数
^^^^^^^^^^^^

在二级缓存的基础上,增加以下参数:

- ``--enable_disk_cache``: **启用磁盘缓存** (L3 层),开启三级缓存的核心参数
- ``--disk_cache_storage_size 1000``: **磁盘缓存容量**,单位为 GB,此处设置为 1TB
  
  - 容量规划: 每 GB 大约可以缓存 ~13K tokens 的 KV Cache
  - 建议根据存储空间和业务需求设置,通常设置为数百 GB 到数 TB
  - 1TB 容量约可缓存 13M tokens 的 KV Cache

- ``--disk_cache_dir /mnt/ssd/disk_cache_dir``: **磁盘缓存目录**,指定用于持久化缓存数据的目录
  
  - 如果不设置,会使用系统临时目录
  - 强烈建议使用 SSD/NVMe 存储,避免使用 HDD (性能差距可达 10-100 倍)
  - 确保目录具有足够的读写权限和磁盘空间
  - 建议使用独立的磁盘分区,避免影响系统盘

- ``--cpu_cache_token_page_size 256``: 启用磁盘缓存时,建议增大页大小以提高 I/O 效率

**存储设备选择:**

.. list-table::
   :widths: 20 30 30 20
   :header-rows: 1

   * - 存储类型
     - 读取速度
     - 适用场景
     - 成本
   * - NVMe SSD
     - 3-7 GB/s
     - 推荐,性能最佳
     - 中等
   * - SATA SSD
     - 500 MB/s
     - 可用,性能尚可
     - 较低
   * - HDD
     - 100-200 MB/s
     - 不推荐,性能较差
     - 最低

3. 完整配置示例
~~~~~~~~~~~~~~~

以下是一个生产环境的完整配置示例:

.. code-block:: bash

    #!/bin/bash
    
    # 环境变量配置
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export LOADWORKER=18
    
    # 配置 Hugepages (可选,需要 root 权限)
    # echo 200000 > /proc/sys/vm/nr_hugepages
    
    # 启动服务
    python -m lightllm.server.api_server \
        --host 0.0.0.0 \
        --port 8000 \
        --model_dir /models/Qwen3-235B-A22B \
        --tokenizer_mode auto \
        --tp 8 \
        --graph_max_batch_size 500 \
        --enable_fa3 \
        --mem_fraction 0.88 \
        --max_total_token_num 100000 \
        --max_req_input_len 32768 \
        --max_req_total_len 65536 \
        --enable_cpu_cache \
        --cpu_cache_storage_size 400 \
        --cpu_cache_token_page_size 128 \
        --enable_disk_cache \
        --disk_cache_storage_size 1000 \
        --disk_cache_dir /mnt/nvme/lightllm_cache \
        --log_level info

**额外参数说明:**

- ``--host 0.0.0.0``: 监听所有网络接口
- ``--port 8000``: 服务端口
- ``--max_total_token_num 100000``: 系统最大 token 数量,影响最大并发能力
- ``--max_req_input_len 32768``: 单个请求最大输入长度
- ``--max_req_total_len 65536``: 单个请求最大总长度 (输入 + 输出)
- ``--log_level info``: 日志级别,可选 debug/info/warning/error

多机部署
--------

多级缓存也支持多机部署,可以跨节点共享缓存。

**Node 0 启动命令:**

.. code-block:: bash

    export nccl_host=$1  # 主节点 IP
    LOADWORKER=18 python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --tp 16 \
        --enable_fa3 \
        --mem_fraction 0.88 \
        --enable_cpu_cache \
        --cpu_cache_storage_size 400 \
        --cpu_cache_token_page_size 256 \
        --enable_disk_cache \
        --disk_cache_storage_size 1000 \
        --disk_cache_dir /mnt/shared_storage/cache \
        --nnodes 2 \
        --node_rank 0 \
        --nccl_host $nccl_host \
        --nccl_port 2732

**Node 1 启动命令:**

.. code-block:: bash

    export nccl_host=$1  # 主节点 IP
    LOADWORKER=18 python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --tp 16 \
        --enable_fa3 \
        --mem_fraction 0.88 \
        --enable_cpu_cache \
        --cpu_cache_storage_size 400 \
        --cpu_cache_token_page_size 256 \
        --enable_disk_cache \
        --disk_cache_storage_size 1000 \
        --disk_cache_dir /mnt/shared_storage/cache \
        --nnodes 2 \
        --node_rank 1 \
        --nccl_host $nccl_host \
        --nccl_port 2732

**注意事项:**

- 多机部署时,建议使用共享存储 (如 NFS) 作为磁盘缓存目录
- 确保所有节点可以访问相同的磁盘缓存目录
- 网络带宽会影响跨节点的缓存同步性能

性能调优建议
------------

1. **内存分配策略**

   - L1 (GPU): 保留用于最热数据,建议 ``mem_fraction`` 设置为 0.85-0.90
   - L2 (CPU): 分配系统内存的 60%-80%,留出足够的操作系统缓存
   - L3 (Disk): 根据存储空间灵活配置,建议至少 500GB

2. **页大小选择**

   .. code-block:: text

       场景                      推荐页大小        说明
       短文本对话                64-128           减少碎片,提高命中率
       长文本处理                256-512          提高传输效率
       混合场景                  128-256          平衡性能

3. **存储优化**

   - 使用 RAID 0 或多块 NVMe 以提高 I/O 吞吐
   - 定期清理不活跃的缓存数据
   - 监控磁盘 I/O 使用率,避免成为瓶颈

4. **监控指标**

   关注以下指标以评估多级缓存效果:
   
   - GPU 显存使用率 (通过 ``nvidia-smi`` 查看)
   - CPU 内存使用率 (通过 ``free -h`` 查看)
   - 磁盘 I/O 使用率 (通过 ``iostat`` 查看)
   - 缓存命中率 (通过日志查看)
   - 请求延迟分布

常见问题
--------

**Q: 启动时报错 "Failed to import LightMem library" 怎么办?**

A: 这表示没有安装 LightMem 库或安装失败。请按以下步骤排查:

1. 检查是否安装了 LightMem:

   .. code-block:: bash

       python -c "import light_mem"

2. 如果报 ImportError,请重新安装:

   .. code-block:: bash

       cd /path/to/lightllm/LightMem
       pip install -v .

3. 检查系统依赖是否完整:

   .. code-block:: bash

       # Ubuntu/Debian
       sudo apt-get install cmake build-essential libboost-all-dev
       
       # 或使用 Conda
       conda install -c conda-forge cmake cxx-compiler boost libboost-devel

4. 如果编译失败,检查 CMake 版本 (需要 >= 3.25):

   .. code-block:: bash

       cmake --version

**Q: 如何判断是否需要启用多级缓存?**

A: 如果遇到以下情况,建议启用多级缓存:

- GPU 显存不足,无法处理长文本或高并发
- 需要大量缓存历史对话上下文
- 希望降低 GPU 成本,用更便宜的内存/存储替代

**Q: CPU 缓存和磁盘缓存的容量如何规划?**

A: 容量规划参考:

- GPU 显存: 每 GB 约缓存 ~13K tokens (取决于模型)
- CPU 内存: 容量与 GPU 相同
- 磁盘存储: 容量与 GPU 相同
- 建议比例: L1:L2:L3 = 1:5-10:20-50

**Q: 启用多级缓存会影响延迟吗?**

A: 会有一定影响:

- L1 (GPU) 访问: ~1ms
- L2 (CPU) 访问: ~10-50ms (首次加载)
- L3 (Disk) 访问: ~100-500ms (首次加载)
- 后续访问会缓存到更高层级,延迟降低

**Q: 磁盘缓存目录可以在多个实例间共享吗?**

A: 不建议。每个实例应使用独立的缓存目录,以避免数据冲突和一致性问题。多机部署场景除外(需要共享存储支持)。

**Q: 如何清理缓存数据?**

A: 停止服务后,直接删除 ``disk_cache_dir`` 目录即可:

.. code-block:: bash

    rm -rf /mnt/ssd/disk_cache_dir

CPU 内存缓存会在进程结束后自动释放。

**Q: Hugepages 配置失败怎么办?**

A: Hugepages 配置需要 root 权限和足够的连续内存:

.. code-block:: bash

    # 查看当前配置
    cat /proc/meminfo | grep Huge
    
    # 清理内存缓存后重试
    echo 3 > /proc/sys/vm/drop_caches
    echo 200000 > /proc/sys/vm/nr_hugepages

如果仍然失败,可以不使用 Hugepages,性能会略有下降但不影响基本功能。

相关文档
--------

- :doc:`api_server_args_zh`: 完整的 API Server 参数说明
- :doc:`deepseek_deployment`: DeepSeek 模型部署指南
- :doc:`../getting_started/quickstart`: 快速开始指南
- `LightMem GitHub <https://github.com/ModelTC/LightMem>`_: LightMem 库源码和详细文档
