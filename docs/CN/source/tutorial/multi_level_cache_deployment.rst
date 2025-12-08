.. _multi_level_cache_deployment:

多级缓存部署指南
================

LightLLM 支持多级 KV Cache 缓存机制,通过 GPU (L1)、CPU (L2) 和磁盘 (L3) 三级缓存的组合,可以大幅降低部署成本并提升长文本场景下的吞吐能力。本文档详细介绍如何配置和使用多级缓存功能。

前置依赖
--------

使用L3缓存需要安装 **LightMem** 库。LightMem 是一个高性能的 KV Cache 磁盘管理库，专为大语言模型推理系统设计。

.. note::
   
   如果只使用二级缓存 (L1 + L2)，即 GPU + CPU 缓存，则不需要安装 LightMem。
   只有启用 ``--enable_disk_cache`` 参数时才需要 LightMem 支持。

安装 LightMem
~~~~~~~~~~~~~

**源码位置:**

- https://github.com/ModelTC/LightMem

**安装:**

- 详细安装方式参考 LightMem 仓库的 `README 文档 <https://github.com/ModelTC/LightMem/blob/main/README.md>`_。

多级缓存架构
------------

LightLLM 的多级缓存系统采用分层设计:

- **L1 Cache (GPU 显存)**: 最快速的缓存层,存储热点请求的 KV Cache,提供最低延迟
- **L2 Cache (CPU 内存)**: 中速缓存层,存储相对较冷的 KV Cache,成本低于 GPU
- **L3 Cache (磁盘存储)**: 最大容量缓存层,存储长期不活跃的 KV Cache,成本最低

**工作原理:**

1. 目前的机制是会将GPU cache中的数据原模原样备份一份到CPU cache中，并非只存储GPU cache放不下的内容
2. L1、L2、L3 cache都基于LRU淘汰策略进行数据管理
3. 为了避免L3缓存频繁写盘，可通过LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH环境变量控制写入的最小长度阈值，如果设为0，则所有L2数据都会写入L3缓存
4. 查询时，会先查询L1找出命中的最长前缀，再去L2查询以继续增加最长匹配前缀，最后再去L3查询剩余部分

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
- ``--enable_fa3``: 启用 Flash Attention 3.0,提升注意力计算速度，也可以换成flashinfer后端性能更佳
- ``--mem_fraction 0.88``: GPU 显存使用比例,建议设置为 0.88及以下

CPU 缓存参数
^^^^^^^^^^^^

- ``--enable_cpu_cache``: **启用 CPU 缓存** (L2 层),这是开启二级缓存的核心参数
- ``--cpu_cache_storage_size 400``: **CPU 缓存容量**,单位为 GB,此处设置为 400GB
  
  - 容量规划: 每 2GB 大约可以缓存 10K tokens 的 KV Cache (取决于模型配置)
  - 建议设置为系统可用内存的 50~60%
  - 对于 2T 内存的机器,建议设置为1~1.2TB

- ``--cpu_cache_token_page_size 64``: **CPU 缓存页大小**,单位为 token 数量
  
  - 默认值为 256,建议范围 64-512
  - 较小的页大小 (如 64) 适合细粒度的缓存管理,减少内存碎片,提高命中率
  - 较大的页大小 (如 256) 适合大批量数据迁移,提高传输效率
  - 该值需要权衡内存利用率和传输开销

**性能优化建议:**

1. **使用 Hugepages**: 执行如下命令并设置环境变量LIGHTLLM_HUGE_PAGE_ENABLE可启用大页模式，启用大页内存可以显著提升服务启动速度，如果觉得服务启动太久可以开启大页模式加速，注意大页模式会长期占据内存空间

   .. code-block:: bash

        sudo sed -i 's/^GRUB_CMDLINE_LINUX=\"/& default_hugepagesz=1G \
        hugepagesz=1G hugepages={需要启用的大页容量}/' /etc/default/grub
        sudo update-grub
        sudo reboot

2. L1 + L2 + L3 三级缓存 (GPU + CPU + Disk)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

适合超长文本或极高并发场景,提供最大的缓存容量。

.. important::
   
   使用三级缓存需要先安装 **LightMem** 库,请参考上文"前置依赖"章节完成安装。

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
  
  - 容量规划: 每 2GB 大约可以缓存 10K tokens 的 KV Cache
  - 建议根据存储空间和业务需求设置,通常设置为数百 GB 到数 TB
  - 1TB 容量约可缓存 5M tokens 的 KV Cache

- ``--disk_cache_dir /mnt/ssd/disk_cache_dir``: **磁盘缓存目录**,指定用于持久化缓存数据的目录
  
  - 如果不设置,会使用系统临时目录
  - 强烈建议使用 SSD/NVMe 存储,避免使用 HDD (性能差距可达 10-100 倍)
  - 确保目录具有足够的读写权限和磁盘空间
  - 注意使用磁盘缓存时, 保证使用的SSD硬盘是长寿命的硬盘, 否则可能会快速消耗其使用寿命。

相关文档
--------

- `LightMem GitHub <https://github.com/ModelTC/LightMem>`_: LightMem 库源码和详细文档
