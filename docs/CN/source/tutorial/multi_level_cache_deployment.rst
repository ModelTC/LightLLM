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

1. 缓存放置行为由 ``--cache_placement_strategy`` 控制，可选择兼容旧行为的逐层备份或自适应分层放置
2. L1、L2、L3 cache都基于LRU淘汰策略进行数据管理
3. 为了避免L3缓存频繁写盘，可通过LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH环境变量控制写入的最小长度阈值，如果设为0，则所有L2数据都会写入L3缓存
4. 查询时，会先查询L1找出命中的最长前缀，再去L2查询以继续增加最长匹配前缀，最后再去L3查询剩余部分

缓存放置策略
~~~~~~~~~~~~

``--cache_placement_strategy`` 用于选择请求完成后的 KV Cache 放置策略，可选值如下：

- ``adaptive``（默认）：冷启动阶段先收集 128 个请求，快速生成 GPU 与低层缓存路径之间的首个长度分界点；之后保留最近 512 个请求的滑动窗口，每 36 个放置步更新一次分界点。短请求放入 GPU，长请求放入 CPU；开启 Disk 时，长请求沿 CPU → Disk 路径异步落盘。由于 Disk 必须经过 CPU，计算比例时低层有效容量取 CPU 与 Disk 容量的较大值，而不是二者之和。默认只使用物理 GPU token 容量的 80% 进行放置估算，为运行态请求预留容量。首次小窗口尚未填满、没有可用分界点时，使用 ``legacy`` 行为完成冷启动。
- ``legacy``：兼容原有的逐层备份行为。请求始终保留在 GPU cache，同时复制到所有已开启的下级缓存；开启 CPU cache 时写入 GPU 和 CPU，同时开启 Disk cache 时写入 GPU、CPU 和 Disk。

未开启 ``--enable_cpu_cache`` 时，该参数不会改变运行行为，所有完成请求都只写入 GPU cache。

.. note::

   Disk cache 是通过 CPU cache 异步写入的，因此使用 Disk cache 时仍需同时开启 ``--enable_cpu_cache``。
   ``LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH`` 的最小写盘长度限制对两种策略都生效。

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
        --mem_fraction 0.88 \
        --enable_cpu_cache \
        --cpu_cache_storage_size 400 \
        --cpu_cache_token_page_size 64 \
        --cache_placement_strategy adaptive

**参数说明:**

基础参数
^^^^^^^^

- ``LOADWORKER=18``: 模型加载线程数,提高模型加载速度,建议设置为 CPU 核心数的一半
- ``--model_dir``: 模型文件路径,支持本地路径或 HuggingFace 模型名称
- ``--tp 8``: 张量并行度,使用 8 个 GPU 进行模型推理
- ``--graph_max_batch_size 500``: CUDA Graph 最大批次大小,影响吞吐量和显存占用
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

- ``--cache_placement_strategy adaptive``: **缓存放置策略**，默认使用自适应分层；如需保持原有的 GPU + CPU 逐层备份行为，设置为 ``legacy``
- ``LIGHTLLM_CACHE_PLACEMENT_GPU_CAPACITY_RATIO=0.8``: **GPU 容量估算比例**，adaptive 默认使用物理 GPU token 容量的 ``0.8`` 进行放置估算，合法范围为 ``(0, 1]``

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
        --mem_fraction 0.88 \
        --enable_cpu_cache \
        --cpu_cache_storage_size 400 \
        --cpu_cache_token_page_size 256 \
        --cache_placement_strategy adaptive \
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

如需使用兼容旧行为的三级逐层备份策略，将启动参数改为：

.. code-block:: bash

    --cache_placement_strategy legacy

相关文档
--------

- `LightMem GitHub <https://github.com/ModelTC/LightMem>`_: LightMem 库源码和详细文档
