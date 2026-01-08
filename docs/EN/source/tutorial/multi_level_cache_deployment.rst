.. _multi_level_cache_deployment:

Multi-Level Cache Deployment Guide
===================================

LightLLM supports a multi-level KV Cache mechanism. By combining three cache levels - GPU (L1), CPU (L2), and Disk (L3) - it can significantly reduce deployment costs and improve throughput for long-text scenarios. This document provides detailed instructions on configuring and using the multi-level cache functionality.

Prerequisites
-------------

Using L3 cache requires the **LightMem** library. LightMem is a high-performance KV Cache disk management library specifically designed for large language model inference systems.

.. note::
   
   If you only use two-level caching (L1 + L2), i.e., GPU + CPU cache, LightMem is not required.
   LightMem support is only needed when the ``--enable_disk_cache`` parameter is enabled.

Installing LightMem
~~~~~~~~~~~~~~~~~~~

**Source Code Location:**

- https://github.com/ModelTC/LightMem

**Installation:**

- For detailed installation instructions, refer to the LightMem repository's `README documentation <https://github.com/ModelTC/LightMem/blob/main/README.md>`_.

Multi-Level Cache Architecture
-------------------------------

LightLLM's multi-level cache system adopts a hierarchical design:

- **L1 Cache (GPU Memory)**: The fastest cache layer, storing hot request KV Cache with the lowest latency
- **L2 Cache (CPU Memory)**: Medium-speed cache layer, storing relatively cold KV Cache at lower cost than GPU
- **L3 Cache (Disk Storage)**: Maximum capacity cache layer, storing long-term inactive KV Cache at the lowest cost

**Working Principle:**

1. The current mechanism creates an exact backup copy of GPU cache data in CPU cache, not just storing content that doesn't fit in GPU cache
2. L1, L2, and L3 caches all use LRU eviction strategy for data management
3. To avoid frequent disk writes in L3 cache, you can use the LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH environment variable to control the minimum length threshold for writes. If set to 0, all L2 data will be written to L3 cache
4. During queries, L1 is checked first to find the longest matching prefix, then L2 is queried to continue extending the longest matching prefix, and finally L3 is queried for the remaining part

**Applicable Scenarios:**

- Ultra-long text processing (e.g., million-token level context)
- High-concurrency conversation scenarios (requiring caching of large amounts of conversation history)
- Cost-sensitive deployments (replacing expensive GPU memory with cheaper RAM and disk)
- Prompt Cache scenarios (reusing common prompt prefixes)

Deployment Solutions
--------------------

1. L1 + L2 Two-Level Cache (GPU + CPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for most scenarios, significantly increasing cache capacity while maintaining high performance.

**Startup Command:**

.. code-block:: bash

    # Enable GPU + CPU two-level cache
    LOADWORKER=18 python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3-235B-A22B \
        --tp 8 \
        --graph_max_batch_size 500 \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend fa3 \
        --mem_fraction 0.88 \
        --enable_cpu_cache \
        --cpu_cache_storage_size 400 \
        --cpu_cache_token_page_size 64

**Parameter Description:**

Basic Parameters
^^^^^^^^^^^^^^^^

- ``LOADWORKER=18``: Number of model loading threads to speed up model loading. Recommended to set to half the number of CPU cores
- ``--model_dir``: Model file path, supports local path or HuggingFace model name
- ``--tp 8``: Tensor parallelism degree, using 8 GPUs for model inference
- ``--graph_max_batch_size 500``: CUDA Graph maximum batch size, affects throughput and memory usage
- ``--llm_prefill_att_backend fa3``: Enable Flash Attention 3.0 to improve attention computation speed. You can also switch to flashinfer backend for better performance
- ``--mem_fraction 0.88``: GPU memory usage ratio, recommended to set to 0.88 or below

CPU Cache Parameters
^^^^^^^^^^^^^^^^^^^^

- ``--enable_cpu_cache``: **Enable CPU cache** (L2 layer), the core parameter for enabling two-level cache
- ``--cpu_cache_storage_size 400``: **CPU cache capacity** in GB, set to 400GB here
  
  - Capacity planning: Every 2GB can cache approximately 10K tokens of KV Cache (depending on model configuration)
  - Recommended to set to 50~60% of available system memory
  - For machines with 2TB memory, recommended to set to 1~1.2TB

- ``--cpu_cache_token_page_size 64``: **CPU cache page size** in number of tokens
  
  - Default value is 256, recommended range 64-512
  - Smaller page sizes (e.g., 64) are suitable for fine-grained cache management, reducing memory fragmentation and improving hit rates
  - Larger page sizes (e.g., 256) are suitable for bulk data migration, improving transfer efficiency
  - This value needs to balance memory utilization and transfer overhead

**Performance Optimization Suggestions:**

1. **Using Hugepages**: Execute the following commands and set the LIGHTLLM_HUGE_PAGE_ENABLE environment variable to enable huge page mode. Enabling huge page memory can significantly improve service startup speed. If you find the service takes too long to start, you can enable huge page mode for acceleration. Note that huge page mode will occupy memory space for the long term

   .. code-block:: bash

        sudo sed -i 's/^GRUB_CMDLINE_LINUX=\"/& default_hugepagesz=1G \
        hugepagesz=1G hugepages={required_huge_page_capacity}/' /etc/default/grub
        sudo update-grub
        sudo reboot

2. L1 + L2 + L3 Three-Level Cache (GPU + CPU + Disk)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for ultra-long text or extremely high-concurrency scenarios, providing maximum cache capacity.

.. important::
   
   Using three-level cache requires installing the **LightMem** library first. Please refer to the "Prerequisites" section above to complete the installation.

**Startup Command:**

.. code-block:: bash

    # Enable GPU + CPU + Disk three-level cache
    LOADWORKER=18 python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3-235B-A22B \
        --tp 8 \
        --graph_max_batch_size 500 \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend fa3 \
        --mem_fraction 0.88 \
        --enable_cpu_cache \
        --cpu_cache_storage_size 400 \
        --cpu_cache_token_page_size 256 \
        --enable_disk_cache \
        --disk_cache_storage_size 1000 \
        --disk_cache_dir /mnt/ssd/disk_cache_dir

**Parameter Description:**

Disk Cache Parameters
^^^^^^^^^^^^^^^^^^^^^

In addition to the two-level cache, add the following parameters:

- ``--enable_disk_cache``: **Enable disk cache** (L3 layer), the core parameter for enabling three-level cache
- ``--disk_cache_storage_size 1000``: **Disk cache capacity** in GB, set to 1TB here
  
  - Capacity planning: Every 2GB can cache approximately 10K tokens of KV Cache
  - Recommended to set based on storage space and business needs, typically ranging from hundreds of GB to several TB
  - 1TB capacity can cache approximately 5M tokens of KV Cache

- ``--disk_cache_dir /mnt/ssd/disk_cache_dir``: **Disk cache directory**, specifying the directory for persisting cache data
  
  - If not set, the system temporary directory will be used
  - Strongly recommended to use SSD/NVMe storage, avoid using HDD (performance difference can be 10-100x)
  - Ensure the directory has sufficient read/write permissions and disk space

Related Documentation
---------------------

- `LightMem GitHub <https://github.com/ModelTC/LightMem>`_: LightMem library source code and detailed documentation
