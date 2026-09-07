Gemma4 hybrid 缓存参数
=====================

Gemma4 的 full-attention 层保留 token 粒度的 KV，sliding-attention 层使用
GPU 请求窗口及临时 chunk 区域。大小页继续沿用现有 linear hybrid 的虚拟
token 边界和匹配规则；窗口运行态与缓存快照分别存储。

显存容量
--------

当前 sliding 的小页快照在 GPU 上，不同于 linear 小页的 CPU 状态池。
一个快照包含所有物理 sliding KV 层的窗口，其每卡大小为：

.. code-block:: text

   物理 sliding 层数 × window × 2 × 每卡 KV heads × head_dim × dtype 字节数

共享 KV 的逻辑层不重复计费。例如 Gemma4-31B、TP4、BF16 下，一个窗口快照为：

.. code-block:: text

   50 × 1024 × 2 × 4 × 256 × 2 = 200 MiB / GPU

需要分别考虑以下容量：

- ``--running_max_req_size`` 决定请求窗口数量，另有一个 hold request 窗口。
- ``--batch_max_tokens`` 等参数决定临时 chunk 区域大小，不能忽略长 chunk 的显存。
- ``--linear_att_cache_size`` 是小页状态槽位数，不是 token 数。
- 大页状态池、full KV、模型权重及计算工作区也占用显存。

在 DP1 下，未指定 ``linear_att_cache_size`` 时，它默认是
``running_max_req_size`` 的两倍。上述 31B 配置若使用 256 个请求槽位，
仅请求窗口和 512 个小页就约需 150 GiB，尚未包含权重和临时 chunk 区域。
因此应显式按显存预算设置这两个参数；不要直接套用 full-attention 模型的并发槽位配置。
内存管理器会计入大小页池并检查容量，不会静默缩减用户指定的小页数量。

切分与性能比较
--------------

- ``--linear_att_hash_page_size`` 是当前 hybrid 树的 hash 分块粒度，默认 512。
- 大页覆盖的 token 数是
  ``linear_att_hash_page_size * linear_att_page_block_num``。
  例如 hash 为 512、block num 为 32 时，大页为 16384 token。
- ``--chunked_prefill_size`` 是单轮上限，不是每轮固定长度。
  当前流程还会在大页边界和请求尾部 checkpoint 处截断。
- 请求尾部 checkpoint 为 ``floor((prompt_len - 1) / hash_page_size) * hash_page_size``。
  不会在每一个 hash 分块末尾都保存请求快照；尾部 checkpoint 若落在 chunk 内，
  可能多产生一轮 prefill。

性能比较应报告 chunk、大小页参数、缓存实际命中长度和 CUDA Graph 配置。
256-token chunk、32-token hash 等边界压力配置不能作为常规部署速度的唯一基线。
大页 16384 与 chunk 4096/8192 在从零开始时对齐，但尾部 checkpoint 仍可能额外截断。

当前能力边界
------------

本实现尚不支持 CPU cache、PD、MTP、量化 KV、DP prompt-cache fetch 和 diverse mode，
并要求启用 chunked prefill。带跨层 KV sharing 的配置不支持 microbatch overlap：
临时 KV 区域尚未按微批隔离。这里的 microbatch overlap 不包括普通的 CPU/GPU 调度重叠。

任意长度的 request-level 小页、输入与输出双 checkpoint，以及纯 sliding 模型的
提前淘汰策略不属于本次 hybrid 接入的范围。
