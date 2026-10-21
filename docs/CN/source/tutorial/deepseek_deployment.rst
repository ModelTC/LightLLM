.. _deepseek_deployment:

DeepSeek 模型部署指南
=====================

LightLLM 支持多种 DeepSeek 模型的部署方案，包括 DeepSeek-R1、DeepSeek-V2、DeepSeek-V3 等。本文档详细介绍各种部署模式和配置方案。

部署模式概览
------------

LightLLM 支持以下几种部署模式：

1. **单机 TP 模式**: 使用张量并行在单机上部署
2. **单机 EP 模式**: 使用专家并行在单机上部署
3. **多机 TP 模式**: 跨多台机器使用张量并行
4. **多机 EP 模式**: 跨多台机器使用专家并行
5. **PD 分离模式**: 将预填充和解码分离部署
6. **多 PD Master 模式**: 支持多个 PD Master 节点

1. 单机部署方案
---------------

1.1 单机 TP 模式 (Tensor Parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

适用于单台 H200 机器部署 DeepSeek-R1 模型。

**启动命令:**

.. code-block:: bash

    # H200 单机 DeepSeek-R1 TP 模式
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 8

**参数说明:**
- `LOADWORKER=18`: 模型加载线程数，提高加载速度
- `--tp 8`: 张量并行度，使用8个GPU
- `--port 8088`: 服务端口

1.2 单机 DP + EP 模式 (Data Parallel + Expert Parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

适用于 DeepSeek-V2/V3 等 MoE 模型的专家并行部署。

**启动命令:**

.. code-block:: bash

    # H200 单机 DeepSeek-R1 DP + EP 模式
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 8 \
    --dp 8 \
    --enable_ep_moe

**参数说明:**
- `--enable_ep_moe`: 设置专家并行模式
- `--tp 8`: 张量并行度
- `--dp 8`: 数据并行度，通常设置为与 tp 相同的值

**可选优化参数:**
- `--enable_prefill_microbatch_overlap`: 启用预填充微批次重叠
- `--enable_decode_microbatch_overlap`: 启用解码微批次重叠

2. 多机部署方案
---------------

2.1 多机 TP 模式
~~~~~~~~~~~~~~~~

适用于跨多台 H200/H100 机器部署。

**Node 0 启动命令:**

.. code-block:: bash

    # H200/H100 多机 DeepSeek-R1 TP 模式 Node 0
    # 使用方法: sh multi_node_tp_node0.sh <nccl_host>
    export nccl_host=$1
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 16 \
    --nnodes 2 \
    --node_rank 0 \
    --nccl_host $nccl_host \
    --nccl_port 2732

**Node 1 启动命令:**

.. code-block:: bash

    # H200/H100 多机 DeepSeek-R1 TP 模式 Node 1
    # 使用方法: sh multi_node_tp_node1.sh <nccl_host>
    export nccl_host=$1
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 16 \
    --nnodes 2 \
    --node_rank 1 \
    --nccl_host $nccl_host \
    --nccl_port 2732

**参数说明:**
- `--nnodes 2`: 总节点数
- `--node_rank 0/1`: 当前节点排名
- `--nccl_host`: NCCL 通信主机地址
- `--nccl_port 2732`: NCCL 通信端口

2.2 多机 EP 模式
~~~~~~~~~~~~~~~~

适用于跨多台机器部署 MoE 模型。

**Node 0 启动命令:**

.. code-block:: bash

    # H200 多机 DeepSeek-R1 EP 模式 Node 0
    # 使用方法: sh multi_node_ep_node0.sh <nccl_host>
    export nccl_host=$1
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 16 \
    --dp 16 \
    --nnodes 2 \
    --node_rank 0 \
    --nccl_host $nccl_host \
    --nccl_port 2732 --enable_ep_moe

**Node 1 启动命令:**

.. code-block:: bash

    # H200 多机 DeepSeek-R1 EP 模式 Node 1
    # 使用方法: sh multi_node_ep_node1.sh <nccl_host>
    export nccl_host=$1
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 16 \
    --dp 16 \
    --nnodes 2 \
    --node_rank 1 \
    --nccl_host $nccl_host \
    --nccl_port 2732 --enable_ep_moe

**可选优化参数:**
- `--enable_prefill_microbatch_overlap`: 启用预填充微批次重叠
- `--enable_decode_microbatch_overlap`: 启用解码微批次重叠

3. PD 分离部署方案
------------------

PD (Prefill-Decode) 分离模式将预填充和解码阶段分离部署，可以更好地利用硬件资源。

.. warning::

    如果宿主机或容器设置了 ``HTTP_PROXY``、``HTTPS_PROXY`` 或 ``ALL_PROXY``，
    Prefill/Decode 节点到 PD Master 的 WebSocket 注册连接可能被发送到代理服务器。
    典型日志为 ``proxy rejected connection: HTTP 403``。此时 P/D 节点自身的
    HTTP 服务可能已正常启动，但 PD Master 仍显示 0 个已注册节点，
    ``/readiness``、``/health`` 和 ``/healthz`` 会返回 HTTP 503。

    启动服务前，应将所有 PD Master、Prefill、Decode 和 Config Server 的内网 IP
    或主机名加入 ``NO_PROXY``，并同时设置小写的 ``no_proxy`` 以兼容不同网络库。

    .. code-block:: bash

        export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${pd_master_ip},${host},127.0.0.1,localhost"
        export no_proxy="$NO_PROXY"

    Kubernetes 部署还应包含实际使用的 Service DNS 名和 Service 后缀，例如
    ``pd-master.default.svc,.default.svc,.svc``。不要只配置 ``localhost``：跨 Pod 通信使用的
    Service DNS 或 Pod IP 也必须绕过代理。

3.1 单 PD Master 模式
~~~~~~~~~~~~~~~~~~~~~

**步骤 1: 启动 PD Master 服务**

.. code-block:: bash

    # PD Master for DeepSeek-R1
    # 使用方法: sh pd_master.sh <pd_master_ip>
    export pd_master_ip=$1
    python -m lightllm.server.api_server --model_dir /path/DeepSeek-R1 \
    --run_mode "pd_master" \
    --host $pd_master_ip \
    --port 60011

**步骤 2: 启动 Prefill 服务**

.. code-block:: bash

    # PD prefill 模式 for DeepSeek-R1 (DP+EP) on H200
    # 使用方法: sh pd_prefill.sh <host> <pd_master_ip>
    # 默认使用 NCCL 数据面；如需使用 NIXL，请为 prefill 和 decode 统一设置 --pd_trans_mode nixl
    # nvidia-cuda-mps-control -d，运行MPS(可选, 有mps支持性能会好特别多，但是部分显卡和驱动环境开启mps会容易出现错误，建议升级驱动到较高版本，特别是H系列卡)

    export host=$1
    export pd_master_ip=$2
    nvidia-cuda-mps-control -d 
    LOADWORKER=18 python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "prefill" \
    --pd_trans_mode nccl \
    --tp 8 \
    --dp 8 \
    --host $host \
    --port 8019 \
    --nccl_port 2732 \
    --disable_cudagraph \
    --pd_master_ip $pd_master_ip \
    --pd_master_port 60011 \
    --enable_ep_moe
    # 如果需要启用微批次重叠，可以取消注释以下行
    #--enable_prefill_microbatch_overlap

**步骤 3: 启动 Decode 服务**

.. code-block:: bash

    # PD decode 模式 for DeepSeek-R1 (DP+EP) on H200
    # 使用方法: sh pd_decode.sh <host> <pd_master_ip>
    # 默认使用 NCCL 数据面；如需使用 NIXL，请为 prefill 和 decode 统一设置 --pd_trans_mode nixl
    export host=$1
    export pd_master_ip=$2
    nvidia-cuda-mps-control -d
    LOADWORKER=18 python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "decode" \
    --pd_trans_mode nccl \
    --tp 8 \
    --dp 8 \
    --host $host \
    --port 8121 \
    --nccl_port 12322 \
    --disable_cudagraph \
    --pd_master_ip $pd_master_ip \
    --pd_master_port 60011 \
    --enable_ep_moe
    # 如果需要启用微批次重叠，可以取消注释以下行
    #--enable_decode_microbatch_overlap

3.2 多 PD Master 模式
~~~~~~~~~~~~~~~~~~~~~

支持多个 PD Master 节点，提供更好的负载均衡和高可用性。

**步骤 1: 启动 Config Server**

.. code-block:: bash

    # Config Server
    # 使用方法: sh config_server.sh <config_server_host>
    export config_server_host=$1
    python -m lightllm.server.api_server \
    --run_mode "config_server" \
    --config_server_host $config_server_host \
    --config_server_port 60088

**步骤 2: 启动多个 PD Master**

.. code-block:: bash

    # PD Master 1
    # 使用方法: sh pd_master_1.sh <host> <config_server_host>
    export host=$1
    export config_server_host=$2
    python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "pd_master" \
    --host $host \
    --port 60011 \
    --config_server_host $config_server_host \
    --config_server_port 60088

    # PD Master 2
    # 使用方法: sh pd_master_2.sh <host> <config_server_host>
    export host=$1
    export config_server_host=$2
    python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "pd_master" \
    --host $host \
    --port 60012 \
    --config_server_host $config_server_host \
    --config_server_port 60088

**步骤 3: 启动 Prefill 和 Decode 服务**

.. code-block:: bash

    # Prefill 服务
    export host=$1
    export config_server_host=$2
    nvidia-cuda-mps-control -d
    LOADWORKER=18 python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "prefill" \
    --pd_trans_mode nccl \
    --host $host \
    --port 8019 \
    --tp 8 \
    --dp 8 \
    --nccl_port 2732 \
    --disable_cudagraph \
    --config_server_host $config_server_host \
    --config_server_port 60088 \
    --enable_ep_moe
    # 如果需要启用微批次重叠，可以取消注释以下行
    #--enable_prefill_microbatch_overlap

    # Decode 服务
    export host=$1
    export config_server_host=$2
    nvidia-cuda-mps-control -d
    LOADWORKER=18 python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "decode" \
    --pd_trans_mode nccl \
    --host $host \
    --port 8121 \
    --nccl_port 12322 \
    --tp 8 \
    --dp 8 \
    --config_server_host $config_server_host \
    --config_server_port 60088 \
    --enable_ep_moe
    # 如果需要启用微批次重叠，可以取消注释以下行
    #--enable_decode_microbatch_overlap

4. 测试和验证
-------------

4.1 基础功能测试
~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl http://server_ip:server_port/generate \
         -H "Content-Type: application/json" \
         -d '{
               "inputs": "What is AI?",
               "parameters":{
                 "max_new_tokens":17, 
                 "frequency_penalty":1
               }
              }'

4.2 性能基准测试
~~~~~~~~~~~~~~~~

.. code-block:: bash

    # DeepSeek-R1 性能测试
    cd test
    python benchmark/service/benchmark_client.py \
    --num_clients 100 \
    --input_num 2000 \
    --tokenizer_path /path/DeepSeek-R1/ \
    --url http://127.0.0.1:8088/generate_stream

以上所有脚本可以参考 `test/start_scripts/multi_pd_master/` 目录下的脚本。

5. DeepSeek-V4.1-Flash 文本支持
-----------------------------

V4.1 通过模型注册表选择 ``DeepseekV41TpPartModel``，复用 V4 的请求分页、
请求级 SWA 状态、FlashMLA attention、MoE、TP 通信和服务入口。模型差异集中在
独立的 ``lightllm/models/deepseek_v41/`` 中；通过继承和导入复用 V4 的已有实现：

* C2 非重叠门控压缩与 C1 投影；仅配置声明的 source layer 分配全局 KV/indexer K。
* 层间复用 sparse selection，以及 candidate block + top-k 两级 indexer。
* single-pass mHC：消费前一个子层的 pre 系数，输出当前子层的 pre 系数。
* Engram 使用 CPU pinned FP8 表、GPU UVA gather 和现有 TP all-reduce；
  哈希历史跟随完整 token 槽位，GPU 前缀命中时复用对应历史。

Hopper 的 dense 投影保留 FP8 block-32 权重，复用 Triton FP8×FP8 GEMM，输入按 group-32 动态量化；
MXFP4 专家通过现有量化选择器使用 vLLM Marlin W4A16，并补齐 TP 切分后的非对齐维度。
对于转换后的 FP8 block-128 checkpoint，dense 投影和专家复用已有 DeepGEMM FP8×FP8 路径，
UE8M0 权重 scale 由现有加载器读取。
主 attention 复用 V4 的 584-byte FP8/BF16 cache，SWA 复用请求管理器的 RMSNorm/RoPE/FP8 融合写入。
压缩 KV 通过一个 kernel 完成 RoPE 和真实 FP4 打包（per-16 E4M3 scale），写入时由同一 scatter kernel
转换为 Hopper FlashMLA 所需的 FP8/BF16 格式，不再生成独立的 BF16 假量化张量。
indexer Q/K 使用真实 packed MXFP4（per-32 UE8M0 scale），K 每条存储 64-byte 数据和 4-byte scale；
Hopper scorer 在寄存器中解包后执行 BF16 点积，不进行额外的 FP8 K 量化。
主 KV 的 cache 兼容路径仍有额外的量化误差；MXFP4 checkpoint 的 Marlin 专家使用 BF16 激活，
与原生 FP8 路径不同。目前不能据此声称与原生 FP4 cache 等精度或等性能。

依赖现有 FlashMLA，以及提供 MXFP4 ``fused_marlin_moe``、``clamp_limit`` 和
``get_accelerator_view_from_cpu_tensor`` 的 vLLM。
safetensors 需要支持 checkpoint 中的 ``F8_E8M0`` 类型（实测 0.8.0 可读取，0.4.5 无法读取）。
模型目录需要包含官方 ``encoding/encoding.py`` 和 tokenizer 文件。
两个 Engram 表合计约需 203 GB（约 189 GiB）CPU pinned memory（TP 分摊），加载专家时另有 CPU staging 开销。

单机 TP8 文本启动示例（在已具备上述依赖的环境中）：

.. code-block:: bash

    cd /mtc/wanzihao/lightllm-dsv41
    export PYTHONPATH=$PWD
    export LOADWORKER=16
    export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
    ulimit -Sn 1048576
    python -m lightllm.server.api_server \
        --model_dir /path/DeepSeek-V4.1-Flash \
        --tp 8 \
        --batch_max_tokens 8192 \
        --chunked_prefill_size 8192 \
        --port 18902

Engram 表使用按实际字节数分配的内存映射，预触页后注册为 pinned memory，
避免 PyTorch pinned allocator 将每张约 23.6 GiB 的 TP4 分片向上取整到 32 GiB。
DP ranks 通过现有服务共享内存机制复用同一节点上的 TP 分片，每张表仅加载一次。

当前开放 normal text TP 和 DP/EP，支持 chunked prefill、GPU prefix cache、decode 和 decode CUDA Graph。
GPU prefix cache 复用现有 hybrid checkpoint；256-token 边界已闭合 C2 分组，快照仅需 SWA 状态。
暂不支持视觉、PD、DSpark/MTP、CPU cache、prefill CUDA Graph、microbatch overlap 和 TP/SP 混合模式，
启动时会检查不兼容配置。CED decoder SWA bounded replay 在最后一个 KV source 层之后，
将每个请求的后续层输入缩减为本次 prefill 的最后 128 个 token，并将 SWA 可见范围限制在回放段内。
最后一个 KV source 层仍处理全部输入，完整写入全局 KV；请求分页和 prefix cache 的推进保持原有路径。
该回放近似重建 decoder SWA 状态；请求完整 prompt logprobs 时保留完整前向。
API 自动选择 V4.1 DSML parser；数值 ``reasoning_effort`` 范围为 1–100，
聊天模板使用官方编码器。流式工具调用会在完整 calls 块到齐后发送 JSON 参数。

双机各 8 卡 H100 的 DP16/EP16 使用 ``tools/launch_deepseek_v41_dpep16.sh``：
在两台机器的同一 checkout 上分别传入 ``0 NODE0_IP`` 和 ``1 NODE0_IP``。
脚本使用转换后的 FP8 block-128 checkpoint；``--tp 16 --dp 16 --enable_ep_moe``
对应 16 个全局 rank、每个 DP rank 内 attention TP1、专家 EP16。
跨机 legacy DeepEP low-latency 路径在 NVSHMEM 3.5.19+ 上需要
`DeepEP PR #696 <https://github.com/deepseek-ai/DeepEP/pull/696>`_ 的 RC QP 索引修复。
本次 H100 环境使用 CUDA 13、PyTorch 2.11、NVSHMEM 3.6.5 和包含该修复的 DeepEP wheel；
脚本显式指定当前环境的 NCCL/NVSHMEM 路径，覆盖 wheel 中记录的构建环境路径。

验证用例：

.. code-block:: bash

    python -m pytest unit_tests/models/deepseek_v41/test_v41_model.py \
        unit_tests/server/test_deepseek_v41.py

GPU 用例覆盖 FP4 打包/舍入、跨 chunk C2、分页 causal/candidate indexer、KV source 共享、
MXFP4 专家补齐、Engram UVA/跨页哈希、DP 共享表和 shifted mHC；
小型 FP4/FP8 量化模型通过真实权重加载器，比较整段/分块 prefill、decode 以及 CUDA Graph replay 的 logits。
此前兼容路径使用完整 V4.1 Flash 权重在 4×H200、TP4、关闭 CUDA Graph 的配置下通过文本 smoke：
短算术、翻译、中文问答，以及 1220-token 输入的分块 prefill；四个请求均正常结束。
这些测试不替代精度评测或性能基准。

2026-10-21，转换后的 FP8 block-128/UE8M0 完整模型在双机 16×H100、DP16/EP16 上
通过 16 个 rank 的 decode CUDA Graph 捕获和两个节点的真实生成验证。
GSM8K 全量 test 1319 题、训练集 8-shot、raw completion、greedy、并发 256，
GPU prompt cache 开启，两个节点均分请求：1206/1319（91.43%），耗时 206.6 秒，
无请求错误、无答案解析失败、无长度截断。

开启 decoder bounded replay、移除固定 ``max_total_token_num`` 后，同配置热缓存回归
为 1218/1319（92.34%），耗时 61.8 秒，无请求错误、答案解析失败或截断。
本次自动容量估算按实际历史缓存布局 1656 bytes/token 计算，每个 DP rank 最终分配
12,318,976 个 token slots（约 19 GiB）；SWA 和压缩器状态在估算前已分配。
该容量估算使用剩余显存和 ``mem_fraction`` 预算，不包含完整 forward 峰值测量。
