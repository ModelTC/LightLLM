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

5. DeepSeek-V4.1-Flash 支持
-------------------------

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
压缩 KV 在同一个 kernel 内完成 RoPE、FP8 量化和 scatter，直接写入 FlashMLA cache。
SWA 同样直接量化到最终缓存格式，不再先做一次 group-32 FP8 舍入再还原浮点。
indexer 复用 V4 的 FP8 Q 量化和 K 写入；V4.1 跳过 Hadamard，K 每条使用 128-byte FP8 数据
和一个 FP32 scale，scorer 直接执行 FP8×FP8 点积。Hopper 的 query tile 至少包含 64 个 head 槽位，
满足 FP8 warpgroup MMA 的形状要求；超出实际 head 数的槽位不参与结果。
attention/indexer 路径不再打包或解包 FP4。原始 MXFP4 checkpoint 的专家仍通过 Marlin 使用 BF16 激活；
转换后的 FP8 checkpoint 则按量化配置选择 FP8 专家实现。

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

当前开放 normal TP 和 DP/EP，支持文本、图像、chunked prefill、GPU prefix cache、decode 和 decode CUDA Graph。
GPU prefix cache 复用现有 hybrid checkpoint；256-token 边界已闭合 C2 分组，快照仅需 SWA 状态。
暂不支持 PD、DSpark 之外的 MTP、CPU cache、prefill CUDA Graph、microbatch overlap 和 TP/SP 混合模式，
启动时会检查不兼容配置。CED decoder SWA bounded replay 在最后一个 KV source 层之后，
将每个请求的后续层输入缩减为本次 prefill 的最后 128 个 token，并将 SWA 可见范围限制在回放段内。
最后一个 KV source 层仍处理全部输入，完整写入全局 KV；请求分页和 prefix cache 的推进保持原有路径。
该回放近似重建 decoder SWA 状态；请求完整 prompt logprobs 时保留完整前向。
API 自动选择 V4.1 DSML parser；数值 ``reasoning_effort`` 范围为 1–100，
聊天模板使用官方编码器。流式工具调用会在完整 calls 块到齐后发送 JSON 参数。

视觉复用 V4 的 ViT、aligner、embedding cache 和多模态 pre-layer；嵌套 ``vision_config``
由 V4.1 视觉模型在初始化时适配为 V4 接口参数。V4.1 的图像 token 按行排列，包含首尾标记和每行换行，
不使用 V4 的 N-layout 或 compressor padding，checkpoint 也没有 ``image_pad`` 参数。
ViT 内部使用双向 attention；图像 token 进入语言模型后使用因果 SWA，按视觉 bias 选择 MoE 专家。
图像位置不参与 Engram，也会阻断跨图像的 n-gram。启动时默认根据 checkpoint 开启视觉；
仅服务文本时可传 ``--disable_vision``。

完整 V4.1 checkpoint 的视觉权重已通过严格加载校验；固定相同 RoPE 表和 SDPA math 后端时，
640×480 图像经完整 ViT + aligner 得到的 192×5120 embedding 与官方实现逐元素一致。

DSpark 复用现有 V4 draft model 的加载、KV 提交、proposer 和调度器，V4.1 适配层仅替换
shifted-mHC proposal forward。Target 从配置声明的层输入收集 residual-stream 均值；draft 按
``dspark_n_routed_experts`` / ``dspark_num_experts_per_tok`` 配置专家，复用目标 embedding/head，
加载 checkpoint 中的 ``markov_head.embed/head`` 和 confidence head。
C2 压缩使用请求管理器已有的状态环，保留 speculative verify 后仍可能被接受的偶数位置。
在上述启动参数后追加：

.. code-block:: bash

    --mtp_mode dspark --mtp_step 5 --mtp_draft_model_dir /path/DeepSeek-V4.1-Flash

``mtp_step`` 不得超过 checkpoint 的 ``dspark_block_size``。当前公共 proposer 要求每个 prompt
token 都有 target hidden，因此开启 DSpark 时保留完整 prefill，不启用 CED bounded replay；
普通推理仍使用 CED。小模型已覆盖视觉与 DSpark 组合、target verify、draft proposal、
Markov/confidence 输出和 decode CUDA Graph。DP/EP 下，KV-only 提交不执行通信清理；
草稿与 target 专家数量不同时，完整 proposal 结束后恢复共享 DeepEP buffer 的 target 计数区布局，
该清理也包含在 decode CUDA Graph 中。

双机各 8 卡 H100、FP8 block-128 权重、固定 ``mtp_step=5`` 的完整模型验收：
GSM8K 8-shot、1319 题、并发 256、贪心解码得到 1207/1319（91.51%）；
接受 90607 个草稿 token / 验证 190465 个草稿 token（47.57%），
每轮平均接受 2.38 个草稿 token，含 target token 的平均接受长度为 3.38。
首轮耗时 97.84 秒；相同数据的热缓存复测为 1204/1319（91.28%）、接受率 48.04%、
耗时 34.12 秒，输入 token 缓存命中比例 85.7%，其中一题达到 1024-token 输出上限并计为错题。
8 张合成图片的数字、颜色和形状均回答正确，草稿接受率 40.87%，双图及多轮图片请求也已通过；
图像结果仅覆盖这些功能用例。

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
        unit_tests/models/deepseek_v41/test_v41_vision.py \
        unit_tests/server/test_deepseek_v41.py

GPU 用例覆盖直接 FP8 RoPE/cache 写入、跨 chunk C2、分页 causal/candidate indexer、KV source 共享、
MXFP4 专家补齐、Engram UVA/跨页哈希、DP 共享表和 shifted mHC；
小型 FP4/FP8 量化模型通过真实权重加载器，比较整段/分块 prefill、decode 以及 CUDA Graph replay 的 logits。
此前兼容路径使用完整 V4.1 Flash 权重在 4×H200、TP4、关闭 CUDA Graph 的配置下通过文本 smoke：
短算术、翻译、中文问答，以及 1220-token 输入的分块 prefill；四个请求均正常结束。
这些测试不替代精度评测或性能基准。

2026-10-21，切换直接 FP8 attention 之前，转换后的 FP8 block-128/UE8M0 完整模型在双机 16×H100、DP16/EP16 上
通过 16 个 rank 的 decode CUDA Graph 捕获和两个节点的真实生成验证。
GSM8K 全量 test 1319 题、训练集 8-shot、raw completion、greedy、并发 256，
GPU prompt cache 开启，两个节点均分请求：1206/1319（91.43%），耗时 206.6 秒，
无请求错误、无答案解析失败、无长度截断。

开启 decoder bounded replay、移除固定 ``max_total_token_num`` 后，同配置热缓存回归
为 1218/1319（92.34%），耗时 61.8 秒，无请求错误、答案解析失败或截断。
该次自动容量估算按旧历史缓存布局 1656 bytes/token 计算，每个 DP rank 最终分配
12,318,976 个 token slots（约 19 GiB）；SWA 和压缩器状态在估算前已分配。
该容量估算使用剩余显存和 ``mem_fraction`` 预算，不包含完整 forward 峰值测量。
直接 FP8 indexer 使用的历史缓存布局为 1800 bytes/token，自动容量估算随实际布局更新；
旧布局的缓存容量和 GSM8K 分数不代表切换后的结果。

切换为直接 FP8 attention/indexer 后，双机 H100 DP16/EP16 的实际缓存容量为每个 DP rank
11,333,376 个 token slots（约 19 GiB）。同样的 GSM8K 配置，冷前缀运行
1199/1319（90.90%）、52.5 秒，热前缀运行 1197/1319（90.75%）、49.7 秒，
均无请求错误、答案解析失败或截断。新结果低于之前一次 92.34% 的结果，尚未证明精度等价。
