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
    --tp 8 \
    --enable_fa3

**参数说明:**
- `LOADWORKER=18`: 模型加载线程数，提高加载速度
- `--tp 8`: 张量并行度，使用8个GPU
- `--enable_fa3`: 启用 Flash Attention 3.0
- `--port 8088`: 服务端口

1.2 单机 DP + EP 模式 (Data Parallel + Expert Parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

适用于 DeepSeek-V2/V3 等 MoE 模型的专家并行部署。

**启动命令:**

.. code-block:: bash

    # H200 单机 DeepSeek-R1 DP + EP 模式
    MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 8 \
    --dp 8 \
    --enable_fa3

**参数说明:**
- `MOE_MODE=EP`: 设置专家并行模式
- `--tp 8`: 张量并行度
- `--dp 8`: 数据并行度，通常设置为与 tp 相同的值
- `--enable_fa3`: 启用 Flash Attention 3.0

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
    --enable_fa3 \
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
    --enable_fa3 \
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
    MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 16 \
    --dp 16 \
    --enable_fa3 \
    --nnodes 2 \
    --node_rank 0 \
    --nccl_host $nccl_host \
    --nccl_port 2732

**Node 1 启动命令:**

.. code-block:: bash

    # H200 多机 DeepSeek-R1 EP 模式 Node 1
    # 使用方法: sh multi_node_ep_node1.sh <nccl_host>
    export nccl_host=$1
    MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 16 \
    --dp 16 \
    --enable_fa3 \
    --nnodes 2 \
    --node_rank 1 \
    --nccl_host $nccl_host \
    --nccl_port 2732

**可选优化参数:**
- `--enable_prefill_microbatch_overlap`: 启用预填充微批次重叠
- `--enable_decode_microbatch_overlap`: 启用解码微批次重叠

3. PD 分离部署方案
------------------

PD (Prefill-Decode) 分离模式将预填充和解码阶段分离部署，可以更好地利用硬件资源。

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
    # nvidia-cuda-mps-control -d，运行MPS(可选, 有mps支持性能会好特别多，但是部分显卡和驱动环境开启mps会容易出现错误，建议升级驱动到较高版本，特别是H系列卡)

    export host=$1
    export pd_master_ip=$2
    nvidia-cuda-mps-control -d 
    MOE_MODE=EP KV_TRANS_USE_P2P=1 LOADWORKER=18 python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "prefill" \
    --tp 8 \
    --dp 8 \
    --host $host \
    --port 8019 \
    --nccl_port 2732 \
    --enable_fa3 \
    --disable_cudagraph \
    --pd_master_ip $pd_master_ip \
    --pd_master_port 60011
    # 如果需要启用微批次重叠，可以取消注释以下行
    #--enable_prefill_microbatch_overlap

**步骤 3: 启动 Decode 服务**

.. code-block:: bash

    # PD decode 模式 for DeepSeek-R1 (DP+EP) on H200
    # 使用方法: sh pd_decode.sh <host> <pd_master_ip>
    export host=$1
    export pd_master_ip=$2
    nvidia-cuda-mps-control -d
    MOE_MODE=EP KV_TRANS_USE_P2P=1 LOADWORKER=18 python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "decode" \
    --tp 8 \
    --dp 8 \
    --host $host \
    --port 8121 \
    --nccl_port 12322 \
    --enable_fa3 \
    --disable_cudagraph \
    --pd_master_ip $pd_master_ip \
    --pd_master_port 60011
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
    MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "prefill" \
    --host $host \
    --port 8019 \
    --tp 8 \
    --dp 8 \
    --nccl_port 2732 \
    --enable_fa3 \
    --disable_cudagraph \
    --config_server_host $config_server_host \
    --config_server_port 60088
    # 如果需要启用微批次重叠，可以取消注释以下行
    #--enable_prefill_microbatch_overlap

    # Decode 服务
    export host=$1
    export config_server_host=$2
    nvidia-cuda-mps-control -d
    MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "decode" \
    --host $host \
    --port 8121 \
    --nccl_port 12322 \
    --tp 8 \
    --dp 8 \
    --enable_fa3 \
    --config_server_host $config_server_host \
    --config_server_port 60088
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
    python benchmark_client.py \
    --num_clients 100 \
    --input_num 2000 \
    --tokenizer_path /path/DeepSeek-R1/ \
    --url http://127.0.0.1:8088/generate_stream

以上所有脚本可以参考 `test/start_scripts/multi_pd_master/` 目录下的脚本。