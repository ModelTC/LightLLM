.. _deepseek_deployment:

DeepSeek Model Deployment Guide
===============================

LightLLM supports various deployment solutions for DeepSeek models, including DeepSeek-R1, DeepSeek-V2, DeepSeek-V3, etc. This document provides detailed information on various deployment modes and configuration solutions.

Deployment Mode Overview
-----------------------

LightLLM supports the following deployment modes:

1. **Single Machine TP Mode**: Deploy using tensor parallelism on a single machine
2. **Single Machine EP Mode**: Deploy using expert parallelism on a single machine
3. **Multi-Machine TP Mode**: Use tensor parallelism across multiple machines
4. **Multi-Machine EP Mode**: Use expert parallelism across multiple machines
5. **PD Separation Mode**: Separate prefill and decode deployment
6. **Multi PD Master Mode**: Support multiple PD Master nodes

1. Single Machine Deployment Solutions
-------------------------------------

1.1 Single Machine TP Mode (Tensor Parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for deploying DeepSeek-R1 model on a single H200 machine.

**Launch Command:**

.. code-block:: bash

    # H200 Single Machine DeepSeek-R1 TP Mode
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 8 \
    --enable_fa3

**Parameter Description:**
- `LOADWORKER=18`: Model loading thread count, improves loading speed
- `--tp 8`: Tensor parallelism degree, using 8 GPUs
- `--enable_fa3`: Enable Flash Attention 3.0
- `--port 8088`: Service port

1.2 Single Machine DP + EP Mode (Data Parallel + Expert Parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for expert parallelism deployment of MoE models like DeepSeek-V2/V3.

**Launch Command:**

.. code-block:: bash

    # H200 Single Machine DeepSeek-R1 DP + EP Mode
    MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 8 \
    --dp 8 \
    --enable_fa3

**Parameter Description:**
- `MOE_MODE=EP`: Set expert parallelism mode
- `--tp 8`: Tensor parallelism degree
- `--dp 8`: Data parallelism degree, usually set to the same value as tp
- `--enable_fa3`: Enable Flash Attention 3.0

**Optional Optimization Parameters:**
- `--enable_prefill_microbatch_overlap`: Enable prefill microbatch overlap
- `--enable_decode_microbatch_overlap`: Enable decode microbatch overlap

2. Multi-Machine Deployment Solutions
------------------------------------

2.1 Multi-Machine TP Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for deployment across multiple H200/H100 machines.

**Node 0 Launch Command:**

.. code-block:: bash

    # H200/H100 Multi-Machine DeepSeek-R1 TP Mode Node 0
    # Usage: sh multi_node_tp_node0.sh <nccl_host>
    export nccl_host=$1
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 16 \
    --enable_fa3 \
    --nnodes 2 \
    --node_rank 0 \
    --nccl_host $nccl_host \
    --nccl_port 2732

**Node 1 Launch Command:**

.. code-block:: bash

    # H200/H100 Multi-Machine DeepSeek-R1 TP Mode Node 1
    # Usage: sh multi_node_tp_node1.sh <nccl_host>
    export nccl_host=$1
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 16 \
    --enable_fa3 \
    --nnodes 2 \
    --node_rank 1 \
    --nccl_host $nccl_host \
    --nccl_port 2732

**Parameter Description:**
- `--nnodes 2`: Total number of nodes
- `--node_rank 0/1`: Current node rank
- `--nccl_host`: NCCL communication host address
- `--nccl_port 2732`: NCCL communication port

2.2 Multi-Machine EP Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for deploying MoE models across multiple machines.

**Node 0 Launch Command:**

.. code-block:: bash

    # H200 Multi-Machine DeepSeek-R1 EP Mode Node 0
    # Usage: sh multi_node_ep_node0.sh <nccl_host>
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

**Node 1 Launch Command:**

.. code-block:: bash

    # H200 Multi-Machine DeepSeek-R1 EP Mode Node 1
    # Usage: sh multi_node_ep_node1.sh <nccl_host>
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

**Optional Optimization Parameters:**
- `--enable_prefill_microbatch_overlap`: Enable prefill microbatch overlap
- `--enable_decode_microbatch_overlap`: Enable decode microbatch overlap

3. PD Separation Deployment Solutions
------------------------------------

PD (Prefill-Decode) separation mode separates prefill and decode stages for deployment, which can better utilize hardware resources.

3.1 Single PD Master Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Launch PD Master Service**

.. code-block:: bash

    # PD Master for DeepSeek-R1
    # Usage: sh pd_master.sh <pd_master_ip>
    export pd_master_ip=$1
    python -m lightllm.server.api_server --model_dir /path/DeepSeek-R1 \
    --run_mode "pd_master" \
    --host $pd_master_ip \
    --port 60011

**Step 2: Launch Prefill Service**

.. code-block:: bash

    # PD prefill mode for DeepSeek-R1 (DP+EP) on H200
    # Usage: sh pd_prefill.sh <host> <pd_master_ip>
    # nvidia-cuda-mps-control -d, run MPS (optional, performance will be much better with mps support, but some graphics cards and driver environments may encounter errors when enabling mps, it's recommended to upgrade to a higher driver version, especially for H-series cards)

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
    --pd_master_ip $pd_master_ip 