.. _deepseek_deployment:

DeepSeek Model Deployment Guide
===============================

LightLLM supports various deployment solutions for DeepSeek models, including DeepSeek-R1, DeepSeek-V2, DeepSeek-V3, etc. This document provides detailed information on various deployment modes and configuration solutions.

Deployment Mode Overview
------------------------

LightLLM supports the following deployment modes:

1. **Single node TP Mode**: Deploy using tensor parallelism on a single node
2. **Single node EP Mode**: Deploy using expert parallelism on a single node
3. **Multi-node TP Mode**: Use tensor parallelism across multiple nodes
4. **Multi-node EP Mode**: Use expert parallelism across multiple nodes
5. **PD disaggregation Mode**: Separate prefill and decode deployment
6. **Multi PD Master Mode**: Support multiple PD Master nodes

1. Single node Deployment Solutions
-------------------------------------

1.1 Single node TP Mode (Tensor Parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for deploying DeepSeek-R1 model on a single H200 node.

**Launch Command:**

.. code-block:: bash

    # H200 Single node DeepSeek-R1 TP Mode
    LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 8 \
    --enable_fa3

**Parameter Description:**
- `LOADWORKER=18`: Model loading thread count, improves loading speed
- `--tp 8`: Tensor parallelism, using 8 GPUs
- `--enable_fa3`: Enable Flash Attention 3.0
- `--port 8088`: Service port

1.2 Single node DP + EP Mode (Data Parallel + Expert Parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for expert parallelism deployment of MoE models like DeepSeek-V2/V3.

**Launch Command:**

.. code-block:: bash

    # H200 Single node DeepSeek-R1 DP + EP Mode
    MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8088 \
    --model_dir /path/DeepSeek-R1 \
    --tp 8 \
    --dp 8 \
    --enable_fa3

**Parameter Description:**
- `MOE_MODE=EP`: Set expert parallelism mode
- `--tp 8`: Tensor parallelism
- `--dp 8`: Data parallelism, usually set to the same value as tp
- `--enable_fa3`: Enable Flash Attention 3.0

**Optional Optimization Parameters:**
- `--enable_prefill_microbatch_overlap`: Enable prefill microbatch overlap
- `--enable_decode_microbatch_overlap`: Enable decode microbatch overlap

2. Multi-node Deployment Solutions
------------------------------------

2.1 Multi-node TP Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for deployment across multiple H200/H100 nodes.

**Node 0 Launch Command:**

.. code-block:: bash

    # H200/H100 Multi-node DeepSeek-R1 TP Mode Node 0
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

    # H200/H100 Multi-node DeepSeek-R1 TP Mode Node 1
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

2.2 Multi-node EP Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Suitable for deploying MoE models across multiple nodes.

**Node 0 Launch Command:**

.. code-block:: bash

    # H200 Multi-node DeepSeek-R1 EP Mode Node 0
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

    # H200 Multi-node DeepSeek-R1 EP Mode Node 1
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

3. PD disaggregation Deployment Solutions
------------------------------------------

PD (Prefill-Decode) disaggregation mode separates prefill and decode stages for deployment, which can better utilize hardware resources.

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
    # nvidia-cuda-mps-control -d, run MPS (optional, performance will be much better with mps support, but some GPUs may encounter errors when enabling mps, it's recommended to upgrade to a higher driver version, especially for H-series cards)

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

**Step 3: Launch Decode Service**

.. code-block:: bash

    # PD decode mode for DeepSeek-R1 (DP+EP) on H200
    # Usage: sh pd_decode.sh <host> <pd_master_ip>
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
    # if you want to enable microbatch overlap, you can uncomment the following lines
    #--enable_decode_microbatch_overlap

3.2 Multi PD Master Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Supports multiple PD Master nodes, providing better load balancing and high availability.

**Step 1: Launch Config Server**

.. code-block:: bash

    # Config Server
    # Usage: sh config_server.sh <config_server_host>
    export config_server_host=$1
    python -m lightllm.server.api_server \
    --run_mode "config_server" \
    --config_server_host $config_server_host \
    --config_server_port 60088

**Step 2: Launch Multiple PD Masters**

.. code-block:: bash

    # PD Master 1
    # Usage: sh pd_master_1.sh <host> <config_server_host>
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
    # Usage: sh pd_master_2.sh <host> <config_server_host>
    export host=$1
    export config_server_host=$2
    python -m lightllm.server.api_server \
    --model_dir /path/DeepSeek-R1 \
    --run_mode "pd_master" \
    --host $host \
    --port 60012 \
    --config_server_host $config_server_host \
    --config_server_port 60088

**Step 3: Launch Prefill and Decode Services**

.. code-block:: bash

    # Prefill Service
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
    # if you want to enable microbatch overlap, you can uncomment the following lines
    #--enable_prefill_microbatch_overlap

    # Decode Service
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
    # if you want to enable microbatch overlap, you can uncomment the following lines
    #--enable_decode_microbatch_overlap

4. Testing and Validation
-------------------------

4.1 Basic Functionality Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

4.2 Performance Benchmark Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # DeepSeek-R1 Performance Testing
    cd test
    python benchmark_client.py \
    --num_clients 100 \
    --input_num 2000 \
    --tokenizer_path /path/DeepSeek-R1/ \
    --url http://127.0.0.1:8088/generate_stream

All the above scripts can be referenced from the scripts in the `test/start_scripts/multi_pd_master/` directory. 