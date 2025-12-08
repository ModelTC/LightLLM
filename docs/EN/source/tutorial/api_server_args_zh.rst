.. _tutorial/api_server_args_zh:

APIServer Parameter Details
===========================

This document provides detailed information about all startup parameters and their usage for LightLLM APIServer.

Basic Configuration Parameters
------------------------------

.. option:: --run_mode

    Set the running mode, optional values:
    
    * ``normal``: Single server mode (default)
    * ``prefill``: Prefill mode (for pd disaggregation running mode)
    * ``decode``: Decode mode (for pd disaggregation running mode)
    * ``pd_master``: pd master node mode (for pd disaggregation running mode)
    * ``config_server``: Configuration server mode (for pd disaggregation mode, used to register pd_master nodes and get pd_master node list), specifically designed for large-scale, high-concurrency scenarios, used when `pd_master` encounters significant CPU bottlenecks.

.. option:: --host

    Server listening address, default is ``127.0.0.1``

.. option:: --port

    Server listening port, default is ``8000``

.. option:: --httpserver_workers

    HTTP server worker process count, default is ``1``

.. option:: --zmq_mode

    ZMQ communication mode, optional values:
    
    * ``tcp://``: TCP mode
    * ``ipc:///tmp/``: IPC mode (default)
    
    Can only choose from ``['tcp://', 'ipc:///tmp/']``

PD disaggregation Mode Parameters
----------------------------------

.. option:: --pd_master_ip

    PD master node IP address, default is ``0.0.0.0``
    
    This parameter needs to be set when run_mode is set to prefill or decode

.. option:: --pd_master_port

    PD master node port, default is ``1212``
    
    This parameter needs to be set when run_mode is set to prefill or decode

.. option:: --pd_decode_rpyc_port

    Port used by decode nodes for kv move manager rpyc server in PD mode, default is ``42000``

.. option:: --config_server_host

    Host address in configuration server mode

.. option:: --config_server_port

    Port number in configuration server mode

Model Configuration Parameters
------------------------------

.. option:: --model_name

    Model name, used to distinguish internal model names, default is ``default_model_name``
    
    Can be obtained via ``host:port/get_model_name``

.. option:: --model_dir

    Model weight directory path, the application will load configuration, weights, and tokenizer from this directory

.. option:: --tokenizer_mode

    Tokenizer loading mode, optional values:
    
    * ``slow``: Slow mode, loads fast but runs slow, suitable for debugging and testing
    * ``fast``: Fast mode (default), achieves best performance
    * ``auto``: Auto mode, tries to use fast mode, falls back to slow mode if it fails

.. option:: --load_way

    Model weight loading method, default is ``HF`` (Huggingface format)
    
    Llama models also support ``DS`` (Deepspeed) format

.. option:: --trust_remote_code

    Whether to allow using custom model definition files on Hub

Memory and Batch Processing Parameters
---------------------------------------

.. option:: --max_total_token_num

    Total token number of kv cache. 
    
    If not specified, will be automatically calculated based on mem_fraction

.. option:: --mem_fraction

    Memory usage ratio, default is ``0.9``
    
    If OOM occurs during runtime, you can specify a smaller value

.. option:: --batch_max_tokens

    Maximum token count for new batches, controls prefill batch size to prevent OOM

.. option:: --running_max_req_size

    Maximum number of requests for simultaneous forward inference, default is ``1000``

.. option:: --max_req_total_len

    Maximum value of request input length + request output length, default is ``16384``

.. option:: --eos_id

    End stop token ID, can specify multiple values. If None, will be loaded from config.json

.. option:: --tool_call_parser

    OpenAI interface tool call parser type, optional values:
    
    * ``qwen25``
    * ``llama3``
    * ``mistral``

Different Parallel Mode Setting Parameters
-------------------------------------------

.. option:: --nnodes

    Number of nodes, default is ``1``

.. option:: --node_rank

    Current node rank, default is ``0``

.. option:: --multinode_httpmanager_port

    Multi-node HTTP manager port, default is ``12345``

.. option:: --multinode_router_gloo_port

    Multi-node router gloo port, default is ``20001``

.. option:: --tp

    Model tensor parallelism size, default is ``1``

.. option:: --dp

    Data parallelism size, default is ``1``
    
    This is a useful parameter for deepseekv2. When using deepseekv2 model, set dp equal to the tp parameter.
    In other cases, please do not set it, keep the default value of 1.

.. option:: --nccl_host

    nccl_host used to build PyTorch distributed environment, default is ``127.0.0.1``
    
    For multi-node deployment, should be set to the master node's IP

.. option:: --nccl_port

    nccl_port used to build PyTorch distributed environment, default is ``28765``

.. option:: --use_config_server_to_init_nccl

    Use tcp store server started by config_server to initialize nccl, default is False
    
    When set to True, --nccl_host must equal config_server_host, --nccl_port must be unique for config_server,
    do not use the same nccl_port for different inference nodes, this will be a serious error

Attention Type Selection Parameters
------------------------------------

.. option:: --mode

    Model inference mode, can specify multiple values:
    
    * ``triton_int8kv``: Use int8 to store kv cache, can increase token capacity, uses triton kernel
    * ``ppl_int8kv``: Use int8 to store kv cache, uses ppl fast kernel
    * ``ppl_fp16``: Use ppl fast fp16 decode attention kernel
    * ``triton_flashdecoding``: Flashdecoding mode for long context, currently supports llama llama2 qwen
    * ``triton_gqa_attention``: Fast kernel for models using GQA
    * ``triton_gqa_flashdecoding``: Fast flashdecoding kernel for models using GQA
    * ``triton_fp8kv``: Use float8 to store kv cache, currently only used for deepseek2
    
    Need to read source code to confirm specific modes supported by all models 

Scheduling Parameters
---------------------

.. option:: --router_token_ratio

    Threshold for determining if the service is busy, default is ``0.0``. Once the kv cache usage exceeds this value, it will directly switch to conservative scheduling.

.. option:: --router_max_new_token_len

    The request output length used by the scheduler when evaluating request kv usage, default is ``1024``, generally lower than the max_new_tokens set by the user. This parameter only takes effect when --router_token_ratio is greater than 0.
    Setting this parameter will make request scheduling more aggressive, allowing the system to process more requests simultaneously, but will inevitably cause request pause and recalculation.

.. option:: --router_max_wait_tokens

    Trigger scheduling of new requests every router_max_wait_tokens decoding steps, default is ``6``

.. option:: --disable_aggressive_schedule

    Disable aggressive scheduling
    
    Aggressive scheduling may cause frequent prefill interruptions during decoding. Disabling it can make the router_max_wait_tokens parameter work more effectively.

.. option:: --disable_dynamic_prompt_cache

    Disable kv cache caching

.. option:: --chunked_prefill_size

    Chunked prefill size, default is ``4096``

.. option:: --disable_chunked_prefill

    Whether to disable chunked prefill

.. option:: --diverse_mode

    Multi-result output mode

.. option:: --schedule_time_interval

    Schedule time interval, default is ``0.03``, unit is seconds

Output Constraint Parameters
----------------------------

.. option:: --token_healing_mode

.. option:: --output_constraint_mode

    Set the output constraint backend, optional values:
    
    * ``outlines``: Use outlines backend
    * ``xgrammar``: Use xgrammar backend
    * ``none``: No output constraint (default)

.. option:: --first_token_constraint_mode

    Constrain the allowed range of the first token
    Use environment variable FIRST_ALLOWED_TOKENS to set the range, e.g., FIRST_ALLOWED_TOKENS=1,2

Multimodal Parameters
---------------------

.. option:: --enable_multimodal

    Whether to allow loading additional visual models

.. option:: --enable_multimodal_audio

    Whether to allow loading additional audio models (requires --enable_multimodal)

.. option:: --enable_mps

    Whether to enable nvidia mps for multimodal services

.. option:: --cache_capacity

    Cache server capacity for multimodal resources, default is ``200``

.. option:: --visual_infer_batch_size

    Number of images processed in each inference batch, default is ``1``

.. option:: --visual_gpu_ids

    List of GPU IDs to use, e.g., 0 1 2

.. option:: --visual_tp

    Number of tensor parallel instances for ViT, default is ``1``

.. option:: --visual_dp

    Number of data parallel instances for ViT, default is ``1``

.. option:: --visual_nccl_ports

    List of NCCL ports for ViT, e.g., 29500 29501 29502, default is [29500]

Performance Optimization Parameters
-----------------------------------

.. option:: --disable_custom_allreduce

    Whether to disable custom allreduce

.. option:: --enable_custom_allgather

    Whether to enable custom allgather

.. option:: --enable_tpsp_mix_mode

    The inference backend will use TP SP mixed running mode
    
    Currently only supports llama and deepseek series models

.. option:: --enable_prefill_microbatch_overlap

    The inference backend will use microbatch overlap mode for prefill
    
    Currently only supports deepseek series models

.. option:: --enable_decode_microbatch_overlap

    The inference backend will use microbatch overlap mode for decoding
    
.. option:: --enable_flashinfer_prefill

    The inference backend will use flashinfer's attention kernel for prefill
    
.. option:: --enable_flashinfer_decode

    The inference backend will use flashinfer's attention kernel for decoding
    
.. option:: --enable_fa3

    The inference backend will use fa3 attention kernel for prefill and decoding

.. option:: --disable_cudagraph

    Disable cudagraph in the decoding phase

.. option:: --graph_max_batch_size

    Maximum batch size that can be captured by cuda graph in the decoding phase, default is ``256``

.. option:: --graph_split_batch_size

    Controls the interval for generating CUDA graphs during decoding, default is ``32``
    
    For values from 1 to the specified graph_split_batch_size, CUDA graphs will be generated continuously.
    For values from graph_split_batch_size to graph_max_batch_size,
    a new CUDA graph will be generated for every increase of graph_grow_step_size.
    Properly configuring this parameter can help optimize the performance of CUDA graph execution.

.. option:: --graph_grow_step_size

    For batch_size values from graph_split_batch_size to graph_max_batch_size,
    a new CUDA graph will be generated for every increase of graph_grow_step_size, default is ``16``

.. option:: --graph_max_len_in_batch

    Maximum sequence length that can be captured by cuda graph in the decoding phase, default is ``max_req_total_len``

Quantization Parameters
-----------------------

.. option:: --quant_type

    Quantization method, optional values:
    
    * ``ppl-w4a16-128``
    * ``flashllm-w6a16``
    * ``ao-int4wo-[32,64,128,256]``
    * ``ao-int8wo``
    * ``ao-fp8w8a16``
    * ``ao-fp6w6a16``
    * ``vllm-w8a8``
    * ``vllm-fp8w8a8``
    * ``vllm-fp8w8a8-b128``
    * ``triton-fp8w8a8-block128``
    * ``none`` (default)

.. option:: --quant_cfg

    Path to quantization configuration file. Can be used for mixed quantization.
    
    Examples can be found in test/advanced_config/mixed_quantization/llamacls-mix-down.yaml.

.. option:: --vit_quant_type

    ViT quantization method, optional values:
    
    * ``ppl-w4a16-128``
    * ``flashllm-w6a16``
    * ``ao-int4wo-[32,64,128,256]``
    * ``ao-int8wo``
    * ``ao-fp8w8a16``
    * ``ao-fp6w6a16``
    * ``vllm-w8a8``
    * ``vllm-fp8w8a8``
    * ``none`` (default)

.. option:: --vit_quant_cfg

    Path to ViT quantization configuration file. Can be used for mixed quantization.
    
    Examples can be found in lightllm/common/quantization/configs.

Sampling and Generation Parameters
----------------------------------

.. option:: --sampling_backend

    Implementation used for sampling, optional values:
    
    * ``triton``: Use torch and triton kernel (default)
    * ``sglang_kernel``: Use sglang_kernel implementation

.. option:: --return_all_prompt_logprobs

    Return logprobs for all prompt tokens

.. option:: --use_reward_model

    Use reward model

.. option:: --long_truncation_mode

    How to handle when input_token_len + max_new_tokens > max_req_total_len, optional values:
    
    * ``None``: Throw exception (default)
    * ``head``: Remove some head tokens to make input_token_len + max_new_tokens <= max_req_total_len
    * ``center``: Remove some tokens at the center position to make input_token_len + max_new_tokens <= max_req_total_len

.. option:: --use_tgi_api

    Use tgi input and output format

MTP Multi-Prediction Parameters
-------------------------------

.. option:: --mtp_mode

    Supported mtp modes, it is recommended to use deepseekv3_eagle for better performance, optional values:
    
    * ``deepseekv3_vanilla``
    * ``deepseekv3_eagle``
    * ``None``: Do not enable mtp (default)

.. option:: --mtp_draft_model_dir

    Path to the draft model for MTP multi-prediction functionality
    
    Used to load the MTP multi-output token model.

.. option:: --mtp_step

    Specify the number of additional tokens predicted by the draft model, default is ``0``
    
    Currently this feature only supports DeepSeekV3/R1 models.
    Increasing this value allows more predictions, but ensure the model is compatible with the specified number of steps.
    Currently deepseekv3/r1 models only support 1 step

DeepSeek Redundant Expert Parameters
------------------------------------

.. option:: --ep_redundancy_expert_config_path

    Path to redundant expert configuration. Can be used for deepseekv3 models.

.. option:: --auto_update_redundancy_expert

    Whether to update redundant experts for deepseekv3 models through online expert usage counters.

Monitoring and Logging Parameters
---------------------------------

.. option:: --disable_log_stats

    Disable throughput statistics logging

.. option:: --log_stats_interval

    Interval for recording statistics (seconds), default is ``10``

.. option:: --health_monitor

    Check service health status and restart on error

.. option:: --metric_gateway

    Address for collecting monitoring metrics

.. option:: --job_name

    Job name for monitoring, default is ``lightllm``

.. option:: --grouping_key

    Grouping key for monitoring, format is key=value, can specify multiple

.. option:: --push_interval

    Interval for pushing monitoring metrics (seconds), default is ``10``

.. option:: --enable_monitor_auth

    Whether to enable authentication for push_gateway 