APIServer Parameter Details
==========================

This document provides detailed information about all startup parameters and their usage for LightLLM APIServer.

Basic Configuration Parameters
-----------------------------

.. option:: --run_mode

    Set the running mode, optional values:
    
    * ``normal``: Single server mode (default)
    * ``prefill``: Prefill mode (for pd separation running mode)
    * ``decode``: Decode mode (for pd separation running mode)
    * ``pd_master``: pd master node mode (for pd separation running mode)
    * ``config_server``: Configuration server mode (for pd separation mode, used to register pd_master nodes and get pd_master node list), specifically designed for large-scale, high-concurrency scenarios, used when `pd_master` encounters significant CPU bottlenecks.

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

PD Separation Mode Parameters
----------------------------

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
-----------------------------

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
------------------------------------

.. option:: --max_total_token_num

    Total token count supported by GPU and model, equals max_batch * (input_len + output_len)
    
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
----------------------------------------

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
---------------------------------

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