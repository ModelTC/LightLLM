.. _tutorial/api_server_args_zh:

APIServer 参数详解
==================

本文档详细介绍了 LightLLM APIServer 的所有启动参数及其用法。

基础配置参数
------------

.. option:: --run_mode

    设置运行模式，可选值：
    
    * ``normal``: 单服务器模式（默认）
    * ``prefill``: 预填充模式（用于 pd 分离运行模式）
    * ``decode``: 解码模式（用于 pd 分离运行模式）
    * ``pd_master``: pd 主节点模式（用于 pd 分离运行模式）
    * ``config_server``: 配置服务器模式（用于 pd 分离模式，用于注册 pd_master 节点并获取 pd_master 节点列表）,专门为大规模、高并发场景设计，当 `pd_master` 遇到显著的 CPU 瓶颈时使用。

.. option:: --host

    服务器监听地址，默认为 ``127.0.0.1``

.. option:: --port

    服务器监听端口，默认为 ``8000``

.. option:: --httpserver_workers

    HTTP 服务器工作进程数，默认为 ``1``

.. option:: --zmq_mode

    ZMQ 通信模式，可选值：
    
    * ``tcp://``: TCP 模式
    * ``ipc:///tmp/``: IPC 模式（默认）
    
    只能在 ``['tcp://', 'ipc:///tmp/']`` 中选择

PD 分离模式参数
---------------

.. option:: --pd_master_ip

    PD 主节点 IP 地址，默认为 ``0.0.0.0``
    
    当 run_mode 设置为 prefill 或 decode 时需要设置此参数

.. option:: --pd_master_port

    PD 主节点端口，默认为 ``1212``
    
    当 run_mode 设置为 prefill 或 decode 时需要设置此参数

.. option:: --pd_decode_rpyc_port

    PD 模式下解码节点用于 kv move manager rpyc 服务器的端口，默认为 ``42000``

.. option:: --config_server_host

    配置服务器模式下的主机地址

.. option:: --config_server_port

    配置服务器模式下的端口号

模型配置参数
------------

.. option:: --model_name

    模型名称，用于区分内部模型名称，默认为 ``default_model_name``
    
    可通过 ``host:port/get_model_name`` 获取

.. option:: --model_dir

    模型权重目录路径，应用将从该目录加载配置、权重和分词器

.. option:: --tokenizer_mode

    分词器加载模式，可选值：
    
    * ``slow``: 慢速模式，加载快但运行慢，适合调试和测试
    * ``fast``: 快速模式（默认），获得最佳性能
    * ``auto``: 自动模式，尝试使用快速模式，失败则使用慢速模式

.. option:: --load_way

    模型权重加载方式，默认为 ``HF`` (Huggingface 格式)
    
    llama 模型还支持 ``DS`` (Deepspeed) 格式

.. option:: --trust_remote_code

    是否允许在 Hub 上使用自定义模型定义的文件

内存和批处理参数
----------------

.. option:: --max_total_token_num

    GPU 和模型支持的总 token 数量，等于 max_batch * (input_len + output_len)
    
    如果不指定，将根据 mem_fraction 自动计算

.. option:: --mem_fraction

    内存使用比例，默认为 ``0.9``
    
    如果运行时出现 OOM，可以指定更小的值

.. option:: --batch_max_tokens

    新批次的最大 token 数量，控制预填充批次大小以防止 OOM

.. option:: --running_max_req_size

    同时进行前向推理的最大请求数量，默认为 ``1000``

.. option:: --max_req_total_len

    请求输入长度 + 请求输出长度的最大值，默认为 ``16384``

.. option:: --eos_id

    结束停止 token ID，可以指定多个值。如果为 None，将从 config.json 加载

.. option:: --tool_call_parser

    openai接口工具调用解析器类型，可选值：
    
    * ``qwen25``
    * ``llama3``
    * ``mistral``

不同并行模式设置参数
--------------------

.. option:: --nnodes

    节点数量，默认为 ``1``

.. option:: --node_rank

    当前节点的排名，默认为 ``0``

.. option:: --multinode_httpmanager_port

    多节点 HTTP 管理器端口，默认为 ``12345``

.. option:: --multinode_router_gloo_port

    多节点路由器 gloo 端口，默认为 ``20001``

.. option:: --tp

    模型张量并行大小，默认为 ``1``

.. option:: --dp

    数据并行大小，默认为 ``1``
    
    这是 deepseekv2 的有用参数。使用 deepseekv2 模型时，将 dp 设置为等于 tp 参数。
    其他情况下请不要设置，保持默认值 1。

.. option:: --nccl_host

    用于构建 PyTorch 分布式环境的 nccl_host，默认为 ``127.0.0.1``
    
    多节点部署时，应设置为主节点的 IP

.. option:: --nccl_port

    用于构建 PyTorch 分布式环境的 nccl_port，默认为 ``28765``

.. option:: --use_config_server_to_init_nccl

    使用由 config_server 启动的 tcp store 服务器初始化 nccl，默认为 False
    
    设置为 True 时，--nccl_host 必须等于 config_server_host，--nccl_port 对于 config_server 必须是唯一的，
    不要为不同的推理节点使用相同的 nccl_port，这将是严重错误

attention类型选择参数
---------------------

.. option:: --mode

    模型推理模式，可以指定多个值：
    
    * ``triton_int8kv``: 使用 int8 存储 kv cache，可增加 token 容量，使用 triton kernel
    * ``ppl_int8kv``: 使用 int8 存储 kv cache，使用 ppl 快速 kernel
    * ``ppl_fp16``: 使用 ppl 快速 fp16 解码注意力 kernel
    * ``triton_flashdecoding``: 用于长上下文的 flashdecoding 模式，当前支持 llama llama2 qwen
    * ``triton_gqa_attention``: 使用 GQA 的模型的快速 kernel
    * ``triton_gqa_flashdecoding``: 使用 GQA 的模型的快速 flashdecoding kernel
    * ``triton_fp8kv``: 使用 float8 存储 kv cache，目前仅用于 deepseek2
    
    需要阅读源代码以确认所有模型支持的具体模式

调度参数
--------

.. option:: --router_token_ratio

    判断服务是否繁忙的阈值，默认为 ``0.0``，一旦kv cache 使用率超过此值，则会直接变为保守调度。

.. option:: --router_max_new_token_len

    调度器评估请求kv占用时，使用的请求输出长度，默认为 ``1024``，一般低于用户设置的max_new_tokens。该参数只在 --router_token_ratio 大于0时生效。
    设置改参数，会使请求调度更为激进，系统同时处理的请求数会更多，同时也会不可避免的造成请求的暂停重计算。

.. option:: --router_max_wait_tokens

    每 router_max_wait_tokens 解码步骤后触发一次调度新请求，默认为 ``6``

.. option:: --disable_aggressive_schedule

    禁用激进调度
    
    激进调度可能导致解码期间频繁的预填充中断。禁用它可以让 router_max_wait_tokens 参数更有效地工作。

.. option:: --disable_dynamic_prompt_cache

    禁用kv cache 缓存

.. option:: --chunked_prefill_size

    分块预填充大小，默认为 ``4096``

.. option:: --disable_chunked_prefill

    是否禁用分块预填充

.. option:: --diverse_mode

    多结果输出模式

.. option:: --schedule_time_interval

    调度时间间隔，默认为 ``0.03``，单位为秒


输出约束参数
------------

.. option:: --token_healing_mode

.. option:: --output_constraint_mode

    设置输出约束后端，可选值：
    
    * ``outlines``: 使用 outlines 后端
    * ``xgrammar``: 使用 xgrammar 后端
    * ``none``: 无输出约束（默认）

.. option:: --first_token_constraint_mode

    约束第一个 token 的允许范围
    使用环境变量 FIRST_ALLOWED_TOKENS 设置范围，例如 FIRST_ALLOWED_TOKENS=1,2

多模态参数
----------

.. option:: --enable_multimodal

    是否允许加载额外的视觉模型

.. option:: --enable_multimodal_audio

    是否允许加载额外的音频模型（需要 --enable_multimodal）

.. option:: --enable_mps

    是否为多模态服务启用 nvidia mps

.. option:: --cache_capacity

    多模态资源的缓存服务器容量，默认为 ``200``

.. option:: --visual_infer_batch_size

    每次推理批次中处理的图像数量，默认为 ``1``

.. option:: --visual_gpu_ids

    要使用的 GPU ID 列表，例如 0 1 2

.. option:: --visual_tp

    ViT 的张量并行实例数量，默认为 ``1``

.. option:: --visual_dp

    ViT 的数据并行实例数量，默认为 ``1``

.. option:: --visual_nccl_ports

    为 ViT 构建分布式环境的 NCCL 端口列表，例如 29500 29501 29502，默认为 [29500]

性能优化参数
------------

.. option:: --disable_custom_allreduce

    是否禁用自定义 allreduce

.. option:: --enable_custom_allgather

    是否启用自定义 allgather

.. option:: --enable_tpsp_mix_mode

    推理后端将使用 TP SP 混合运行模式
    
    目前仅支持 llama 和 deepseek系列 模型

.. option:: --enable_prefill_microbatch_overlap

    推理后端将为预填充使用微批次重叠模式
    
    目前仅支持 deepseek系列 模型

.. option:: --enable_decode_microbatch_overlap

    推理后端将为解码使用微批次重叠模式
    
.. option:: --enable_flashinfer_prefill

    推理后端将为预填充使用 flashinfer 的注意力 kernel
    
.. option:: --enable_flashinfer_decode

    推理后端将为解码使用 flashinfer 的注意力 kernel
    
.. option:: --enable_fa3

    推理后端将为预填充和解码使用 fa3 注意力 kernel

.. option:: --disable_cudagraph

    禁用解码阶段的 cudagraph

.. option:: --graph_max_batch_size

    解码阶段可以被 cuda graph 捕获的最大批次大小，默认为 ``256``

.. option:: --graph_split_batch_size

    控制解码期间生成 CUDA graph 的间隔，默认为 ``32``
    
    对于从 1 到指定 graph_split_batch_size 的值，将连续生成 CUDA graph。
    对于从 graph_split_batch_size 到 graph_max_batch_size 的值，
    每增加 graph_grow_step_size 就会生成一个新的 CUDA graph。
    正确配置此参数可以帮助优化 CUDA graph 执行的性能。

.. option:: --graph_grow_step_size

    对于从 graph_split_batch_size 到 graph_max_batch_size 的 batch_size 值，
    每增加 graph_grow_step_size 就会生成一个新的 CUDA graph，默认为 ``16``

.. option:: --graph_max_len_in_batch

    解码阶段可以被 cuda graph 捕获的最大序列长度，默认为 ``0``
    
    默认值为 8192。如果遇到更大的值，将转为 eager 模式。

量化参数
--------

.. option:: --quant_type

    量化方法，可选值：
    
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
    * ``none`` (默认)

.. option:: --quant_cfg

    量化配置文件路径。可用于混合量化。
    
    示例可以在 test/advanced_config/mixed_quantization/llamacls-mix-down.yaml 中找到。

.. option:: --vit_quant_type

    ViT 量化方法，可选值：
    
    * ``ppl-w4a16-128``
    * ``flashllm-w6a16``
    * ``ao-int4wo-[32,64,128,256]``
    * ``ao-int8wo``
    * ``ao-fp8w8a16``
    * ``ao-fp6w6a16``
    * ``vllm-w8a8``
    * ``vllm-fp8w8a8``
    * ``none`` (默认)

.. option:: --vit_quant_cfg

    ViT 量化配置文件路径。可用于混合量化。
    
    示例可以在 lightllm/common/quantization/configs 中找到。

采样和生成参数
--------------

.. option:: --sampling_backend

    采样使用的实现，可选值：
    
    * ``triton``: 使用 torch 和 triton kernel（默认）
    * ``sglang_kernel``: 使用 sglang_kernel 实现

.. option:: --return_all_prompt_logprobs

    返回所有提示 token 的 logprobs

.. option:: --use_reward_model

    使用奖励模型

.. option:: --long_truncation_mode

    当 input_token_len + max_new_tokens > max_req_total_len 时的处理方式，可选值：
    
    * ``None``: 抛出异常（默认）
    * ``head``: 移除一些头部 token 使 input_token_len + max_new_tokens <= max_req_total_len
    * ``center``: 移除中心位置的一些 token 使 input_token_len + max_new_tokens <= max_req_total_len

.. option:: --use_tgi_api

    使用 tgi 输入和输出格式

MTP 多预测参数
--------------

.. option:: --mtp_mode

    支持的 mtp 模式，建议使用 eagle_with_att获得更好的性能体验，可选值：
    
    * ``vanilla_with_att``
    * ``eagle_with_att``
    * ``vanilla_no_att``
    * ``eagle_no_att``
    * ``None``: 不启用 mtp（默认）

.. option:: --mtp_draft_model_dir

    MTP 多预测功能的草稿模型路径
    
    用于加载 MTP 多输出 token 模型。

.. option:: --mtp_step

    指定使用草稿模型预测的额外 token 数量，默认为 ``0``
    
    目前此功能仅支持 DeepSeekV3/R1 模型。
    增加此值允许更多预测，但确保模型与指定的步数兼容。
    目前 deepseekv3/r1 模型仅支持 1 步

DeepSeek 冗余专家参数
---------------------

.. option:: --ep_redundancy_expert_config_path

    冗余专家配置的路径。可用于 deepseekv3 模型。

.. option:: --auto_update_redundancy_expert

    是否通过在线专家使用计数器为 deepseekv3 模型更新冗余专家。

监控和日志参数
--------------

.. option:: --disable_log_stats

    禁用吞吐量统计日志记录

.. option:: --log_stats_interval

    记录统计信息的间隔（秒），默认为 ``10``

.. option:: --health_monitor

    检查服务健康状态并在出错时重启

.. option:: --metric_gateway

    收集监控指标的地址

.. option:: --job_name

    监控的作业名称，默认为 ``lightllm``

.. option:: --grouping_key

    监控的分组键，格式为 key=value，可以指定多个

.. option:: --push_interval

    推送监控指标的间隔（秒），默认为 ``10``

.. option:: --enable_monitor_auth

    是否为 push_gateway 开启身份验证