APIServer 参数详解
=============================


使用方法
++++++++++++

.. code-block:: console

    python -m lightllm.server.api_server [-h] [--host HOST] [--port PORT] [--model_dir MODEL_DIR]
                                                [--tokenizer_mode TOKENIZER_MODE] [--load_way LOAD_WAY]
                                                [--max_total_token_num MAX_TOTAL_TOKEN_NUM]
                                                [--batch_max_tokens BATCH_MAX_TOKENS] [--eos_id EOS_ID [EOS_ID ...]]
                                                [--running_max_req_size RUNNING_MAX_REQ_SIZE] [--tp TP]
                                                [--max_req_input_len MAX_REQ_INPUT_LEN]
                                                [--max_req_total_len MAX_REQ_TOTAL_LEN] [--nccl_port NCCL_PORT]
                                                [--mode MODE [MODE ...]] [--trust_remote_code] [--disable_log_stats]
                                                [--log_stats_interval LOG_STATS_INTERVAL]
                                                [--router_token_ratio ROUTER_TOKEN_RATIO]
                                                [--router_max_new_token_len ROUTER_MAX_NEW_TOKEN_LEN]
                                                [--router_max_wait_tokens ROUTER_MAX_WAIT_TOKENS]
                                                [--use_dynamic_prompt_cache]
                                                [--splitfuse_block_size SPLITFUSE_BLOCK_SIZE] [--splitfuse_mode]
                                                [--beam_mode] [--diverse_mode] [--token_healing_mode]
                                                [--enable_multimodal] [--cache_capacity CACHE_CAPACITY]
                                                [--cache_reserved_ratio CACHE_RESERVED_RATIO]
                                                [--data_type {fp16,float16,bf16,bfloat16,fp32,float32}]
                                                [--return_all_prompt_logprobs] [--use_reward_model]
                                                [--long_truncation_mode {None,head,center}] [--use_tgi_api]
                                                [--health_monitor] [--metric_gateway METRIC_GATEWAY]
                                                [--job_name JOB_NAME] [--grouping_key GROUPING_KEY]
                                                [--push_interval PUSH_INTERVAL] [--enable_monitor_auth]

参数说明
++++++++

:code:`--host` 
    服务IP地址

    默认值：127.0.0.1

:code:`--port`
    服务端口

    默认值：8000

:code:`--model_dir`
    模型权重目录路径，将从该目录加载配置、权重和分词器

:code:`--tokenizer_mode`
    tokenizer加载模式，可以是 ``slow`` 、 ``fast`` 或 ``auto`` ，慢速模式加载快但运行慢，慢速模式有利于调试和测试，快速模式可以获得最佳性能，自动模式将尝试使用快速模式，如果失败将使用慢速模式
    
    默认值：slow

:code:`--load_way`
    加载权重的方式，默认为 ``HF`` （huggingface 格式），llama模型也支持 ``DS`` （Deepspeed）
    
    默认值：HF

:code:`--max_total_token_num`
    GPU能支持的总token数，等于 max_batch * (input_len + output_len)

    默认值：6000

:code:`--eos_id`
    模型终止输出的 token id

    默认值：[2]

:code:`--running_max_req_size`
    同一时间内进行推理的最大请求数

    默认值：1000

:code:`--tp`
    模型进行张量并行的尺寸

    默认值：1

:code:`--max_req_input_len`
    单个请求最大的输入token量

    默认值：2048

:code:`--max_req_total_len`
    单个请求最大的输入token量+输出token量

    默认值：3072

:code:`--nccl_port`
    创建pytorch分布式环境使用的nccl端口

    默认值：28765

:code:`--mode`
    一个列表，用来对某些适配的模型开启某些算子从而进行加速，可选的方案包括：

    :code:`[triton_int8kv | ppl_int8kv | ppl_fp16 | triton_flashdecoding | triton_gqa_attention | triton_gqa_flashdecoding | triton_w4a16 | triton_w8a16 | triton_w8a8 | lmdeploy_w4a16 | ppl_w4a16 | ppl_w8a8 | ppl_w8a8_mixdown]`
    
    其中，

    * triton_flashdecoding ： 适用于长文本，当前支持的模型包括 llama/llama2/qwen
    * triton_gqa_attention 和 triton_gqa_flashdecoding ：使用于使用GQA的模型
    * triton_int8kv ：使用int8来存储 kv cache，可以提升可容纳的token总数
    * ppl_int8kv ：使用int8来存储 kv cache，并且使用 ppl 核函数进行加速
    * ppl_fp16 ：使用 ppl 的 fp16精度的 decode attention 核函数
    * triton_int8weight：使用int8来存储参数
    * triton_int4weight 和 lmdeploy_int4weight 和 ppl_int4weight：使用int4来存储参数

    .. tip::
        在你使用某种模式之前，你需要先阅读源码确保该模式支持你想要使用的模型。

    默认值：[]

:code:`--trust_remote_code`
    是否允许使用Hub仓库中上传者在上传文件中自定义的模型

    默认值：False

:code:`--disable_log_stats`
    禁用日志系统记录吞吐量统计信息。

    默认值：False

:code:`--log_stats_interval`
    以秒为单位的日志统计间隔。

    默认值：0.0

:code:`--router_token_ratio`
    控制router调度的token的比例

    默认值：0.0

:code:`--router_max_new_token_len`
    对于Router的请求最大的新Token量

    默认值：1024


:code:`--router_max_wait_tokens`
    每次在进行或者等待 router_max_wait_tokens 轮次以后，router 就会调度新请求 

    默认值：10

:code:`--use_dynamic_prompt_cache`
    是否使用 ``use_dynamic_prompt_cache``

    默认值：False

:code:`--splitfuse_block_size`
    splitfuse 块大小

    默认值：256

:code:`--splitfuse_mode`
    是否使用 ``splitfuse`` 模式

    默认值：False

:code:`--beam_mode`
    是否使用 ``beamsearch`` 模式

    默认值：False

:code:`--diverse_mode`
    是否使用 ``diversity generation`` 模式

    默认值：False

:code:`--token_healing_mode`
    是否使用 ``code model infer`` 模式

    默认值：False

:code:`--enable_multimodal`
    是否使用多模态模型

    默认值：False

:code:`--cache_capacity`
    多模态资源缓存服务器的最大缓存量

    默认值：200

:code:`--cache_reserved_ratio`
    清除后资源后，缓存服务器预留容量的比例

    默认值：0.5

:code:`--data_type`
    模型权重的数据格式，可能的选择：fp16, float16, bf16, bfloat16, fp32, float32

    默认值：“float16”

:code:`--return_all_prompt_logprobs`
    是否返回每个提示中所有标记的对数概率

    默认值：False

:code:`--use_reward_model`
    是否使用 reward 类模型

    默认值：False 

:code:`--long_truncation_mode`
    用于选择对于过长的输入的处理方式，有如下的选择;

    * None : 返回异常
    * head ：移除起始的一些token
    * center：移除中间的某些token

    默认值：None

:code:`--use_tgi_api`
    使用 tgi 的输入和输出格式

    默认值：False

:code:`--health_monitor`
    是否开启健康检查，健康检查会不断检查服务器的健康状况，并在出错时进行重启

    默认值：False

:code:`--metric_gateway`
    对指标进行监控的IP地址


:code:`--job_name`
    监视器的作业名称

    默认值：“lightllm”

:code:`--grouping_key`
    监视器的 grouping_key，格式为 key=value

    默认值：[]

:code:`--push_interval`
    以秒为单位的推送监控指标的时间间隔

    默认值：10

:code:`--enable_monitor_auth`
    是否开启push_gateway的认证

    默认值：False