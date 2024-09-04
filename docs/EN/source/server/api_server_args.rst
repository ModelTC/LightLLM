APIServer Args
=============================


Usage
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

Arguments
++++++++++++++

:code:`--host` 
    Service IP address.

    Default : 127.0.0.1

:code:`--port`
    Service port.

    Default : 8000

:code:`--model_dir`
    The model weight dir path, the app will load config, weights and tokenizer from this dir.

:code:`--tokenizer_mode`
    Tokenizer load mode, can be slow, fast or auto, slow mode load fast but run slow, slow mode is good for debug and test, fast mode get best performance, auto mode will try to use fast mode, if failed will use slow mode. 
    
    Default : slow

:code:`--load_way`
    the way of loading model weights, the default is HF(Huggingface format), llama also supports DS(Deepspeed)    
    
    Default : HF

:code:`--max_total_token_num`
    the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)
    
    Default : 6000

:code:`--batch_max_tokens`
    max tokens num for new cat batch, it control prefill batch size to Preventing OOM

    Default : None

:code:`--eos_id`
    eos stop token id

    Default : [2]

:code:`--running_max_req_size`
    the max size for forward requests in the same time

    Default : 1000

:code:`--tp`
    model tp parral size, the default is 1

    Default : 1

:code:`--max_req_input_len`
    the max value for req input tokens num

    Default : 2048

:code:`--max_req_total_len`
    the max value for req_input_len + req_output_len

    Default : 2048 + 1024

:code:`--nccl_port`
    the nccl_port to build a distributed environment for PyTorch

    Default : 28765

:code:`--mode`
    Model mode: [triton_int8kv | ppl_int8kv | ppl_fp16 | triton_flashdecoding| triton_gqa_attention | triton_gqa_flashdecoding| triton_w4a16 | triton_w8a16 | triton_w8a8 | lmdeploy_w4a16| ppl_w4a16 | ppl_w8a8 | ppl_w8a8_mixdown],
    
    * triton_flashdecoding mode is for long context, current support llama llama2 qwen;
    * triton_gqa_attention and triton_gqa_flashdecoding is fast kernel for model which use GQA;
    * triton_int8kv mode use int8 to store kv cache, can increase token capacity, use triton kernel;
    * ppl_int8kv mode use int8 to store kv cache, and use ppl fast kernel;
    * ppl_fp16 mode use ppl fast fp16 decode attention kernel;
    * triton_int8weight and triton_int4weight and lmdeploy_int4weight or ppl_int4weight mode
    * use int8 and int4 to store weights;
    * you need to read source code to make sure the supported detail mode for all models

    Default : []

:code:`--trust_remote_code`
    Whether or not to allow for custom models defined on the Hub in their own modeling files.

    Default : False

:code:`--disable_log_stats`
    disable logging throughput stats.

    Default : False

:code:`--log_stats_interval`
    log stats interval in second.

    Default : 10

:code:`--router_token_ratio`
    token ratio to control router dispatch

    Default : 0.0

:code:`--router_max_new_token_len`
    the request max new token len for router

    Default : 1024

:code:`--router_max_wait_tokens`
    schedule new requests after every router_max_wait_tokens decode steps.
    
    Default : 10

:code:`--use_dynamic_prompt_cache`
    use_dynamic_prompt_cache test

    Default : False

:code:`--splitfuse_block_size`
    splitfuse block size

    Default : 256

:code:`--splitfuse_mode`
    use ``splitfuse`` mode

    Default : False

:code:`--beam_mode`
    use ``beamsearch`` mode

    Default : False

:code:`--diverse_mode`
    use ``diversity generation`` mode

    Default : False

:code:`--token_healing_mode`
    use ``code model infer`` mode

    Default : False

:code:`--enable_multimodal`
    Whether or not to allow to load additional multimodal models.

    Default : False

:code:`--cache_capacity`
    cache server capacity for multimodal resources

    Default : 200

:code:`--cache_reserved_ratio`
    cache server reserved capacity ratio after clear

    Default : 0.5

:code:`--data_type`
    the data type of the model weight, choices : fp16, float16, bf16, bfloat16, fp32, float32

    Default : “float16”

:code:`--return_all_prompt_logprobs`
    return_all_prompt_logprobs

    Default : False

:code:`--use_reward_model`
    use reward model.

    Default : False 

:code:`--long_truncation_mode`
    use to select the handle way when input token len > max_req_input_len.

    * None : raise Exception
    * head : remove some head tokens to make input token len <= max_req_input_len
    * center : remove some tokens in center loc to make input token len <= max_req_input_len

    Default : None

:code:`--use_tgi_api`
    use tgi input and ouput format

    Default : False

:code:`--health_monitor`
    check the health of service and restart when error

    Default : False

:code:`--metric_gateway`
    address for collecting monitoring metrics


:code:`--job_name`
    job name for monitor

    Default : “lightllm”

:code:`--grouping_key`
    grouping_key for the monitor in the form key=value

    Default : []

:code:`--push_interval`
    interval of pushing monitoring metrics

    Default : 10

:code:`--enable_monitor_auth`
    Whether to open authentication for push_gateway

    Default : False