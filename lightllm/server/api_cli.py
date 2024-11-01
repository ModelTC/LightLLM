import argparse


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--model_name",
        type=str,
        default="default_model_name",
        help="just help to distinguish internal model name, use 'host:port/get_model_name' to get",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="the model weight dir path, the app will load config, weights and tokenizer from this dir",
    )
    parser.add_argument(
        "--tokenizer_mode",
        type=str,
        default="slow",
        help="""tokenizer load mode, can be slow, fast or auto, slow mode load fast but run slow,
          slow mode is good for debug and test, fast mode get best performance, auto mode will
          try to use fast mode, if failed will use slow mode""",
    )
    parser.add_argument(
        "--load_way",
        type=str,
        default="HF",
        help="""the way of loading model weights, the default is HF(Huggingface format), llama also supports
          DS(Deepspeed)""",
    )
    parser.add_argument(
        "--max_total_token_num",
        type=int,
        default=None,
        help="the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)",
    )
    parser.add_argument(
        "--mem_fraction",
        type=float,
        default=0.9,
        help="""Memory usage ratio, default is 0.9, you can specify a smaller value if OOM occurs at runtime.
        If max_total_token_num is not specified, it will be calculated automatically based on this value.""",
    )
    parser.add_argument(
        "--batch_max_tokens",
        type=int,
        default=None,
        help="max tokens num for new cat batch, it control prefill batch size to Preventing OOM",
    )
    parser.add_argument(
        "--eos_id", nargs="+", type=int, default=None, help="eos stop token id, if None, will load from config.json"
    )
    parser.add_argument(
        "--running_max_req_size", type=int, default=1000, help="the max size for forward requests in the same time"
    )
    parser.add_argument("--tp", type=int, default=1, help="model tp parral size, the default is 1")
    parser.add_argument("--max_req_input_len", type=int, default=2048, help="the max value for req input tokens num")
    parser.add_argument(
        "--max_req_total_len", type=int, default=2048 + 1024, help="the max value for req_input_len + req_output_len"
    )
    parser.add_argument(
        "--nccl_port", type=int, default=28765, help="the nccl_port to build a distributed environment for PyTorch"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=[],
        nargs="+",
        help="""Model mode: [triton_int8kv | ppl_int8kv | ppl_fp16 | triton_flashdecoding
                        | triton_gqa_attention | triton_gqa_flashdecoding
                        | triton_w4a16 | triton_w8a16 | triton_w8a8 | lmdeploy_w4a16
                        | ppl_w4a16 | ppl_w8a8 | ppl_w8a8_mixdown],
                        triton_flashdecoding mode is for long context, current support llama llama2 qwen;
                        triton_gqa_attention and triton_gqa_flashdecoding is fast kernel for model which use GQA;
                        triton_int8kv mode use int8 to store kv cache, can increase token capacity, use triton kernel;
                        ppl_int8kv mode use int8 to store kv cache, and use ppl fast kernel;
                        ppl_fp16 mode use ppl fast fp16 decode attention kernel;
                        triton_int8weight and triton_int4weight and lmdeploy_int4weight or ppl_int4weight mode
                        use int8 and int4 to store weights;
                        you need to read source code to make sure the supported detail mode for all models""",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    )
    parser.add_argument("--disable_log_stats", action="store_true", help="disable logging throughput stats.")
    parser.add_argument("--log_stats_interval", type=int, default=10, help="log stats interval in second.")

    parser.add_argument("--router_token_ratio", type=float, default=0.0, help="token ratio to control router dispatch")
    parser.add_argument(
        "--router_max_new_token_len", type=int, default=1024, help="the request max new token len for router"
    )

    parser.add_argument(
        "--router_max_wait_tokens",
        type=int,
        default=10,
        help="schedule new requests after every router_max_wait_tokens decode steps.",
    )

    parser.add_argument("--use_dynamic_prompt_cache", action="store_true", help="use_dynamic_prompt_cache test")

    parser.add_argument("--splitfuse_block_size", type=int, default=256, help="splitfuse block size")

    parser.add_argument("--splitfuse_mode", action="store_true", help="use splitfuse mode")
    parser.add_argument("--beam_mode", action="store_true", help="use beamsearch mode")
    parser.add_argument("--diverse_mode", action="store_true", help="diversity generation mode")
    parser.add_argument("--token_healing_mode", action="store_true", help="code model infer mode")
    parser.add_argument("--simple_constraint_mode", action="store_true", help="output constraint mode")
    parser.add_argument(
        "--first_token_constraint_mode",
        action="store_true",
        help="""constraint the first token allowed range,
                        use env FIRST_ALLOWED_TOKENS to set the range, like FIRST_ALLOWED_TOKENS=1,2 ..""",
    )
    parser.add_argument(
        "--enable_multimodal", action="store_true", help="Whether or not to allow to load additional multimodal models."
    )
    parser.add_argument(
        "--cache_capacity", type=int, default=200, help="cache server capacity for multimodal resources"
    )
    parser.add_argument(
        "--cache_reserved_ratio", type=float, default=0.5, help="cache server reserved capacity ratio after clear"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
        default=None,
        help="the data type of the model weight",
    )
    parser.add_argument("--return_all_prompt_logprobs", action="store_true", help="return all prompt tokens logprobs")

    parser.add_argument("--use_reward_model", action="store_true", help="use reward model")

    parser.add_argument(
        "--long_truncation_mode",
        type=str,
        choices=[None, "head", "center"],
        default=None,
        help="""use to select the handle way when input token len > max_req_input_len.
                        None : raise Exception
                        head : remove some head tokens to make input token len <= max_req_input_len
                        center : remove some tokens in center loc to make input token len <= max_req_input_len""",
    )
    parser.add_argument("--use_tgi_api", action="store_true", help="use tgi input and ouput format")
    parser.add_argument(
        "--health_monitor", action="store_true", help="check the health of service and restart when error"
    )
    parser.add_argument("--metric_gateway", type=str, default=None, help="address for collecting monitoring metrics")
    parser.add_argument("--job_name", type=str, default="lightllm", help="job name for monitor")
    parser.add_argument(
        "--grouping_key", action="append", default=[], help="grouping_key for the monitor in the form key=value"
    )
    parser.add_argument("--push_interval", type=int, default=10, help="interval of pushing monitoring metrics")
    parser.add_argument(
        "--visual_infer_batch_size", type=int, default=4, help="number of images to process in each inference batch"
    )
    parser.add_argument(
        "--visual_gpu_ids", nargs="+", type=int, default=[0], help="List of GPU IDs to use, e.g., 0 1 2"
    )
    parser.add_argument("--visual_tp", type=int, default=1, help="number of tensort parallel instances for ViT")
    parser.add_argument("--visual_dp", type=int, default=1, help="number of data parallel instances for ViT")
    parser.add_argument(
        "--visual_nccl_ports",
        nargs="+",
        type=int,
        default=[29500],
        help="List of NCCL ports to build a distributed environment for Vit, e.g., 29500 29501 29502",
    )
    parser.add_argument(
        "--enable_monitor_auth", action="store_true", help="Whether to open authentication for push_gateway"
    )
    parser.add_argument("--disable_cudagraph", action="store_true", help="Disable the cudagraph of the decoding stage")
    parser.add_argument(
        "--graph_max_batch_size",
        type=int,
        default=16,
        help="""Maximum batch size that can be captured by the cuda graph for decodign stage.
                The default value is 8. It will turn into eagar mode if encounters a larger value.""",
    )
    parser.add_argument(
        "--graph_max_len_in_batch",
        type=int,
        default=8192,
        help="""Maximum sequence length that can be captured by the cuda graph for decodign stage.
                The default value is 8192. It will turn into eagar mode if encounters a larger value. """,
    )
    return parser
