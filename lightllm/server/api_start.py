import multiprocessing as mp
import os
import uuid
import subprocess
import math
from lightllm.utils.start_utils import process_manager
from .metrics.manager import start_metric_manager
from .embed_cache.manager import start_cache_manager
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import set_env_start_args, set_unique_server_name, get_unique_server_name
from lightllm.utils.shm_port_args import get_shm_port_args
from lightllm.utils.net_utils import validate_ports
from .detokenization.manager import start_detokenization_process
from .router.manager import start_router_process
from lightllm.utils.multinode_utils import send_and_receive_node_ip
from lightllm.utils.redis_utils import start_redis_service
from lightllm.utils.shm_size_check import check_recommended_shm_size
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.config_utils import (
    has_audio_module,
    has_vision_module,
    is_linear_att_mixed_model,
    auto_set_max_req_total_len,
    get_model_type,
    auto_set_fused_shared_experts,
    auto_set_response_parsers,
)
from lightllm.utils.dist_check_utils import auto_configure_allreduce_flags_from_args
from lightllm.utils.device_utils import is_sm100_gpu
from lightllm.utils.auto_shm_cleanup import register_sysv_shm_for_cleanup

logger = init_logger(__name__)


def _set_envs_and_config(args: StartArgs):
    mp.set_start_method("spawn", force=True)


def _launch_subprocesses(args: StartArgs):
    _set_envs_and_config(args)

    auto_set_max_req_total_len(args)
    auto_set_fused_shared_experts(args)
    set_unique_server_name(args)
    model_type = get_model_type(args.model_dir)
    if args.pd_kv_page_num is None:
        args.pd_kv_page_num = 8 if model_type == "deepseek_v4" else 16
    if args.pd_kv_page_size is None:
        args.pd_kv_page_size = 2048 if model_type == "deepseek_v4" else 1024
    if args.enable_cpu_cache and model_type == "deepseek_v4" and args.llm_kv_type in (None, "None"):
        args.llm_kv_type = "fp8kv_dsa"

    if args.enable_mps:
        from lightllm.utils.device_utils import enable_mps

        enable_mps()

    if args.run_mode not in ["normal", "prefill", "decode", "visual_only"]:
        return

    if args.run_mode in ("prefill", "decode") and model_type == "deepseek_v4":
        if args.tp != args.dp:
            raise ValueError("DeepSeek-V4 PD requires one TP rank per DP replica (--tp must equal --dp)")

    # 通过模型的参数判断是否是多模态模型，包含哪几种模态, 并设置是否启动相应得模块
    if args.disable_vision is None:
        if has_vision_module(args.model_dir):
            args.disable_vision = False
        else:
            args.disable_vision = True
    if args.disable_audio is None:
        if has_audio_module(args.model_dir):
            args.disable_audio = False
        else:
            args.disable_audio = True

    # pd 分离模式下，不启动多模态的模块
    if args.run_mode == "decode":
        args.disable_audio = True
        args.disable_vision = True

    if args.disable_vision and args.disable_audio:
        args.enable_multimodal = False
    else:
        args.enable_multimodal = True

    if args.enable_cpu_cache:
        # 生成一个用于创建cpu kv cache的共享内存id。
        args.cpu_kv_cache_shm_id = uuid.uuid1().int % 123456789
        register_sysv_shm_for_cleanup(args.cpu_kv_cache_shm_id)

    if args.enable_multimodal:
        args.multi_modal_cache_shm_id = uuid.uuid1().int % 123456789

    # 调度参数的自动设置, 人工设置则听人工的
    if args.router_token_ratio is None:
        if args.run_mode in ["normal"]:
            args.router_token_ratio = 0.85
        else:
            # pd 分离模式下，不开启高级调度
            args.router_token_ratio = 0.0
    # 部分模式还不能支持与高级动态调度算法协同，to do.
    if args.diverse_mode:
        assert args.router_token_ratio == 0.0

    # performance_mode 参数处理
    if args.performance_mode == "personal":
        args.running_max_req_size = 6
        args.batch_max_tokens = 2048
        args.chunked_prefill_size = 1024
        args.embed_cache_storage_size = 0.8
        args.graph_max_batch_size = 6
        logger.info(
            f"performance_mode is personal, set running_max_req_size to 3,"
            f"batch_max_tokens to 2048, chunked_prefill_size to 1024,"
            f"graph_max_batch_size to 32"
        )

    dp_size_in_node = max(1, args.dp // args.nnodes)
    args.per_dp_running_max_req_size = args.running_max_req_size // dp_size_in_node
    args.graph_max_batch_size = min(args.graph_max_batch_size, args.per_dp_running_max_req_size)
    logger.info(
        "set per-DP running request limit: global=%d, local_dp=%d, per_dp=%d, graph_max_batch_size=%d",
        args.running_max_req_size,
        dp_size_in_node,
        args.per_dp_running_max_req_size,
        args.graph_max_batch_size,
    )

    if not args.disable_shm_warning:
        check_recommended_shm_size(args)

    assert args.zmq_mode in ["tcp://", "ipc:///tmp/"]

    logger.info(f"use tgi api: {args.use_tgi_api}")

    # 当使用config_server来初始化nccl时，nccl_host和config_server_host必须一致
    if args.use_config_server_to_init_nccl:
        assert args.config_server_host == args.nccl_host

    assert (
        args.mem_fraction > 0 and args.mem_fraction < 1
    ), f"Invalid mem_fraction {args.mem_fraction}, The expected value is between 0 and 1."

    if args.graph_max_len_in_batch == 0:
        args.graph_max_len_in_batch = args.max_req_total_len

    # mode setting check.
    if args.output_constraint_mode != "none":
        assert args.disable_dynamic_prompt_cache is False
        assert args.disable_chunked_prefill is False
    if args.token_healing_mode:
        assert args.disable_dynamic_prompt_cache is False
        assert args.disable_chunked_prefill is False
    if args.diverse_mode:
        assert args.disable_dynamic_prompt_cache is False
        assert args.disable_chunked_prefill is False
    if args.use_reward_model:
        assert args.disable_dynamic_prompt_cache is True, "need add --disable_dynamic_prompt_cache"
        assert args.disable_chunked_prefill is True, "need add --disable_chunked_prefill"
    # FP8 KV cache mode checks
    if args.llm_kv_type in ["fp8kv_sph", "fp8kv_spt"]:
        assert (
            args.kv_quant_calibration_config_path is not None
        ), "fp8kv inference mode requires --kv_quant_calibration_config_path. "

    if args.enable_prefill_microbatch_overlap or args.enable_decode_microbatch_overlap:
        args.enable_tpsp_mix_mode = True

    if args.enable_prefill_decode_mixed:
        assert args.run_mode == "normal", "--enable_prefill_decode_mixed only supports run_mode normal"

    if args.enable_dp_prefill_balance:
        assert args.enable_tpsp_mix_mode and args.dp > 1, "need set --enable_tpsp_mix_mode firstly and --dp > 1"

    if args.enable_prefill_eplb:
        assert args.enable_ep_moe, "--enable_prefill_eplb requires --enable_ep_moe"
        assert not args.enable_prefill_cudagraph, "--enable_prefill_eplb does not support --enable_prefill_cudagraph"
        # EPLB updates expert weights in place, but SM100 Mega-MoE caches transformed weights by tensor data_ptr.
        assert not is_sm100_gpu(), "--enable_prefill_eplb does not support SM100"
        assert (
            args.eplb_num_redundant_experts_per_rank > 0
        ), "--eplb_num_redundant_experts_per_rank must be greater than 0"
        assert args.mtp_mode is None, "--enable_prefill_eplb does not support MTP modes"

    if args.enable_ep_moe:
        allowed_ep_prefill_att_backends = {"auto", "fa3", "triton", "flashqla"}
        for backend in args.llm_prefill_att_backend:
            assert backend in allowed_ep_prefill_att_backends, (
                "When --enable_ep_moe is enabled, --llm_prefill_att_backend must be one of "
                f"{sorted(allowed_ep_prefill_att_backends)}; flashinfer is not supported."
            )
        allowed_ep_decode_att_backends = {"auto", "fa3", "triton"}
        for backend in args.llm_decode_att_backend:
            assert backend in allowed_ep_decode_att_backends, (
                "When --enable_ep_moe is enabled, --llm_decode_att_backend must be one of "
                f"{sorted(allowed_ep_decode_att_backends)}; flashinfer is not supported."
            )

    # mtp params check
    if args.mtp_mode is not None:
        if args.mtp_draft_model_dir is None:
            args.mtp_draft_model_dir = [args.model_dir] * args.mtp_step
        assert args.mtp_step > 0
    else:
        assert args.mtp_draft_model_dir is None
        assert args.mtp_step == 0

    # automatically set visual_dp based on visual_tp and tp.
    # In visual proxy mode keep the caller-provided visual_dp / visual_tp.
    if not args.visual_use_proxy_mode and args.visual_tp < args.tp and args.tp % args.visual_tp == 0:
        args.visual_dp = args.tp // args.visual_tp
    if args.afs_image_embed_dir is not None:
        os.makedirs(args.afs_image_embed_dir, mode=0o777, exist_ok=True)
        os.chmod(args.afs_image_embed_dir, 0o777)

    # 检查GPU数量是否足够
    if args.visual_gpu_ids is None:
        args.visual_gpu_ids = list(range(args.visual_dp * args.visual_tp))
    total_required_gpus = args.visual_dp * args.visual_tp
    if len(args.visual_gpu_ids) < total_required_gpus:
        raise ValueError(
            f"Not enough GPUs specified. You need at least {total_required_gpus}, but got {len(args.visual_gpu_ids)}."
        )
    else:
        args.visual_gpu_ids = args.visual_gpu_ids[:total_required_gpus]

    if args.visual_dp <= 0:
        raise ValueError("visual_dp must be a positive integer.")

    if args.visual_infer_batch_size is None:
        args.visual_infer_batch_size = args.visual_dp

    # 检查visual_infer_batch_size是否合理
    if args.visual_infer_batch_size // args.visual_dp < 1 or args.visual_infer_batch_size % args.visual_dp != 0:
        raise ValueError(
            f"visual_infer_batch_size ({args.visual_infer_batch_size}) must be "
            f"a positive integer multiple of visual_dp ({args.visual_dp})"
        )

    if not args.disable_audio:
        if args.audio_tp != 1:
            raise ValueError(
                "audio_tp > 1 is not supported for the audio encoder yet; use --audio_dp for multi-GPU data parallel."
            )
        if args.audio_gpu_ids is None:
            args.audio_gpu_ids = list(range(args.audio_dp * args.audio_tp))
        total_audio_gpus = args.audio_dp * args.audio_tp
        if len(args.audio_gpu_ids) < total_audio_gpus:
            raise ValueError(
                f"Not enough audio GPUs specified. Need at least {total_audio_gpus}, "
                f"but got {len(args.audio_gpu_ids)}."
            )
        args.audio_gpu_ids = args.audio_gpu_ids[:total_audio_gpus]
        if args.audio_dp <= 0:
            raise ValueError("audio_dp must be a positive integer.")
        if args.audio_infer_batch_size is None:
            args.audio_infer_batch_size = args.audio_dp * 4
        if args.audio_infer_batch_size < 1:
            raise ValueError("audio_infer_batch_size must be >= 1.")
        if args.audio_infer_batch_size // args.audio_dp < 1 or args.audio_infer_batch_size % args.audio_dp != 0:
            raise ValueError(
                f"audio_infer_batch_size ({args.audio_infer_batch_size}) must be "
                f"a positive integer multiple of audio_dp ({args.audio_dp})."
            )

    if args.disable_chunked_prefill:
        args.chunked_prefill_size = args.max_req_total_len
        # 普通模式下
        if args.batch_max_tokens is None:
            args.batch_max_tokens = args.max_req_total_len
        else:
            assert args.batch_max_tokens >= args.max_req_total_len, (
                f"batch_max_tokens must >= max_req_total_len, "
                f"but got {args.batch_max_tokens}, {args.max_req_total_len}"
            )
    else:
        # chunked 模式下
        if args.batch_max_tokens is None:
            args.batch_max_tokens = 16384 // args.dp
        if args.chunked_prefill_size is None:
            args.chunked_prefill_size = args.batch_max_tokens // 2
        assert args.batch_max_tokens >= args.chunked_prefill_size, (
            "chunked prefill mode, batch_max_tokens must >= chunked_prefill_size, "
            f"but got {args.batch_max_tokens}, {args.chunked_prefill_size}"
        )

    # linear att cache 参数自动设置
    if args.linear_att_cache_size is None:
        # linear_att_cache_size 只会在 qwen3.5 等混合线性层模型中生效。
        default_cache_size = args.running_max_req_size * 2
        dp_size_in_node = max(1, args.dp // args.nnodes)
        per_dp_cache_size = max(1, math.ceil(args.running_max_req_size / dp_size_in_node) * 2)
        args.linear_att_cache_size = min(default_cache_size, per_dp_cache_size)

    if args.run_mode == "decode":
        # PD Decode 节点只接收 prompt 末尾位置的 linear attention state，不具备
        # 中间大页边界对应的 state。因此 Decode 节点必须使用默认值关闭大页功能，
        # 避免请求释放时将不完整的大页 state 写入 radix cache 并触发断言。
        args.linear_att_page_block_num = 10000000

    if args.enable_cpu_cache and is_linear_att_mixed_model(args.model_dir):
        args.cpu_cache_token_page_size = args.linear_att_hash_page_size * args.linear_att_page_block_num
        logger.info(f"set cpu_cache_token_page_size to {args.cpu_cache_token_page_size} for linear hybrid att model")
    elif args.enable_cpu_cache and args.cpu_cache_token_page_size is None:
        args.cpu_cache_token_page_size = 2048 if get_model_type(args.model_dir) == "deepseek_v4" else 256

    # help to manage data stored on Ceph
    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_prepare

        s3_model_prepare(args.model_dir)

    # 如果args.eos_id 是 None, 从 config.json 中读取 eos_token_id 相关的信息，赋值给 args
    if args.eos_id is None:
        from lightllm.utils.config_utils import get_eos_token_ids

        args.eos_id = get_eos_token_ids(args.model_dir)

    auto_set_response_parsers(args)

    if args.data_type is None:
        from lightllm.utils.config_utils import get_dtype

        args.data_type = get_dtype(args.model_dir)
        assert args.data_type in [
            "fp16",
            "float16",
            "bf16",
            "bfloat16",
            "fp32",
            "float32",
        ]

    set_unique_server_name(args)

    # 确保单机上多实列不冲突
    if args.zmq_mode == "ipc:///tmp/":
        zmq_mode = f"{args.zmq_mode}_{get_unique_server_name()}_"
        args.zmq_mode = None  # args 的参数不能直接设置，只能先设置None，再设置才能成功
        args.zmq_mode = zmq_mode
        logger.info(f"zmq mode head: {args.zmq_mode}")

    # p d 分离模式下用于标识节点的id
    args.pd_node_id = uuid.uuid4().int
    # p d 分离模式下，decode节点的调度间隙是0
    if args.run_mode == "decode":
        args.router_max_wait_tokens = 0

    # dp 必须 > 1
    if args.enable_dp_prompt_cache_fetch and args.dp <= 1:
        args.enable_dp_prompt_cache_fetch = False
        logger.warning(
            """dp <= 1 does not support dp_prompt_cache_fetch;
            overriding enable_dp_prompt_cache_fetch to False"""
        )

    auto_configure_allreduce_flags_from_args(args)

    # 校验用户已设置端口冲突（对齐原 PortManager 启动检查范围）
    ports_to_check = [args.port]
    if args.dp == 1 and args.nnodes > 1:
        ports_to_check.extend([args.multinode_httpmanager_port, args.multinode_router_gloo_port])
    if args.node_rank == 0 and args.nccl_port is not None:
        ports_to_check.append(args.nccl_port)
    validate_ports(ports_to_check)

    set_env_start_args(args)
    get_shm_port_args(create=True)
    # 多机用于收发node ip, 这个地方修改了args env,所以需要重新设置一下。
    send_and_receive_node_ip(args)
    set_env_start_args(args)
    logger.info(f"all start args:{args}")

    if args.enable_multimodal:
        process_manager.start_submodule_processes(
            start_funcs=[
                start_cache_manager,
            ],
            start_args=[(args,)],
        )

    if not args.disable_vision:
        if not args.visual_use_proxy_mode:
            from .visualserver.manager import start_visual_process

            process_manager.start_submodule_processes(
                start_funcs=[
                    start_visual_process,
                ],
                start_args=[
                    (args,),
                ],
            )
        else:
            from .visualserver.proxy_manager import start_visual_process

            process_manager.start_submodule_processes(
                start_funcs=[
                    start_visual_process,
                ],
                start_args=[
                    (args,),
                ],
            )

    if not args.disable_audio:
        from .audioserver.manager import start_audio_process

        process_manager.start_submodule_processes(
            start_funcs=[
                start_audio_process,
            ],
            start_args=[
                (args,),
            ],
        )

    if args.enable_cpu_cache:
        from .multi_level_kv_cache.manager import start_multi_level_kv_cache_manager

        process_manager.start_submodule_processes(
            start_funcs=[
                start_multi_level_kv_cache_manager,
            ],
            start_args=[(args,)],
        )

    process_manager.start_submodule_processes(
        start_funcs=[
            start_metric_manager,
        ],
        start_args=[(args,)],
    )

    process_manager.start_submodule_processes(
        start_funcs=[start_router_process, start_detokenization_process],
        start_args=[
            (args,),
            (args,),
        ],
    )

    return process_manager


def _hypercorn_config_args(args: StartArgs):
    if args.hypercorn_config is not None:
        return ["--config", args.hypercorn_config]
    return ["--keep-alive", "10"]


def normal_or_p_d_start(args: StartArgs):
    process_manager = _launch_subprocesses(args)

    # 启动 Hypercorn
    command = [
        "hypercorn",
        *_hypercorn_config_args(args),
        "--workers",
        f"{args.httpserver_workers}",
        "--bind",
        f"{args.host}:{get_shm_port_args().port}",
        "--log-level",
        "info",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "lightllm.server.api_http:app",
    ]

    # 启动子进程
    http_server_process = subprocess.Popen(command)

    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_clear

        s3_model_clear(args.model_dir)

    if args.health_monitor:
        from lightllm.server.health_monitor.manager import start_health_check_process

        process_manager.start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])
    process_manager.setup_signal_handlers(http_server_process)
    process_manager.supervise_processes(http_server_process)


def pd_master_start(args: StartArgs):
    _set_envs_and_config(args)
    set_unique_server_name(args)
    if args.run_mode != "pd_master":
        return

    if args.enable_cpu_cache and get_model_type(args.model_dir) == "deepseek_v4":
        raise ValueError("DeepSeek-V4 CPU cache does not support pd_master")

    auto_set_max_req_total_len(args)
    auto_set_response_parsers(args)

    # when use config_server to support multi pd_master node, we
    # need generate unique node id for each pd_master node.
    # otherwise, we use the 0 for single pd_master node.
    if args.config_server_host and args.config_server_port:
        args.pd_node_id = uuid.uuid4().int
    else:
        args.pd_node_id = 0

    logger.info(f"use tgi api: {args.use_tgi_api}")

    validate_ports([args.port])
    set_env_start_args(args)
    get_shm_port_args(create=True)
    logger.info(f"all start args:{args}")

    process_manager.start_submodule_processes(
        start_funcs=[
            start_metric_manager,
        ],
        start_args=[(args,)],
    )

    command = [
        "hypercorn",
        *_hypercorn_config_args(args),
        "--workers",
        "1",
        "--bind",
        f"{args.host}:{get_shm_port_args().port}",
        "--log-level",
        "info",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "lightllm.server.api_http:app",
    ]

    http_server_process = subprocess.Popen(command)

    if args.health_monitor:
        from lightllm.server.health_monitor.manager import start_health_check_process

        process_manager.start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])

    process_manager.setup_signal_handlers(http_server_process)
    process_manager.supervise_processes(http_server_process)


def visual_only_start(args):
    from lightllm.server.core.objs.start_args_type import StartArgs

    args: StartArgs = args
    _set_envs_and_config(args)
    if args.afs_image_embed_dir is not None:
        os.makedirs(args.afs_image_embed_dir, mode=0o777, exist_ok=True)
        os.chmod(args.afs_image_embed_dir, 0o777)

    set_unique_server_name(args)

    if args.visual_gpu_ids is None:
        args.visual_gpu_ids = list(range(args.visual_dp * args.visual_tp))
    if args.visual_infer_batch_size is None:
        args.visual_infer_batch_size = args.visual_dp
    if args.data_type is None:
        from lightllm.utils.config_utils import get_dtype

        args.data_type = get_dtype(args.model_dir)
        assert args.data_type in [
            "fp16",
            "float16",
            "bf16",
            "bfloat16",
            "fp32",
            "float32",
        ]

    args.visual_node_id = uuid.uuid4().int

    ports_to_check = []
    if args.visual_rpyc_port is not None:
        ports_to_check.append(args.visual_rpyc_port)
    validate_ports(ports_to_check)
    set_env_start_args(args)
    get_shm_port_args(create=True)
    logger.info(f"all start args:{args}")

    from .visualserver.visual_only_manager import start_visual_process

    process_manager.start_submodule_processes(
        start_funcs=[
            start_visual_process,
        ],
        start_args=[
            (args,),
        ],
    )
    process_manager.setup_signal_handlers()
    process_manager.supervise_processes()


def config_server_start(args):
    set_unique_server_name(args)
    if args.run_mode != "config_server":
        return

    ports_to_check = [args.config_server_port]
    if args.config_server_visual_redis_port is not None:
        ports_to_check.append(args.config_server_visual_redis_port)
    validate_ports(ports_to_check)
    set_env_start_args(args)
    get_shm_port_args(create=True)
    logger.info(f"all start args:{args}")

    if args.config_server_visual_redis_port is not None:
        start_redis_service(args)

    command = [
        "hypercorn",
        *_hypercorn_config_args(args),
        "--workers",
        "1",
        "--bind",
        f"{args.config_server_host}:{get_shm_port_args().config_server_port}",
        "--log-level",
        "info",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "lightllm.server.config_server.api_http:app",
    ]

    http_server_process = subprocess.Popen(command)
    process_manager.setup_signal_handlers(http_server_process)
    process_manager.supervise_processes(http_server_process)
