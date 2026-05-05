import os
import sys
import time
import uuid
import subprocess
import signal
from lightllm.utils.net_utils import alloc_can_use_network_port, PortLocker
from lightllm.utils.start_utils import process_manager, kill_recursive
from .metrics.manager import start_metric_manager
from .embed_cache.manager import start_cache_manager
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import set_env_start_args, set_unique_server_name, get_unique_server_name
from lightllm.utils.envs_utils import get_lightllm_gunicorn_keep_alive
from .detokenization.manager import start_detokenization_process
from .router.manager import start_router_process
from lightllm.utils.process_check import is_process_active
from lightllm.utils.multinode_utils import send_and_receive_node_ip
from lightllm.utils.redis_utils import start_redis_service
from lightllm.utils.shm_size_check import check_recommended_shm_size
from lightllm.utils.config_utils import has_audio_module, has_vision_module, is_linear_att_mixed_model

logger = init_logger(__name__)


def setup_signal_handlers(http_server_process, process_manager):
    def signal_handler(sig, frame):
        if sig == signal.SIGINT:
            logger.info("Received SIGINT (Ctrl+C), forcing immediate exit...")
            if http_server_process:
                kill_recursive(http_server_process)

            process_manager.terminate_all_processes()
            logger.info("All processes have been forcefully terminated.")
            sys.exit(0)
        elif sig == signal.SIGTERM:
            logger.info("Received SIGTERM, shutting down gracefully...")
            if http_server_process and http_server_process.poll() is None:
                http_server_process.send_signal(signal.SIGTERM)

                start_time = time.time()
                while (time.time() - start_time) < 60:
                    if not is_process_active(http_server_process.pid):
                        logger.info("httpserver exit")
                        break
                    time.sleep(1)

                if time.time() - start_time < 60:
                    logger.info("HTTP server has exited gracefully")
                else:
                    logger.warning("HTTP server did not exit in time, killing it...")
                    kill_recursive(http_server_process)

            process_manager.terminate_all_processes()
            logger.info("All processes have been terminated gracefully.")
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"start process pid {os.getpid()}")
    if http_server_process:
        logger.info(f"http server pid {http_server_process.pid}")
    return


def normal_or_p_d_start(args):
    from lightllm.server.core.objs.start_args_type import StartArgs

    args: StartArgs = args

    set_unique_server_name(args)

    if args.enable_mps:
        from lightllm.utils.device_utils import enable_mps

        enable_mps()

    if args.run_mode not in ["normal", "prefill", "decode", "nixl_prefill", "nixl_decode", "visual_only"]:
        return

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
    if args.run_mode in ["decode", "nixl_decode"]:
        args.disable_audio = True
        args.disable_vision = True

    if args.disable_vision and args.disable_audio:
        args.enable_multimodal = False
    else:
        args.enable_multimodal = True

    if args.enable_cpu_cache:
        # 生成一个用于创建cpu kv cache的共享内存id。
        args.cpu_kv_cache_shm_id = uuid.uuid1().int % 123456789

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
        args.running_max_req_size = 3
        args.batch_max_tokens = 2048
        args.chunked_prefill_size = 1024
        args.mem_fraction = 0.85
        logger.info(
            f"performance_mode is personal, set running_max_req_size to 3,"
            f"batch_max_tokens to 2048, chunked_prefill_size to 1024, mem_fraction to 0.85"
        )

    if not args.disable_shm_warning:
        check_recommended_shm_size(args)

    assert args.zmq_mode in ["tcp://", "ipc:///tmp/"]
    # 确保单机上多实列不冲突
    if args.zmq_mode == "ipc:///tmp/":
        zmq_mode = f"{args.zmq_mode}_{get_unique_server_name()}_"
        args.zmq_mode = None  # args 的参数不能直接设置，只能先设置None，再设置才能成功
        args.zmq_mode = zmq_mode
        logger.info(f"zmq mode head: {args.zmq_mode}")

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
    if args.return_all_prompt_logprobs:
        assert args.disable_dynamic_prompt_cache is True, "need add --disable_dynamic_prompt_cache"
        assert args.disable_chunked_prefill is True, "need add --disable_chunked_prefill"

    # FP8 KV cache mode checks
    if args.llm_kv_type in ["fp8kv_sph", "fp8kv_spt"]:
        assert (
            args.kv_quant_calibration_config_path is not None
        ), "fp8kv inference mode requires --kv_quant_calibration_config_path. "

    if args.enable_prefill_microbatch_overlap or args.enable_decode_microbatch_overlap:
        args.enable_tpsp_mix_mode = True

    if args.enable_dp_prefill_balance:
        assert args.enable_tpsp_mix_mode and args.dp > 1, "need set --enable_tpsp_mix_mode firstly and --dp > 1"

    # mtp params check
    if args.mtp_mode is not None:
        assert args.mtp_draft_model_dir is not None
        assert args.mtp_step > 0
    else:
        assert args.mtp_draft_model_dir is None
        assert args.mtp_step == 0

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
            assert args.batch_max_tokens >= args.max_req_total_len, f"batch_max_tokens must >= max_req_total_len"
            f"but got {args.batch_max_tokens}, {args.max_req_total_len}"
    else:
        # chunked 模式下
        if args.batch_max_tokens is None:
            args.batch_max_tokens = 16384 // args.dp
        if args.chunked_prefill_size is None:
            args.chunked_prefill_size = args.batch_max_tokens // 2
        assert (
            args.batch_max_tokens >= args.chunked_prefill_size
        ), "chunked prefill mode, batch_max_tokens must >= chunked_prefill_size, "
        f"but got {args.batch_max_tokens}, {args.chunked_prefill_size}"

    # linear att cache 参数自动设置
    if args.linear_att_cache_size is None:
        # linear_att_cache_size 只会在 qwen3.5 等混合线性层模型中生效。
        args.linear_att_cache_size = args.running_max_req_size * 2

    if args.enable_cpu_cache and is_linear_att_mixed_model(args.model_dir):
        args.cpu_cache_token_page_size = args.linear_att_hash_page_size * args.linear_att_page_block_num
        logger.info(f"set cpu_cache_token_page_size to {args.cpu_cache_token_page_size} for linear hybrid att model")

    # help to manage data stored on Ceph
    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_prepare

        s3_model_prepare(args.model_dir)

    # 如果args.eos_id 是 None, 从 config.json 中读取 eos_token_id 相关的信息，赋值给 args
    if args.eos_id is None:
        from lightllm.utils.config_utils import get_eos_token_ids

        args.eos_id = get_eos_token_ids(args.model_dir)

    if args.data_type is None:
        from lightllm.utils.config_utils import get_dtype

        args.data_type = get_dtype(args.model_dir)
        assert args.data_type in ["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"]

    already_uesd_ports = [args.port]
    if args.nccl_port is not None:
        already_uesd_ports.append(args.nccl_port)
    if args.pd_decode_rpyc_port is not None:
        already_uesd_ports.append(args.pd_decode_rpyc_port)
    if args.visual_nccl_ports is not None:
        already_uesd_ports.extend(args.visual_nccl_ports[: args.visual_dp])
    if not args.disable_audio and args.audio_nccl_ports is not None:
        already_uesd_ports.extend(args.audio_nccl_ports[: args.audio_dp])

    # 提前锁定端口，防止在单个机器上启动多个实列的时候，要到模型启动的时候才能
    # 捕获到端口设置冲突的问题
    ports_locker = PortLocker(already_uesd_ports)
    ports_locker.lock_port()

    node_world_size = args.tp // args.nnodes
    can_use_ports = alloc_can_use_network_port(
        num=10 + node_world_size + args.visual_dp * args.visual_tp + args.visual_dp + args.audio_dp,
        used_ports=already_uesd_ports,
    )
    logger.info(f"alloced ports: {can_use_ports}")
    (
        nccl_port,
        router_port,
        detokenization_port,
        http_server_port,
        visual_port,
        audio_port,
        cache_port,
        metric_port,
        multi_level_kv_cache_port,
        pd_decode_rpyc_port,
    ) = can_use_ports[0:10]
    can_use_ports = can_use_ports[10:]

    if args.visual_nccl_ports is None:
        args.visual_nccl_ports = can_use_ports[: args.visual_dp]
        can_use_ports = can_use_ports[args.visual_dp :]
    else:
        args.visual_nccl_ports = args.visual_nccl_ports[: args.visual_dp]

    if args.audio_nccl_ports is None:
        args.audio_nccl_ports = can_use_ports[: args.audio_dp]
        can_use_ports = can_use_ports[args.audio_dp :]
    else:
        args.audio_nccl_ports = args.audio_nccl_ports[: args.audio_dp]

    # 将申请好的端口放入args参数中
    if args.nccl_port is None:
        args.nccl_port = nccl_port
    if args.pd_decode_rpyc_port is None:
        args.pd_decode_rpyc_port = pd_decode_rpyc_port
    args.router_port = router_port
    args.detokenization_port = detokenization_port
    args.http_server_port = http_server_port
    args.visual_port = visual_port
    args.audio_port = audio_port
    args.cache_port = cache_port
    args.metric_port = metric_port
    args.multi_level_kv_cache_port = multi_level_kv_cache_port
    # 申请在 p d 分离模式下，会用的端口
    args.pd_node_infer_rpyc_ports = can_use_ports[0:node_world_size]
    # p d 分离模式下用于标识节点的id
    args.pd_node_id = uuid.uuid4().int
    # p 节点用来建立torch kv 传输分布组的可用端口范围
    args.pd_p_allowed_port_min = 20000
    args.pd_p_allowed_port_max = 30000

    # p d 分离模式下，decode节点的调度间隙是0
    if args.run_mode == "decode":
        args.router_max_wait_tokens = 0

    send_and_receive_node_ip(args)  # 多机用于收发node ip
    # dp 必须 > 1
    if args.enable_dp_prompt_cache_fetch and args.dp <= 1:
        args.enable_dp_prompt_cache_fetch = False
        logger.warning(
            """dp <= 1 does not support dp_prompt_cache_fetch;
            overriding enable_dp_prompt_cache_fetch to False"""
        )

    set_env_start_args(args)
    logger.info(f"all start args:{args}")

    ports_locker.release_port()

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

    # 启动 Hypercorn
    command = [
        "hypercorn",
        "--workers",
        f"{args.httpserver_workers}",
        "--bind",
        f"{args.host}:{args.port}",
        "--log-level",
        "info",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "lightllm.server.api_http:app",
        "--keep-alive",
        f"{get_lightllm_gunicorn_keep_alive()}",
    ]

    # 启动子进程
    http_server_process = subprocess.Popen(command)

    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_clear

        s3_model_clear(args.model_dir)

    if args.health_monitor:
        from lightllm.server.health_monitor.manager import start_health_check_process

        process_manager.start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])
    setup_signal_handlers(http_server_process, process_manager)
    http_server_process.wait()
    return


def pd_master_start(args):
    set_unique_server_name(args)
    if args.run_mode != "pd_master":
        return

    # when use config_server to support multi pd_master node, we
    # need generate unique node id for each pd_master node.
    # otherwise, we use the 0 for single pd_master node.
    if args.config_server_host and args.config_server_port:
        args.pd_node_id = uuid.uuid4().int
    else:
        args.pd_node_id = 0

    logger.info(f"use tgi api: {args.use_tgi_api}")
    logger.info(f"all start args:{args}")

    can_use_ports = alloc_can_use_network_port(
        num=1,
        used_ports=[
            args.port,
        ],
    )
    metric_port = can_use_ports[0]

    args.metric_port = metric_port

    set_env_start_args(args)

    process_manager.start_submodule_processes(
        start_funcs=[
            start_metric_manager,
        ],
        start_args=[(args,)],
    )

    command = [
        "hypercorn",
        "--workers",
        "1",
        "--bind",
        f"{args.host}:{args.port}",
        "--log-level",
        "info",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "lightllm.server.api_http:app",
        "--keep-alive",
        f"{get_lightllm_gunicorn_keep_alive()}",
    ]

    http_server_process = subprocess.Popen(command)

    if args.health_monitor:
        from lightllm.server.health_monitor.manager import start_health_check_process

        process_manager.start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])

    setup_signal_handlers(http_server_process, process_manager)
    http_server_process.wait()


def visual_only_start(args):
    from lightllm.server.core.objs.start_args_type import StartArgs

    args: StartArgs = args
    if args.afs_image_embed_dir is not None:
        os.makedirs(args.afs_image_embed_dir, mode=0o777, exist_ok=True)
        os.chmod(args.afs_image_embed_dir, 0o777)

    already_uesd_ports = []
    already_uesd_ports.append(args.visual_rpyc_port)
    can_use_ports = alloc_can_use_network_port(
        num=5 + args.visual_dp * args.visual_tp + args.visual_dp,
        used_ports=already_uesd_ports,
    )

    if args.visual_gpu_ids is None:
        args.visual_gpu_ids = list(range(args.visual_dp * args.visual_tp))
    if args.visual_infer_batch_size is None:
        args.visual_infer_batch_size = args.visual_dp
    if args.data_type is None:
        from lightllm.utils.config_utils import get_dtype

        args.data_type = get_dtype(args.model_dir)
        assert args.data_type in ["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"]

    logger.info(f"alloced ports: {can_use_ports}")

    args.visual_nccl_ports = can_use_ports[: args.visual_dp]
    can_use_ports = can_use_ports[args.visual_dp :]
    args.visual_node_id = uuid.uuid4().int

    logger.info(f"all start args:{args}")

    set_env_start_args(args)

    from .visualserver.visual_only_manager import start_visual_process

    process_manager.start_submodule_processes(
        start_funcs=[
            start_visual_process,
        ],
        start_args=[
            (args,),
        ],
    )
    setup_signal_handlers(None, process_manager)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        process_manager.terminate_all_processes()
        logger.info("All processes have been terminated gracefully.")
        sys.exit(0)


def config_server_start(args):
    set_unique_server_name(args)
    if args.run_mode != "config_server":
        return

    logger.info(f"all start args:{args}")

    if args.config_server_visual_redis_port is not None:
        start_redis_service(args)

    set_env_start_args(args)

    command = [
        "hypercorn",
        "--workers",
        "1",
        "--bind",
        f"{args.config_server_host}:{args.config_server_port}",
        "--log-level",
        "info",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "lightllm.server.config_server.api_http:app",
        "--keep-alive",
        f"{get_lightllm_gunicorn_keep_alive()}",
    ]

    http_server_process = subprocess.Popen(command)
    setup_signal_handlers(http_server_process, process_manager)
    http_server_process.wait()
