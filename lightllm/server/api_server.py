import torch
from .api_cli import add_cli_args
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.log_utils import init_logger, set_log_node_role

logger = init_logger(__name__)


def launch_server(args: StartArgs):
    _configure_httpserver_workers(args=args)
    if args.use_dp_split_mode_connect_pd_master and args.run_mode != "prefill":
        raise ValueError("--use_dp_split_mode_connect_pd_master only supports prefill")
    set_log_node_role(args.run_mode)
    from .api_start import (
        pd_master_start,
        pd_master_multi_worker_start,
        normal_or_p_d_start,
        config_server_start,
        visual_only_start,
    )

    try:
        # this code will not be ok for settings to fork to subprocess
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logger.warning(f"Failed to set start method: {e}")
    except Exception as e:
        logger.error(f"Failed to set start method: {e}")
        raise e

    if args.run_mode == "pd_master":
        if args.httpserver_workers < 1:
            raise ValueError("--httpserver_workers must be at least 1")
        if args.httpserver_workers == 1:
            pd_master_start(args)
        else:
            pd_master_multi_worker_start(args)
    elif args.run_mode == "config_server":
        config_server_start(args)
    elif args.run_mode in ["visual_only"]:
        visual_only_start(args)
    else:
        normal_or_p_d_start(args)


def _configure_httpserver_workers(args: StartArgs) -> None:
    """设置 HTTP worker 数；跨机 TP 从节点固定为 1，其他节点保留用户配置或按模式取默认值。"""
    is_multinode_tp_slave = args.dp == 1 and args.nnodes > 1 and args.node_rank > 0
    if args.run_mode in ("normal", "prefill", "decode") and is_multinode_tp_slave:
        # 每个 worker 都会绑定同一个 multinode_httpmanager_port，从节点只能启动一个。
        if args.httpserver_workers not in (None, 1):
            logger.warning("Multinode TP slave requires httpserver_workers=1; overriding the configured value")
        args.httpserver_workers = 1
        return

    if args.httpserver_workers is not None:
        return
    if args.run_mode in ("prefill", "decode"):
        args.httpserver_workers = 2
    else:
        args.httpserver_workers = 1
    logger.info(f"run_mode={args.run_mode}, set httpserver_workers to {args.httpserver_workers}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()

    launch_server(StartArgs(**vars(args)))
