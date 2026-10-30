import torch
from .api_cli import add_cli_args
from lightllm.common.quantization import load_quant_config
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.log_utils import init_logger, set_log_node_role

logger = init_logger(__name__)


def launch_server(args: StartArgs):
    for quant_type, quant_cfg in (("quant_type", "quant_cfg"), ("vit_quant_type", "vit_quant_cfg")):
        try:
            load_quant_config(getattr(args, quant_type), getattr(args, quant_cfg), platform=args.hardware_platform)
        except ValueError as e:
            raise ValueError(f"--{quant_type}/--{quant_cfg}: {e}") from e
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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()

    launch_server(StartArgs(**vars(args)))
