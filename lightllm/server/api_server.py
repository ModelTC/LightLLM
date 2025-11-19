import torch
from .api_cli import make_argument_parser
from lightllm.server.core.objs.start_args_type import StartArgs


def launch_server(args: StartArgs):
    from .api_start import pd_master_start, normal_or_p_d_start, config_server_start

    torch.multiprocessing.set_start_method("spawn")  # this code will not be ok for settings to fork to subprocess
    if args.run_mode == "pd_master":
        pd_master_start(args)
    elif args.run_mode == "config_server":
        config_server_start(args)
    else:
        normal_or_p_d_start(args)

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()

    launch_server(StartArgs(**args))