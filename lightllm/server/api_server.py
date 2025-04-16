import torch
from .api_cli import make_argument_parser

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")  # this code will not be ok for settings to fork to subprocess
    parser = make_argument_parser()
    args = parser.parse_args()
    from .api_start import pd_master_start, normal_or_p_d_start, config_server_start

    if args.run_mode == "pd_master":
        pd_master_start(args)
    elif args.run_mode == "config_server":
        config_server_start(args)
    else:
        normal_or_p_d_start(args)
