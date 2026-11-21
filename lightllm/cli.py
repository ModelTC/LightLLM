import argparse
import sys

from lightllm.server.api_cli import add_cli_args
from lightllm.cli_defaults import resolve_tp


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="lightllm", description="Serve models with LightLLM.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser(
        "serve",
        help="Start a model server.",
        description="Start a model server. Example: lightllm serve /models/my-model",
        epilog="All existing server options are accepted. Use --help-all to list them.",
        allow_abbrev=False,
    )
    add_cli_args(serve)
    if "--help-all" not in argv:
        common_options = {
            "help",
            "host",
            "port",
            "tp",
            "data_type",
            "mem_fraction",
            "max_req_total_len",
            "running_max_req_size",
        }
        for action in serve._actions:
            if action.dest not in common_options:
                action.help = argparse.SUPPRESS
    serve.add_argument("model", nargs="?", metavar="MODEL", help="Path to the model weights.")
    serve.add_argument("--help-all", action="help", help="Show all server options and exit.")

    values = vars(parser.parse_args(argv))
    values.pop("command")
    model = values.pop("model")
    if model is not None:
        if values["model_dir"] is not None:
            serve.error("specify MODEL or --model_dir, not both")
        values["model_dir"] = model
    if not values["model_dir"] and values["run_mode"] not in ("pd_master", "config_server"):
        serve.error("a model path is required: lightllm serve /models/my-model")

    # Keep the old parser and all explicit launch flags compatible, while the
    # short positional form gets a useful multi-GPU default.
    has_explicit_tp = any(arg == "--tp" or arg.startswith("--tp=") for arg in argv)
    if values["model_dir"] and not has_explicit_tp and values["run_mode"] == "normal":
        values["tp"] = resolve_tp(values["model_dir"])

    from lightllm.server.api_server import launch_server
    from lightllm.server.core.objs.start_args_type import StartArgs

    launch_server(StartArgs(**values))
