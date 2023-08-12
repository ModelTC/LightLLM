#!/bin/env python3
import os
import argparse

args = argparse.ArgumentParser()
group_container = args.add_argument_group("container")
group_container.add_argument(
    "--image",
    type=str,
    default="ghcr.io/modeltc/lightllm:main",
    help="default to ghcr.io/modeltc/lightllm:main",
)
group_container.add_argument(
    "--name", type=str, required=False, help="set a name to the container"
)
group_container.add_argument(
    "--keep-container",
    "-K",
    action="store_true",
    help="default not to keep the container",
)
group_container.add_argument(
    "--shm-size",
    type=str,
    required=False,
    help="default to half of the RAM size",
)

group_server = args.add_argument_group("server")
group_server.add_argument(
    "-m", "--model", type=str, required=True, help="path to model dir"
)
group_server.add_argument("-p", "--port", type=int, default=8080)
group_server.add_argument(
    "-n", "--num-proc", type=int, default=1, help="number of process/gpus"
)
group_server.add_argument("-mt", "--max-total-tokens", type=int, default=4096)
args = args.parse_args()

model_path = os.path.abspath(args.model)
shm_size = (
    args.shm_size
    if args.shm_size
    else (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2)
)

launch_args = [
    "docker",
    "run",
    "-it",
    "--gpus",
    "all",
    "-p",
    f"{args.port}:{args.port}",
    "-v",
    f"{model_path}:{model_path}",
    "--shm-size",
    str(shm_size),
]
if args.name:
    launch_args.extend(["--name", args.name])
if not args.keep_container:
    launch_args.append("--rm")

launch_args.append(args.image)
launch_args.extend(
    [
        "/bin/bash",
        "/lightllm/tools/resolve_ptx_version",
        "python",
        "-m",
        "lightllm.server.api_server",
        "--model_dir",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        args.port,
        "--tp",
        args.num_proc,
    ]
)

launch_args = list(map(str, launch_args))
print(f'launching: {" ".join(launch_args)}')
os.execvp(launch_args[0], launch_args)
