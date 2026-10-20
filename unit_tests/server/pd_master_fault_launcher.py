"""Test-only launcher that injects failures at real startup boundaries."""

import argparse
import json
from pathlib import Path
import tempfile

import psutil

from lightllm.server.api_cli import add_cli_args
from lightllm.server.api_server import launch_server
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.httpserver_for_pd_master import multiprocess
from lightllm.utils.envs_utils import get_unique_server_name


def observe_cleanup(observation_path):
    original_stop = multiprocess.process_manager.terminate_all_processes

    def stop():
        prefix = "lightllm-pd-master-" + get_unique_server_name() + "-*"
        directories = list(Path(tempfile.gettempdir()).glob(prefix))
        before = [directory.exists() for directory in directories]
        processes = list(multiprocess.process_manager.processes)
        original_stop()

        def process_stopped(process):
            try:
                return not process.is_running()
            except psutil.NoSuchProcess:
                return True

        observation_path.write_text(
            json.dumps(
                {
                    "runtime_paths": [str(directory) for directory in directories],
                    "runtime_before_stop": before,
                    "runtime_after_stop": [directory.exists() for directory in directories],
                    "backends_stopped": all(process_stopped(process) for process in processes),
                }
            ),
            encoding="utf-8",
        )

    multiprocess.process_manager.terminate_all_processes = stop


def inject_failure(stage, haproxy_binary):
    if stage == "backend_start":
        original_get_context = multiprocess.multiprocessing.get_context

        def get_context(method):
            context = original_get_context(method)
            process_count = 0

            class ProcessContext:
                def Process(self, *args, **kwargs):
                    nonlocal process_count
                    process_count += 1
                    if process_count == 2:
                        process = context.Process(*args, **kwargs)

                        def fail_to_start():
                            raise RuntimeError("injected backend startup failure")

                        process.start = fail_to_start
                        return process
                    return context.Process(*args, **kwargs)

            return ProcessContext()

        multiprocess.multiprocessing.get_context = get_context
    elif stage == "proxy_start":
        original_popen = multiprocess.subprocess.Popen

        def popen(command, *args, **kwargs):
            if command[0] == haproxy_binary and kwargs.get("start_new_session"):
                raise OSError("injected HAProxy startup failure")
            return original_popen(command, *args, **kwargs)

        multiprocess.subprocess.Popen = popen


if __name__ == "__main__":
    parser = add_cli_args(argparse.ArgumentParser())
    parser.add_argument("--failure-stage", required=True)
    parser.add_argument("--cleanup-observations", type=Path, required=True)
    options = vars(parser.parse_args())
    failure_stage = options.pop("failure_stage")
    observe_cleanup(options.pop("cleanup_observations"))
    args = StartArgs(**options)
    haproxy_binary = multiprocess.validate_multiprocess_pd_master_args(args)
    inject_failure(failure_stage, haproxy_binary)
    launch_server(args)
