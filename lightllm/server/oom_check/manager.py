import inspect
import os

import setproctitle

from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def should_run_oom_check() -> bool:
    return os.environ.get("LIGHTLLM_CHECK_OOM", "").strip().upper() in ("1", "TRUE", "ON")


def start_oom_check_process(args, pipe_writer):
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::oom_check")
    pipe_writer.send("init ok")

    from lightllm.server.oom_check.runner import run_oom_check

    run_oom_check(
        host="127.0.0.1",
        port=args.port,
        model_dir=args.model_dir,
        trust_remote_code=args.trust_remote_code,
        running_max_req_size=args.running_max_req_size,
        max_req_total_len=args.max_req_total_len,
    )
    logger.info("[OOM_CHECK] subprocess done; exiting.")
