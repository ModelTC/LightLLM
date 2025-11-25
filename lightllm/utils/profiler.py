from dataclasses import dataclass
import os
from typing import Any, Literal, Optional
import threading
import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclass
class ProfilerCmd:
    cmd: Literal["start", "stop"]


class ProcessProfiler:
    def __init__(self, mode: Literal["torch_profiler", "nvtx"], name: Optional[str] = None):
        self.mode: Literal["torch_profiler", "nvtx"] = mode
        self.name: Optional[str] = name
        self.is_active: bool = False
        self.lock = threading.Lock()
        self.tid = threading.get_native_id() if hasattr(threading, "get_native_id") else threading.get_ident()

        logger.warning("-" * 50)
        logger.warning(f"[tgid={os.getpid()} pid={self.tid}] Profiler <{self.name}> initialized with mode: {self.mode}")
        if self.mode == "torch_profiler":
            trace_dir = os.getenv("LIGHTLLM_TRACE_DIR", "./trace")
            self._torch_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,  # additional overhead
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, worker_name=name, use_gzip=True),
            )
            logger.warning(
                "Profiler support for torch.profiler enabled (--enable_profiling=torch_profiler), "
                "trace files will be saved to %s (change it with LIGHTLLM_TRACE_DIR env var)",
                trace_dir,
            )
        elif self.mode == "nvtx":
            self._nvtx_toplevel_mark = "LIGHTLLM_PROFILE"
            logger.warning(
                "Profiler support for NVTX enabled (--enable_profiling=nvtx), toplevel NVTX mark is '%s'\n"
                "you can use it with external profiling tools like NVIDIA Nsight Systems.",
                self._nvtx_toplevel_mark,
            )
            logger.warning(
                "e.g. nsys profile --capture-range=nvtx --nvtx-capture=%s --trace=cuda,nvtx "
                "-e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 [other nsys options] "
                "python -m lightllm.server.api_server --enable_profiling=nvtx [other lightllm options]",
                self._nvtx_toplevel_mark,
            )
        elif self.mode is not None:
            raise ValueError("invalid profiler mode")
        logger.warning("Use /profiler_start and /profiler_stop HTTP GET APIs to start/stop profiling")
        logger.warning("DO NOT enable this feature in production environment")
        logger.warning("-" * 50)

    def _torch_profiler_start(self) -> None:
        torch.cuda.synchronize()
        with self.lock:
            if not hasattr(self, "_torch_profiler_start_tid"):
                # torch profiler only needs to start once per process
                self._torch_profiler_start_tid = self.tid
                self._torch_profiler.start()
        torch.cuda.synchronize()

    def _nvtx_start(self) -> None:
        torch.cuda.synchronize()
        with self.lock:
            if not hasattr(self, "_nvtx_toplevel_ids"):
                self._nvtx_toplevel_ids = {}
            self._nvtx_toplevel_ids[self.tid] = torch.cuda.nvtx.range_start(self._nvtx_toplevel_mark)
        torch.cuda.synchronize()

    def start(self) -> None:
        if self.is_active:
            logger.error("profiler already started, ignore")
            return
        logger.warning(f"[tgid={os.getpid()} pid={self.tid}] Profiler <{self.name}>: profiling start")
        self.is_active = True
        if self.mode == "torch_profiler":
            self._torch_profiler_start()
        elif self.mode == "nvtx":
            self._nvtx_start()

    def _torch_profiler_stop(self) -> None:
        torch.cuda.synchronize()
        with self.lock:
            if hasattr(self, "_torch_profiler_start_tid") and self._torch_profiler_start_tid == self.tid:
                # torch profiler only needs to stop once per process, in the same thread that started it
                del self._torch_profiler_start_tid
                logger.warning(f"Profiler <{self.name}>: torch profiler stopping and saving trace, please wait...")
                try:
                    self._torch_profiler.stop()
                except RuntimeError as e:
                    logger.error(f"Profiler <{self.name}>: torch profiler stop failed: {e}, maybe too short")
                    import traceback

                    traceback.print_exc()
                    return
                logger.warning(f"Profiler <{self.name}>: torch profiler trace saved.")
        torch.cuda.synchronize()

    def _nvtx_stop(self) -> None:
        torch.cuda.synchronize()
        with self.lock:
            if hasattr(self, "_nvtx_toplevel_ids") and self.tid in self._nvtx_toplevel_ids:
                torch.cuda.nvtx.range_end(self._nvtx_toplevel_ids[self.tid])
                del self._nvtx_toplevel_ids[self.tid]
            else:
                logger.error("nvtx profiler stop called without matching start for tid %s", self.tid)
        torch.cuda.synchronize()

    def stop(self) -> None:
        if not self.is_active:
            logger.error("profiler not started, ignore")
            return
        logger.warning(f"[tgid={os.getpid()} pid={self.tid}] Profiler <{self.name}>: profiling stop")
        self.is_active = False
        if self.mode == "torch_profiler":
            self._torch_profiler_stop()
        elif self.mode == "nvtx":
            self._nvtx_stop()

    def cmd(self, cmd_obj: ProfilerCmd) -> None:
        if cmd_obj.cmd == "start":
            self.start()
        elif cmd_obj.cmd == "stop":
            self.stop()
        else:
            raise ValueError(f"invalid profiler ops: {cmd_obj.cmd}")
