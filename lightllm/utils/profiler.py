from dataclasses import dataclass
import os
import threading
import traceback
from typing import Any, Literal, Optional
import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclass
class ProfilerCmd:
    cmd: Literal["start", "stop"]


def _get_thread_id() -> int:
    # Get native thread ID (LWP) for correlation with system tools like htop/nsys
    if hasattr(threading, "get_native_id"):
        return threading.get_native_id()
    return threading.get_ident()


class ProcessProfiler:
    def __init__(
        self,
        mode: Literal["torch_profiler", "nvtx"],
        name: Optional[str] = None,
        use_multi_thread: bool = False,
        torch_profiler_with_stack: bool = True,
    ) -> None:
        """
        Process Level Profiler Manager.
        For multi-threading, set `use_multi_thread=True`
        and call `.multi_thread_helper()` regularly in each worker thread.
        """
        self.mode = mode
        self.name = name or "unnamed"
        self.use_multi_thread = use_multi_thread
        self.torch_profiler_with_stack = torch_profiler_with_stack

        self.is_active: bool = False  # Process-level logical state
        self._threadlocal = threading.local()

        # make sure only one active torch.profiler per process
        self._lock = threading.Lock()
        self._process_torch_profiler_active_tid: int | None = None

        if self.mode == "torch_profiler":
            self._trace_dir = os.getenv("LIGHTLLM_TRACE_DIR", "./trace")
            os.makedirs(self._trace_dir, exist_ok=True)
        elif self.mode == "nvtx":
            self._nvtx_toplevel_mark = "LIGHTLLM_PROFILE"
        else:
            raise ValueError("invalid profiler mode")

        self._log_init_info()

    @property
    def _local(self):
        """Lazy initialization of thread-local storage."""
        if not hasattr(self._threadlocal, "initialized"):
            self._threadlocal.initialized = True
            self._threadlocal.is_active = False
            self._threadlocal.profiler_obj = None
            self._threadlocal.nvtx_range_id = None
        return self._threadlocal

    def _log_init_info(self):
        logger.warning("-" * 50)
        logger.warning(
            f"[pid={os.getpid()} tid={_get_thread_id()}] Profiler <{self.name}> initialized with mode: {self.mode}"
        )
        if self.mode == "torch_profiler":
            logger.warning(
                "Profiler support for torch.profiler enabled (--enable_profiling=torch_profiler), "
                "trace files will be saved to %s (change it with LIGHTLLM_TRACE_DIR env var)",
                self._trace_dir,
            )
        elif self.mode == "nvtx":
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
        logger.warning("Use /profiler_start and /profiler_stop HTTP GET APIs to start/stop profiling")
        logger.warning("DO NOT enable this feature in production environment")
        logger.warning("-" * 50)

    def _torch_profiler_start(self) -> None:
        with self._lock:
            if self._process_torch_profiler_active_tid is not None:
                return
            self._process_torch_profiler_active_tid = _get_thread_id()

        torch.cuda.synchronize()
        worker_name = f"{self.name}_tid{_get_thread_id()}" if self.use_multi_thread else self.name

        trace_handler = torch.profiler.tensorboard_trace_handler(
            self._trace_dir,
            worker_name=worker_name,
            use_gzip=True,
        )

        p = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=None,
            with_stack=self.torch_profiler_with_stack,
            record_shapes=True,
            on_trace_ready=trace_handler,
        )

        self._local.profiler_obj = p
        p.start()
        torch.cuda.synchronize()

    def _nvtx_start(self) -> None:
        torch.cuda.synchronize()
        self._local.nvtx_range_id = torch.cuda.nvtx.range_start(self._nvtx_toplevel_mark)
        torch.cuda.synchronize()

    def _thread_start(self) -> None:
        if self._local.is_active:
            return

        try:
            logger.info(f"[{self.name} @ tid={_get_thread_id()}] Start Profiler.")
            if self.mode == "torch_profiler":
                self._torch_profiler_start()
            elif self.mode == "nvtx":
                self._nvtx_start()

            self._local.is_active = True
        except Exception as e:
            logger.error(
                f"[{self.name} @ tid={_get_thread_id()}] Failed to start profiler in thread {_get_thread_id()}: {e}"
            )
            traceback.print_exc()
            # Reset state on failure to prevent infinite retry loops
            self._local.is_active = False

    def _torch_profiler_stop(self) -> None:
        if self._process_torch_profiler_active_tid != _get_thread_id():
            return

        torch.cuda.synchronize()
        logger.info(f"[{self.name} @ tid={_get_thread_id()}] Saving trace (blocking)...")
        try:
            if self._local.profiler_obj:
                self._local.profiler_obj.stop()
        except Exception as e:
            logger.error(f"[{self.name} @ tid={_get_thread_id()}] Error stopping torch profiler: {e}")
        finally:
            self._local.profiler_obj = None  # Explicitly release reference to allow GC
            self._process_torch_profiler_active_tid = None

        torch.cuda.synchronize()

    def _nvtx_stop(self) -> None:
        torch.cuda.synchronize()
        if self._local.nvtx_range_id is not None:
            torch.cuda.nvtx.range_end(self._local.nvtx_range_id)
            self._local.nvtx_range_id = None
        torch.cuda.synchronize()

    def _thread_stop(self) -> None:
        if not self._local.is_active:
            return

        try:
            if self.mode == "torch_profiler":
                self._torch_profiler_stop()
            elif self.mode == "nvtx":
                self._nvtx_stop()
            logger.info(f"[{self.name} @ tid={_get_thread_id()}] Profiler stopped.")
        except Exception as e:
            logger.error(f"[{self.name} @ tid={_get_thread_id()}] Failed to stop profiler: {e}")
        finally:
            # Mark inactive regardless of success to avoid repeated errors
            self._local.is_active = False

    def start(self) -> None:
        self.is_active = True
        if not self.use_multi_thread:
            self._thread_start()

    def stop(self) -> None:
        self.is_active = False
        if not self.use_multi_thread:
            self._thread_stop()

    def multi_thread_helper(self) -> None:
        """
        **only for multi-threading use cases**
        Worker polling method. Must be called within the inference loop.
        """
        if not self.use_multi_thread:
            return

        # Catch-all to prevent profiler errors from crashing inference logic
        try:
            local_active = self._local.is_active

            if self.is_active and not local_active:
                self._thread_start()
            elif not self.is_active and local_active:
                self._thread_stop()
        except Exception:
            pass

    def cmd(self, cmd_obj: ProfilerCmd) -> None:
        if cmd_obj.cmd == "start":
            self.start()
        elif cmd_obj.cmd == "stop":
            self.stop()
        else:
            raise ValueError(f"Invalid profiler cmd: {cmd_obj.cmd}")
