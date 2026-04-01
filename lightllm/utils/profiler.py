from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
import random
import threading
import time
import traceback
from typing import IO, Any, DefaultDict, List, Literal, Optional, Tuple, get_args
import functools
import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class GlobalPerfContext:
    # static control flags
    STATIC_DISABLED = True
    LAZY_SAVE = True

    # runtime control flags (user control)
    global_enabled = True

    cuda_lock = threading.Lock()
    threadlocal_active_counter_stack: dict[int, list['PerfCounter']] = DefaultDict(list)
    recorded_counters: list['PerfCounter'] = []
    _recorded_counters_lock = threading.Lock()

    _initialized = False
    _program_disabled = False
    _effective_enabled = False
    _force_do_profile_once = False
    _file_save_lock = threading.Lock()
    _lazy_save_chunks: List[Tuple[str, str, list['PerfCounter'], datetime, bool, bool]] = []   # (filename, marker, counter_list, timestamp, save_jsonl, save_log)
    _lazy_save_chunks_lock = threading.Lock()

    @classmethod
    def _init(cls):
        if cls._initialized:
            raise RuntimeError("GlobalPerfContext already initialized")
        cls._initialized = True

        if not cls.STATIC_DISABLED and cls.LAZY_SAVE:
            def _lazy_saver_worker():
                # only trigger save when idle 25s+
                idle_seconds_threshold = 25
                while True:
                    time.sleep(1)
                    with cls._lazy_save_chunks_lock:
                        if not cls._lazy_save_chunks:
                            continue
                        # get last recorded timestamp
                        last_timestamp = cls._lazy_save_chunks[-1][3]
                        idle_time_sec = (datetime.now() - last_timestamp).total_seconds()
                        if idle_time_sec < idle_seconds_threshold:
                            continue
                        logger.debug(f"Lazy saver worker triggered after {idle_time_sec:.1f} s idle")
                        chunks_to_save = cls._lazy_save_chunks
                        cls._lazy_save_chunks = []

                    with cls._file_save_lock:
                        jsonl_files: dict[str, IO] = {}
                        log_files: dict[str, IO] = {}
                        try:
                            for filename, marker, counter_list, timestamp, save_jsonl, save_log in chunks_to_save:
                                if save_jsonl:
                                    if filename not in jsonl_files:
                                        logger.debug(f"open {filename}.jsonl: exist={os.path.exists(f'{filename}.jsonl')}")
                                        if not os.path.exists(f"{filename}.jsonl"):
                                            open(f"{filename}.jsonl", "w").close()
                                        f = open(f"{filename}.jsonl", "r+")
                                        f.seek(0, os.SEEK_END)
                                        jsonl_files[filename] = f
                                        logger.debug(f"create/append mode, offset={jsonl_files[filename].tell()}, inode={os.fstat(jsonl_files[filename].fileno()).st_ino}, fd={jsonl_files[filename].fileno()}")
                                    f_jsonl = jsonl_files[filename]
                                    cls._save_jsonl(f_jsonl, counter_list, marker, timestamp)
                                if save_log:
                                    if filename not in log_files:
                                        logger.debug(f"open {filename}.log: exist={os.path.exists(f'{filename}.log')}")
                                        if not os.path.exists(f"{filename}.log"):
                                            open(f"{filename}.log", "w").close()
                                        f = open(f"{filename}.log", "r+")
                                        f.seek(0, os.SEEK_END)
                                        log_files[filename] = f
                                        logger.debug(f"create/append mode, offset={log_files[filename].tell()}, inode={os.fstat(log_files[filename].fileno()).st_ino}, fd={log_files[filename].fileno()}")
                                    f_log = log_files[filename]
                                    cls._save_log(f_log, counter_list, marker, timestamp)
                        finally:
                            for n, f in jsonl_files.items():
                                f.close()
                                logger.debug(f"closed {n}.jsonl")
                            for n, f in log_files.items():
                                f.close()
                                logger.debug(f"closed {n}.log")
                    logger.debug(f"Lazy saver worker finished saving {len(chunks_to_save)} chunks")

            threading.Thread(target=_lazy_saver_worker, daemon=True).start()

    @classmethod
    @contextmanager
    def disable(cls):
        """Disable profiling within this context manager."""
        cls._program_disabled = True
        try:
            yield
        finally:
            cls._program_disabled = False

    @classmethod
    def _eligible(cls) -> bool:
        return not cls.STATIC_DISABLED and cls.global_enabled and not cls._program_disabled

    @classmethod
    def cudagraph_helper(cls, sample_rate: float = 1) -> bool:
        """helper function to decide whether to do cudagraph profiling"""
        if not cls._eligible():
            return False
        do_profile = random.random() < sample_rate
        cls._force_do_profile_once = do_profile
        return do_profile

    @classmethod
    def set_counter(cls, counter: 'PerfCounter') -> None:
        tid = _get_thread_id()
        with cls._recorded_counters_lock:
            cls.threadlocal_active_counter_stack[tid].append(counter)
            counter.depth = len(cls.threadlocal_active_counter_stack[tid]) - 1
            cls.recorded_counters.append(counter)

    @classmethod
    def unset_counter(cls, counter: 'PerfCounter') -> None:
        tid = _get_thread_id()
        with cls._recorded_counters_lock:
            stack = cls.threadlocal_active_counter_stack
            if tid not in stack or not stack[tid] or stack[tid][-1] is not counter:
                logger.error(f"Mismatched PerfCounter unset operation for {counter.name} (called in different thread?)")
                return

            stack[tid].pop()

    @classmethod
    def begin_with_sample_rate(cls, sample_rate: float = 1) -> None:
        with cls._recorded_counters_lock:
            cls.threadlocal_active_counter_stack = DefaultDict(list)
            cls.recorded_counters = []
        not_capturing = not torch.cuda.is_current_stream_capturing()
        sample_hit = random.random() < sample_rate
        if cls._force_do_profile_once:
            sample_hit = True
            cls._force_do_profile_once = False
        cls._effective_enabled = cls._eligible() and not_capturing and sample_hit

    @classmethod
    def _finalize_counters(cls, stacks: dict[int, list['PerfCounter']], counters: list['PerfCounter']) -> None:
        with cls.cuda_lock:
            if any(stack for stack in stacks.values()):
                logger.error("Some PerfCounters are still active during finalize, which will be ignored.")
                logger.error(f"Active counters: {stacks}")
                still_active_counter_set = {counter for stack in stacks.values() for counter in stack}
                counters = [counter for counter in counters if counter not in still_active_counter_set]

            # found first es
            first_event = counters[0].es
            for counter in counters[1:]:
                if first_event.elapsed_time(counter.es) < 0:
                    first_event = counter.es

            for counter in counters:
                counter.ref_t_start(first_event)

            for counter in counters:
                counter.finalize()

    @classmethod
    def finalize(cls) -> Optional[list['PerfCounter']]:
        with cls._recorded_counters_lock:
            stacks = cls.threadlocal_active_counter_stack
            cls.threadlocal_active_counter_stack = DefaultDict(list)
            counter_list = cls.recorded_counters
            cls.recorded_counters = []

        if not counter_list:
            return
        if torch.cuda.is_current_stream_capturing():
            return

        torch.cuda.synchronize()
        try:
            cls._finalize_counters(stacks, counter_list)
        except Exception as e:
            logger.error(f"Error in finalizing counters: {e}")
            raise e
            return

        return counter_list

    @classmethod
    def _get_perf_str(cls, counter: 'PerfCounter') -> str:
        if counter.type == 'GEMM_OP' and all(k in counter.shapes for k in ("m", 'n', 'k')):
            m, n, k = counter.shapes["m"], counter.shapes["n"], counter.shapes["k"]
            tflops = 2 * m * n * k / (counter.t_elapsed_ms * 1e9)
            return f", {tflops:.3f} TFLOPS"
        if counter.type == 'GEMM_OP' and 'flops' in counter.shapes:
            flops = counter.shapes['flops']
            tflops = flops / (counter.t_elapsed_ms * 1e9)
            return f", {tflops:.3f} TFLOPS"
        if counter.type == 'COMM_OP' and 'size' in counter.shapes:
            num_bytes = counter.shapes['size']
            gbps = num_bytes / (counter.t_elapsed_ms * 1e6)
            return f", {gbps:.3f} GB/s (pseudo)"
        return ''

    @classmethod
    def _save_jsonl(cls, f: IO, counter_list: list['PerfCounter'], marker: str, timestamp: datetime) -> None:
        time_str = f" @ {timestamp.isoformat(timespec='milliseconds')}"
        f.write(f"# PerfCounterContext.finalize {marker}{time_str}\n")
        for counter in counter_list:
            keys = ("name", "type", "shapes", "depth", "t_start_ms", "t_elapsed_ms", "t_start_cpu_timestamp")
            json_str = json.dumps({key: getattr(counter, key) for key in keys} | {"marker": marker})
            _ = json.loads(json_str)  # validate
            f.write(json_str + '\n')

    @classmethod
    def _save_log(cls, f: IO, counter_list: list['PerfCounter'], marker: str, timestamp: datetime) -> None:
        time_str = f" @ {timestamp.isoformat(timespec='milliseconds')}"
        f.write(f"PerfCounterContext.finalize [{marker}]{time_str}:\n")
        for i in range(len(counter_list)):
            counter = counter_list[i]
            shape_str = ", ".join(f"{k}={v}" for k, v in counter.shapes.items()) if counter.shapes is not None else ""
            tabs = '    ' * counter.depth
            last_t = 0.0
            time_delta = ''
            if i > 0:
                if counter_list[i-1].depth < counter.depth:
                    # is child
                    last_t = counter_list[i-1].t_start_ms
                    time_delta = f"(+{counter.t_start_ms - last_t:.3f} ms)"
                else:
                    # is next
                    last_t = counter_list[i-1].t_start_ms + counter_list[i-1].t_elapsed_ms
                    last_t_same_level = last_t
                    if counter_list[i-1].depth != counter.depth:
                        # find last same level
                        for j in range(i-2, -1, -1):
                            if counter_list[j].depth == counter.depth:
                                last_t_same_level = counter_list[j].t_start_ms + counter_list[j].t_elapsed_ms
                                break
                        time_delta = f"(+{counter.t_start_ms - last_t:.3f} ms / ~{counter.t_start_ms - last_t_same_level:.3f})"
                    else:
                        time_delta = f"(+{counter.t_start_ms - last_t:.3f} ms)"
            line = f"{tabs}{counter.name}.{counter.type}({shape_str}): {time_delta} {counter.t_elapsed_ms:.3f} ms"
            line += cls._get_perf_str(counter)
            f.write(line + "\n")
        f.write("\n")

    @classmethod
    def finalize_async(cls, marker: str = "perf", save_jsonl: bool = False, save_log: bool = False, filename: Optional[str] = None) -> None:
        main_t_start = time.time()
        if filename is None:
            filename = marker

        with cls._recorded_counters_lock:
            stacks = cls.threadlocal_active_counter_stack
            cls.threadlocal_active_counter_stack = DefaultDict(list)
            counter_list = cls.recorded_counters
            cls.recorded_counters = []

        if not counter_list:
            return
        if not save_jsonl and not save_log:
            return
        if torch.cuda.is_current_stream_capturing():
            return

        torch.cuda.synchronize()

        def _worker():
            thread_t_start = time.time()
            if torch.cuda.is_current_stream_capturing():
                return
            try:
                cls._finalize_counters(stacks, counter_list)
            except Exception as e:
                logger.error(f"Error in finalizing counters: {e}")
                # raise e
                return

            thread_t_finalize_end = time.time()

            if not cls.LAZY_SAVE:
                with cls._file_save_lock:
                    if save_jsonl:
                        with open(f"{filename}.jsonl", "a") as f_jsonl:
                            cls._save_jsonl(f_jsonl, counter_list, marker, datetime.now())
                    if save_log:
                        with open(f"{filename}.log", "a") as f_log:
                            cls._save_log(f_log, counter_list, marker, datetime.now())
            else:
                with cls._lazy_save_chunks_lock:
                    cls._lazy_save_chunks.append((filename, marker, counter_list, datetime.now(), save_jsonl, save_log))

            thread_t_end = time.time()
            # logger.debug(f"PerfCounterContext.finalize_async worker took {thread_t_end - thread_t_start:.3f} s ({thread_t_finalize_end - thread_t_start:.3f} s in _finalize_counters)")

        threading.Thread(target=_worker).start()
        main_t_end = time.time()
        # logger.debug(f"PerfCounterContext.finalize_async main took {main_t_end - main_t_start:.3f} s")


_PerfType = Literal[
    'GEMM_OP', 'ATTN_OP', 'COMM_OP', 'ACT_OP', 'QUANT_OP', 'OTHER_OP',
    'LAYER', 'BLOCK', 'MODEL',
    '_OTHER'
]

class PerfCounter:
    __slot__ = ("name", "type", "shapes", "depth",
                 "_is_active", "_is_stopped",
                 "finalized", "t_start_ms", "t_start_cpu_timestamp", "t_elapsed_ms",
                 "es", "ee")
    TYPES = get_args(_PerfType)
    def __init__(self, name: Optional[str] = None, type: _PerfType = '_OTHER'):
        self.name = name
        self.type = type
        self.shapes: dict[str, Any] = {}
        self.depth = 0
        self._is_active = False
        self._is_stopped = False

        self.finalized = False
        self.t_start_ms = None
        self.t_start_cpu_timestamp = None
        self.t_elapsed_ms = None

    def __enter__(self) -> 'PerfCounter':
        """context manager usage:
        ```
        with PerfCounter(type="GEMM_OP", name="my_gemm") as p:
            p.record_shape(m=..., n=..., k=...)
            my_function(...)
            ...
        ```
        """
        if not GlobalPerfContext._effective_enabled:
            return self
        if torch.cuda.is_current_stream_capturing():
            return self

        if self._is_active or self._is_stopped:
            raise RuntimeError("PerfCounter already in use, cannot restart")
        self._is_active = True
        self.t_start_cpu_timestamp = time.time()
        self.es = torch.cuda.Event(enable_timing=True)
        self.ee = torch.cuda.Event(enable_timing=True)
        GlobalPerfContext.set_counter(self)
        self.es.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not GlobalPerfContext._effective_enabled:
            return
        if torch.cuda.is_current_stream_capturing():
            return

        self.ee.record()
        # lazy synchronize
        GlobalPerfContext.unset_counter(self)
        self._is_active = False
        self._is_stopped = True

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)

    def is_active(self) -> bool:
        return self._is_active

    def __call__(self, func):
        """decorator usage:
        ```
        @PerfCounter(type="GEMM_OP")
        def my_function(...):
            my_function.record_shape(m=..., n=..., k=...)
        ```
        """
        # when used as a decorator, this object won't directly functional,
        # instead, it acts as a factory that creates a new PerfCounter object for each function call
        if hasattr(func, "_is_perf_counter_factory_wrapped") and func._is_perf_counter_factory_wrapped:
            return func

        if not self.name and hasattr(func, "__name__"):
            self.name = func.__name__
        @functools.wraps(func)
        def wrapped_func(*args, **kwds):
            counter_obj = PerfCounter(self.name, self.type)
            wrapped_func.is_perf_counter_active = counter_obj.is_active
            wrapped_func.record_shape = counter_obj.record_shape
            wrapped_func._current_perf_counter = counter_obj
            with counter_obj:
                return func(*args, **kwds)
        wrapped_func._is_perf_counter_factory_wrapped = True
        wrapped_func._original_func = func
        return wrapped_func

    def wrap_func(self, func):
        """decorator usage:
        ```
        my_function = PerfCounter(type="GEMM_OP").wrap_func(my_function)    # must wrap before every call
        my_function(...)
        ```
        """
        # in .wrap_func() case, unwrap if already wrapped to use the new PerfCounter
        if hasattr(func, "_is_perf_counter_factory_wrapped") and func._is_perf_counter_factory_wrapped:
            logger.warning("Warning: wrong usage: .wrap_func() called on a function already wrapped by @PerfCounter decorator, unwrapping to use the new PerfCounter")
            func = func._original_func
        if hasattr(func, "_is_perf_counter_wrapped") and func._is_perf_counter_wrapped:
            func = func._original_func
        if not self.name and hasattr(func, "__name__"):
            self.name = func.__name__
        @functools.wraps(func)
        def wrapped_func(*args, **kwds):
            with self:
                return func(*args, **kwds)
        wrapped_func.is_perf_counter_active = self.is_active
        wrapped_func.record_shape = self.record_shape
        wrapped_func._is_perf_counter_wrapped = True
        wrapped_func._original_func = func
        wrapped_func._current_perf_counter = self
        return wrapped_func

    def finalize(self) -> None:
        if self.finalized:
            logger.error(f"PerfCounter{self.name} already finalized")
            return

        self.finalized = True
        self.t_elapsed_ms = self.es.elapsed_time(self.ee)
        self.es = None
        self.ee = None

    def ref_t_start(self, first_es: torch.cuda.Event):
        self.t_start_ms = first_es.elapsed_time(self.es)

    def record_shape(self, **kwds: Any) -> None:
        if self.shapes:
            raise RuntimeError("PerfCounter shapes already recorded")
        self.shapes = kwds


GlobalPerfContext._init()


if GlobalPerfContext.STATIC_DISABLED:
    # disable all profiling
    def _dummy_wrap(func):
        try:
            func.is_perf_counter_active = lambda : False
            func.record_shape = lambda **kwds: None
            return func
        except Exception:
            # some built-in functions may not allow setting attributes
            def no_op(*args, **kwds):
                return func(*args, **kwds)
            no_op.is_perf_counter_active = lambda : False
            no_op.record_shape = lambda **kwds: None
            return no_op
    class PerfCounter_noop:
        def __init__(self, *args, **kwds): pass
        def __enter__(self) -> 'PerfCounter_noop': return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def start(self): pass
        def stop(self): pass
        def is_active(self) -> bool: return False
        def __call__(self, func): return _dummy_wrap(func)
        def wrap_func(self, func): return _dummy_wrap(func)
        def record_shape(self, **kwds: Any) -> None: pass
    PerfCounter = PerfCounter_noop

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
