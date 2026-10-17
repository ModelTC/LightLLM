"""Central logging configuration; see docs/CN/source/tutorial/logging.rst."""

import hashlib
import json
import logging
import logging.handlers
import math
import numbers
import multiprocessing
import os
import socket
import sys
import time
import threading
from collections import OrderedDict
from datetime import datetime
from contextvars import ContextVar
import uuid

_FORMAT = "%(asctime)s %(levelname)s [%(compact_context)s] %(source)s%(message)s"
_DATE_FORMAT = "%H:%M:%S"
_LOG_NODE_ROLE = os.environ.get("LIGHTLLM_LOG_NODE_ROLE", "")
_PD_NODE_ROLE_MARKERS = {"prefill": "P", "decode": "D"}
_root_logger = logging.getLogger("lightllm")
_context = {}
_context_pid = os.getpid()
_external_loggers = set()
_module_levels = {}
_log_time_mark_dict = {}
_hostname = socket.gethostname()
_http_request_context = ContextVar("lightllm_http_request", default=None)
_STANDARD_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def set_log_node_role(run_mode: str):
    """Propagate the P/D role to spawned and exec'ed workers."""
    global _LOG_NODE_ROLE
    _LOG_NODE_ROLE = _PD_NODE_ROLE_MARKERS.get(run_mode, "")
    if _LOG_NODE_ROLE:
        os.environ["LIGHTLLM_LOG_NODE_ROLE"] = _LOG_NODE_ROLE
    else:
        os.environ.pop("LIGHTLLM_LOG_NODE_ROLE", None)


def set_log_context(**fields):
    """Set process-local component/rank metadata after process or rank initialization."""
    global _context_pid
    if _context_pid != os.getpid():
        _context.clear()
        _context_pid = os.getpid()
    _context.update(fields)


def _add_context(record):
    context = _context if _context_pid == os.getpid() else {}
    defaults = {
        "node_role": f"[{_LOG_NODE_ROLE}] " if _LOG_NODE_ROLE else "",
        "hostname": _hostname,
        "component": multiprocessing.current_process().name,
        "node_rank": os.getenv("LIGHTLLM_LOG_NODE_RANK", "-"),
        "tp_rank": "-",
        "dp_rank": "-",
        "global_rank": "-",
        "role": _LOG_NODE_ROLE,
        "tp_size": int(os.getenv("LIGHTLLM_LOG_TP_SIZE", "1")),
        "dp_size": int(os.getenv("LIGHTLLM_LOG_DP_SIZE", "1")),
        "nnodes": int(os.getenv("LIGHTLLM_LOG_NNODES", "1")),
        "http_request_id": get_http_request_id(),
        **context,
    }
    for key, value in defaults.items():
        if not hasattr(record, key):
            setattr(record, key, value)


_COMPONENT_ALIASES = {
    "MainProcess": "main",
    "model_infer": "infer",
    "http": "http",
    "start_router_process": "router",
    "start_detokenization_process": "detoken",
    "start_metric_manager": "metrics",
    "start_health_check_process": "health",
    "start_cache_manager": "cache",
    "kv_transfer": "kv-transfer",
    "kv_move": "kv-move",
    "kv_status": "kv-status",
}


def _compact_context(record):
    component = _COMPONENT_ALIASES.get(record.component, record.component)
    if os.getenv("LIGHTLLM_LOG_RUN_MODE") == "pd_master" and component == "http":
        component = "master"
    if component.startswith("start_"):
        component = component[6:]
        for suffix in ("_process", "_manager"):
            if component.endswith(suffix):
                component = component[: -len(suffix)]
    if component == "_init_env" or component.startswith(("Process-", "SpawnProcess-", "ForkProcess-")):
        component = "worker"
    label = f"{record.role}/{component}" if record.role else component
    fields = [label]
    for key, size in (("node", record.nnodes), ("tp", record.tp_size), ("dp", record.dp_size)):
        rank = getattr(record, f"{key}_rank")
        if size > 1 and rank not in (None, "-"):
            fields.append(f"{key}={rank}")
    return " ".join(fields)


class NewLineFormatter(logging.Formatter):
    """Prefix every physical line, including traceback lines, with process context."""

    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

    def format(self, record):
        _add_context(record)
        record = logging.makeLogRecord(record.__dict__.copy())
        if record.name == "py.warnings":
            # Python warning source snippets and trailing blank lines add no value
            # to the console. JSON retains the original warning and source.
            record.msg, record.args = record.getMessage().splitlines()[0], ()
        record.compact_context = _compact_context(record)
        record.source = (
            f"{record.filename}:{record.lineno} "
            if record.levelno <= logging.DEBUG or record.levelno >= logging.ERROR or record.exc_info
            else ""
        )
        # Format a separate empty record to obtain the prefix without splitting on
        # user-controlled message content. Do not mutate records shared by handlers.
        prefix_record = logging.makeLogRecord(record.__dict__.copy())
        prefix_record.msg, prefix_record.args = "", ()
        prefix_record.exc_info = prefix_record.exc_text = prefix_record.stack_info = None
        prefix = super().format(prefix_record)
        return super().format(record).replace("\n", "\r\n" + prefix)


class JsonFormatter(logging.Formatter):
    """One JSON object per event; newlines and tracebacks remain inside the object."""

    def format(self, record):
        _add_context(record)
        result = {
            "timestamp": datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
            "pid": record.process,
        }
        result.update({key: value for key, value in record.__dict__.items() if key not in _STANDARD_RECORD_FIELDS})
        if record.exc_info:
            result["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            result["stack"] = self.formatStack(record.stack_info)
        return json.dumps(result, ensure_ascii=False, default=str)


class ProcessFileHandler(logging.handlers.RotatingFileHandler):
    """Use one rotating file per PID, including when a handler is inherited by fork."""

    def __init__(self, directory, max_bytes, backup_count):
        self.directory = directory
        self.pid = os.getpid()
        super().__init__(self._filename(), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8", delay=True)

    def _filename(self):
        return os.path.abspath(os.path.join(self.directory, f"process.{self.pid}.log"))

    def emit(self, record):
        if self.pid != os.getpid():
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            self.pid = os.getpid()
            self.baseFilename = self._filename()
        super().emit(record)


def _level(value):
    value = str(value).upper()
    if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"Invalid LightLLM log level: {value}")
    return getattr(logging, value)


def _positive_number(name, default, convert=float):
    value = convert(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _setup_logger():
    """Configure owned handlers once per call, leaving other libraries untouched."""
    global _module_levels, _request_sample_rate, _stats_interval
    level = _level(os.getenv("LIGHTLLM_LOG_LEVEL", "INFO"))
    modules = json.loads(os.getenv("LIGHTLLM_LOG_MODULE_LEVELS", "{}"))
    if not isinstance(modules, dict) or any(not name.startswith("lightllm.") for name in modules):
        raise ValueError("LIGHTLLM_LOG_MODULE_LEVELS must map lightllm.* logger names to levels")
    modules = {name: _level(value) for name, value in modules.items()}
    log_format = os.getenv("LIGHTLLM_LOG_FORMAT", "text").lower()
    if log_format not in {"text", "json"}:
        raise ValueError("LIGHTLLM_LOG_FORMAT must be text or json")
    request_rate = float(os.getenv("LIGHTLLM_LOG_REQUEST_SAMPLE_RATE", "0.01"))
    if not math.isfinite(request_rate) or not 0 <= request_rate <= 1:
        raise ValueError("LIGHTLLM_LOG_REQUEST_SAMPLE_RATE must be between 0 and 1")
    stats_interval = _positive_number("LIGHTLLM_LOG_STATS_INTERVAL", 10)
    formatter = JsonFormatter() if log_format == "json" else NewLineFormatter(_FORMAT, _DATE_FORMAT)
    handlers = [logging.StreamHandler(sys.stdout)]
    directory = os.getenv("LIGHTLLM_LOG_DIR")
    if directory:
        os.makedirs(directory, exist_ok=True)
        handlers.append(
            ProcessFileHandler(
                directory,
                _positive_number("LIGHTLLM_LOG_MAX_BYTES", 100 * 1024 * 1024, int),
                _positive_number("LIGHTLLM_LOG_BACKUP_COUNT", 3, int),
            )
        )
    for handler in handlers:
        handler.setFormatter(formatter)
        # Child loggers decide eligibility; an INFO handler would drop module DEBUG.
        handler.setLevel(logging.NOTSET)
        handler._lightllm_owned = True
    old_handlers = [handler for handler in _root_logger.handlers if getattr(handler, "_lightllm_owned", False)]
    for handler in old_handlers:
        _root_logger.removeHandler(handler)
    _root_logger.setLevel(level)
    _root_logger.propagate = False
    for handler in handlers:
        _root_logger.addHandler(handler)
    for name in _module_levels:
        logging.getLogger(name).setLevel(logging.NOTSET)
    # Request/access categories have switches independent of the service level.
    logging.getLogger("lightllm.requests").setLevel(min(level, logging.INFO) if request_rate else logging.NOTSET)
    logging.getLogger("lightllm.access").setLevel(logging.INFO if http_access_enabled() else logging.NOTSET)
    for name, module_level in modules.items():
        logging.getLogger(name).setLevel(module_level)
    _module_levels = modules
    for name in _external_loggers:
        logger = logging.getLogger(name)
        for handler in old_handlers:
            logger.removeHandler(handler)
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
    for handler in old_handlers:
        handler.close()
    _request_sample_rate = request_rate
    _stats_interval = stats_interval


def init_logger(name: str):
    """Retrieve a logger without adding handlers to LightLLM module loggers."""
    logger = logging.getLogger(name)
    if name != "lightllm" and not name.startswith("lightllm."):
        # Scripts also use this API with __name__ == '__main__'.
        _external_loggers.add(name)
        logger.setLevel(_root_logger.level)
        for handler in _root_logger.handlers:
            if getattr(handler, "_lightllm_owned", False):
                logger.addHandler(handler)
        logger.propagate = False
    return logger


def configure_logging_from_args(args):
    """Explicit CLI values override environment defaults and propagate to workers."""
    for attr, env in (
        ("log_level", "LIGHTLLM_LOG_LEVEL"),
        ("log_format", "LIGHTLLM_LOG_FORMAT"),
        ("log_module_levels", "LIGHTLLM_LOG_MODULE_LEVELS"),
        ("log_request_sample_rate", "LIGHTLLM_LOG_REQUEST_SAMPLE_RATE"),
        ("log_stats_interval", "LIGHTLLM_LOG_STATS_INTERVAL"),
        ("log_http_level", "LIGHTLLM_LOG_HTTP_LEVEL"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env] = str(value)
    access = getattr(args, "enable_http_access_log", None)
    if access is not None:
        os.environ["LIGHTLLM_LOG_HTTP_ACCESS"] = "1" if access else "0"
    _level(os.getenv("LIGHTLLM_LOG_HTTP_LEVEL", "WARNING"))
    os.environ["LIGHTLLM_LOG_NODE_RANK"] = str(args.node_rank)
    os.environ["LIGHTLLM_LOG_TP_SIZE"] = str(max(1, getattr(args, "tp", 1) // getattr(args, "dp", 1)))
    os.environ["LIGHTLLM_LOG_DP_SIZE"] = str(getattr(args, "dp", 1))
    os.environ["LIGHTLLM_LOG_NNODES"] = str(getattr(args, "nnodes", 1))
    os.environ["LIGHTLLM_LOG_RUN_MODE"] = args.run_mode
    set_log_node_role(args.run_mode)
    _setup_logger()
    set_log_context(component="launcher", node_rank=args.node_rank)


def should_log_request(request_id):
    """Deterministic sampling by ID; does not consume application RNG state."""
    if _request_sample_rate == 0:
        return False
    if _request_sample_rate == 1:
        return True
    digest = hashlib.blake2b(str(request_id).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2 ** 64 < _request_sample_rate


def http_access_enabled():
    return os.getenv("LIGHTLLM_LOG_HTTP_ACCESS", "0").lower() in {"1", "true", "yes", "on"}


def new_http_request_id(headers):
    """Reuse a bounded printable client ID, otherwise generate a transport ID."""
    value = next((v.decode("latin-1") for k, v in headers if k.lower() == b"x-request-id"), "")
    if not value or len(value) > 128 or any(ord(char) < 33 or ord(char) > 126 for char in value):
        value = uuid.uuid4().hex
    return value


def set_http_request_id(request_id):
    return _http_request_context.set((os.getpid(), request_id))


def reset_http_request_id(token):
    _http_request_context.reset(token)


def get_http_request_id():
    value = _http_request_context.get()
    return value[1] if value is not None and value[0] == os.getpid() else "-"


def summarize_exception(error, limit=240):
    """Keep a useful compiler/worker failure reason without dumping its transcript."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    useful = [
        line
        for line in lines
        if any(marker in line.lower() for marker in ("no such file", "not found", "error:", "runtimeerror:"))
    ]
    reason = (useful or lines or [type(error).__name__])[-1]
    return reason if len(reason) <= limit else reason[: limit - 3] + "..."


def progress_enabled():
    """Only display interactive bars on a terminal with text INFO logging."""
    return (
        sys.stderr.isatty()
        and os.getenv("LIGHTLLM_LOG_FORMAT", "text").lower() == "text"
        and _root_logger.isEnabledFor(logging.INFO)
    )


def log_http_access(method, path, status, duration_ms, http_request_id=None):
    excluded = os.getenv("LIGHTLLM_LOG_HTTP_EXCLUDE_PATHS", "/health,/healthz,/readiness,/liveness,/metrics").split(",")
    if not http_access_enabled() or any(
        prefix and (path == prefix or path.startswith(prefix.rstrip("/") + "/")) for prefix in excluded
    ):
        return
    fields = {
        "event": "http.access",
        "http_request_id": http_request_id or get_http_request_id(),
        "method": method,
        "path": path,
        "status": status,
        "duration_ms": round(duration_ms, 3),
    }
    logging.getLogger("lightllm.access").info(
        "event=http.access http_request_id=%s method=%s path=%r status=%s duration_ms=%.3f",
        fields["http_request_id"],
        method,
        path,
        status,
        duration_ms,
        extra=fields,
        stacklevel=2,
    )


def log_request_summary(request_id, **fields):
    status = fields.get("status", "completed")
    failed = status in {"error", "timeout", "failed", "busy"}
    cancelled = status in {"cancelled", "abort"}
    logger = logging.getLogger("lightllm.requests")
    stage = fields.get("timing_scope") in {"prefill", "decode"}
    if not failed and not cancelled and not should_log_request(request_id):
        return
    event = fields.pop(
        "event", "request.failed" if failed else "request.cancelled" if cancelled else "request.completed"
    )
    client_id = fields.pop("x_request_id", None) or get_http_request_id()
    if client_id != "-":
        fields.setdefault("http_request_id", client_id)
    mtp_enabled = fields.pop("mtp_enabled", False)
    fields = {
        key: _debug_value(value)
        for key, value in fields.items()
        if not (("cache" in key and not value) or (key.startswith("mtp_") and not mtp_enabled))
    }
    method = logger.error if failed else logger.debug if stage or cancelled else logger.info
    method(
        "event=%s request_id=%s %s",
        event,
        request_id,
        " ".join(f"{key}={value!r}" for key, value in fields.items()),
        extra={"event": event, "request_id": request_id, **fields},
        stacklevel=2,
    )


def request_finish_fields(reasons):
    """Preserve failures across best-of children instead of trusting the last token."""
    reasons = sorted(set(reasons))
    status = (
        "error"
        if "error" in reasons
        else "abort"
        if "abort" in reasons
        else "continued"
        if "pd_capacity" in reasons
        else "completed"
    )
    return {"status": status, "finish_reason": reasons[0] if len(reasons) == 1 else reasons}


def get_http_logging_args(application_access=False):
    """Hypercorn access and error logging are independent of service verbosity."""
    level = os.getenv("LIGHTLLM_LOG_HTTP_LEVEL", "warning")
    _level(level)
    args = ["--log-level", level.lower(), "--error-logfile", "-"]
    if http_access_enabled() and not application_access:
        args.extend(["--access-logfile", "-"])
    return args


def get_log_stats_interval():
    return _stats_interval


def log_time_ready(mark_name, time_count: float):
    """Return true once each interval, using a monotonic clock and process-local key."""
    key = (os.getpid(), mark_name)
    now = time.monotonic()
    previous = _log_time_mark_dict.setdefault(key, now)
    if now - previous >= time_count:
        _log_time_mark_dict[key] = now
        return True
    return False


def _debug_value(value, depth=0):
    """Bound diagnostic values without expanding tensors, descriptors or objects."""
    if isinstance(value, str):
        return value if len(value) <= 256 else value[:253] + "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, (list, tuple)):
        return (
            {"count": len(value)}
            if depth
            else {"count": len(value), "sample": [_debug_value(v, depth=1) for v in value[:8]]}
        )
    if isinstance(value, (bytes, bytearray)):
        return {"type": type(value).__name__, "bytes": len(value)}
    return {"type": type(value).__name__}


def log_debug_event(logger, event, *, leader_only=False, stacklevel=2, **fields):
    """A bounded phase/decision event, never a dump of request content."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    context = _context if _context_pid == os.getpid() else {}
    if leader_only and context.get("tp_rank", 0) != 0:
        return
    fields = {key: _debug_value(value) for key, value in list(fields.items())[:16]}
    logger.debug(
        "event=%s %s",
        event,
        " ".join(f"{key}={value!r}" for key, value in fields.items()),
        extra={"event": event, **fields},
        stacklevel=stacklevel,
    )


class DebugStatsLogger:
    """Accumulate hot-loop counts; emit at most once per interval, with idle flush.

    Call ``flush()`` when the owning loop becomes idle so the final partial window
    is not lost. State has fixed cardinality and is protected for worker threads.
    """

    def __init__(self, logger, event, interval=None):
        self.logger, self.event = logger, event
        self.interval = interval if interval is not None else get_log_stats_interval()
        self.pid = os.getpid()
        self.lock = threading.Lock()
        self.count = 0
        self.totals = {}
        self.request_ids = []
        self.request_ids_truncated = False
        self.started = time.monotonic()
        self.last_emit = None

    def _reset_after_fork(self):
        if self.pid != os.getpid():
            self.__init__(self.logger, self.event, self.interval)

    def add(self, request_id=None, **totals):
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        self._reset_after_fork()
        with self.lock:
            self.count += 1
            if request_id is not None and request_id not in self.request_ids:
                if len(self.request_ids) < 8:
                    self.request_ids.append(request_id)
                else:
                    self.request_ids_truncated = True
            for key, value in totals.items():
                self.totals[key] = self.totals.get(key, 0) + value
            self._flush_locked()

    def flush(self):
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        self._reset_after_fork()
        with self.lock:
            self._flush_locked()

    def _flush_locked(self):
        now = time.monotonic()
        if not self.count or (self.last_emit is not None and now - self.last_emit < self.interval):
            return
        fields = {}
        if self.request_ids:
            fields = {"request_id_sample": self.request_ids, "request_ids_truncated": self.request_ids_truncated}
        log_debug_event(
            self.logger,
            self.event,
            stacklevel=4,
            count=self.count,
            window_s=round(now - self.started, 3),
            **fields,
            **self.totals,
        )
        self.count, self.totals = 0, {}
        self.request_ids = []
        self.request_ids_truncated = False
        self.started = self.last_emit = now


_exception_windows = OrderedDict()
_exception_window_pid = os.getpid()
_exception_window_lock = threading.Lock()


def log_repeated_exception(logger, event, **fields):
    """Keep the first traceback; count identical recurring failures by callsite.

    Distinct reasons keep their own traceback. The cache is bounded per process;
    summaries never replace the first error, and are emitted at most every 10 s.
    """
    if not logger.isEnabledFor(logging.ERROR):
        return
    global _exception_window_pid, _exception_window_lock
    if _exception_window_pid != os.getpid():
        _exception_windows.clear()
        _exception_window_lock = threading.Lock()
        _exception_window_pid = os.getpid()
    exc_type, error, traceback = sys.exc_info()
    caller = sys._getframe(1)
    reason = summarize_exception(error) if error is not None else "unknown failure"
    key = (logger.name, event, caller.f_code.co_filename, caller.f_lineno, exc_type, reason)
    now = time.monotonic()
    with _exception_window_lock:
        state = _exception_windows.get(key)
        first = state is None
        if first:
            if len(_exception_windows) >= 128:
                _exception_windows.popitem(last=False)
            state = {"last": now, "repeated": 0}
            _exception_windows[key] = state
        else:
            _exception_windows.move_to_end(key)
            state["repeated"] += 1
            if now - state["last"] < 10:
                return
        repeated = state["repeated"]
        state["last"], state["repeated"] = now, 0
    fields = {key: _debug_value(value) for key, value in list(fields.items())[:12]}
    logger.error(
        "event=%s reason=%r repeated=%s %s",
        event,
        reason,
        repeated,
        " ".join(f"{key}={value!r}" for key, value in fields.items()),
        extra={"event": event, "reason": reason, "repeated": repeated, **fields},
        exc_info=(exc_type, error, traceback) if first and error is not None else None,
        stacklevel=2,
    )


def log_info_on_leader(logger, message, *args, **kwargs):
    """Shared TP configuration is INFO on the group's leader, DEBUG on peers.

    This helper must never be used to suppress worker-specific warnings/errors.
    Unknown ranks (launcher and non-inference processes) retain INFO output.
    """
    context = _context if _context_pid == os.getpid() else {}
    method = logger.info if context.get("tp_rank", 0) == 0 else logger.debug
    kwargs.setdefault("stacklevel", 2)
    method(message, *args, **kwargs)


class PeriodicStatsLogger:
    """Rate-limit all snapshots and confirm idle for a full reporting interval."""

    def __init__(self, logger):
        self.logger = logger
        self.last_time = time.monotonic()
        self.active = None
        self.idle_since = None

    def should_log(self, active):
        now = time.monotonic()
        interval = get_log_stats_interval()
        if active:
            self.idle_since = None
        elif self.idle_since is None:
            self.idle_since = now
        if now - self.last_time < interval:
            return False
        if not active and (self.active is False or now - self.idle_since < interval):
            return False
        self.active = active
        self.last_time = now
        return True

    def log(self, **snapshot):
        event = "engine.stats" if self.active else "engine.idle"
        self.logger.info(
            "event=%s %s",
            event,
            " ".join(
                f"{k}={v}"
                for k, v in snapshot.items()
                if k not in {"prefill_tokens", "output_tokens", "completed_reqs", "window_s", "scope"}
            ),
            extra={"event": event, **snapshot},
            stacklevel=2,
        )


def log_startup_config(logger, args):
    """Only allowlisted operational settings are included, even at DEBUG."""
    fields = (
        "run_mode",
        "model_name",
        "tp",
        "dp",
        "nnodes",
        "node_rank",
        "max_req_total_len",
        "max_total_token_num",
        "batch_max_tokens",
        "mem_fraction",
        "pd_trans_mode",
        "data_type",
        "use_tgi_api",
        "tool_call_parser",
        "reasoning_parser",
        "disable_vision",
        "disable_audio",
        "enable_fused_shared_experts",
    )
    config = {key: getattr(args, key, None) for key in fields}
    logger.info(
        "event=service.config %s",
        " ".join(f"{key}={value}" for key, value in config.items()),
        extra={"event": "service.config", "config": config},
        stacklevel=2,
    )


_setup_logger()
logging.captureWarnings(True)
init_logger("py.warnings")
