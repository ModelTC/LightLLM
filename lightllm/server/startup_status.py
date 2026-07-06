import os
import shutil
import subprocess

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if value < 1024 or unit == "PiB":
            return f"{value:.1f}{unit}"
        value /= 1024


def _read_meminfo() -> dict:
    meminfo = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            key, value = line.split(":", 1)
            parts = value.split()
            if len(parts) >= 2 and parts[1] == "kB":
                meminfo[key] = int(parts[0]) * 1024
    return meminfo


def _disk_line(path: str) -> str:
    usage = shutil.disk_usage(path)
    return (
        f"{path}: total={_format_bytes(usage.total)}, "
        f"used={_format_bytes(usage.used)}, free={_format_bytes(usage.free)}"
    )


def _run_status_cmd(cmd: list[str], timeout: int = 3) -> str:
    if shutil.which(cmd[0]) is None:
        return f"{cmd[0]} not found"
    try:
        ret = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        output = (ret.stdout or ret.stderr).strip()
        return output or f"{cmd[0]} returned no output"
    except Exception as e:
        return f"{cmd[0]} failed: {e}"


def _largest_sysv_shm_segments(limit: int = 10) -> str:
    output = _run_status_cmd(["ipcs", "-m", "-b"])
    if "not found" in output or "failed:" in output:
        return output

    rows = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[4].isdigit():
            rows.append((int(parts[4]), line))
    if not rows:
        return output
    rows.sort()
    return "\n".join(line for _, line in rows[-limit:])


def log_machine_status(args) -> None:
    try:
        mem = _read_meminfo()
        paths = ["/", "/dev/shm"]
        for path in [getattr(args, "model_dir", None), "/data"]:
            if path and os.path.exists(path):
                paths.append(path)
        disk = "\n".join(_disk_line(path) for path in dict.fromkeys(paths))

        gpu = _run_status_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )

        logger.info(
            "machine status at startup "
            f"(RAM capacity uses MemTotal; MemAvailable is a snapshot):\n"
            f"host={os.uname().nodename}, pid={os.getpid()}, run_mode={getattr(args, 'run_mode', None)}, "
            f"model_dir={getattr(args, 'model_dir', None)}\n"
            f"ram: total={_format_bytes(mem.get('MemTotal', 0))}, "
            f"available_snapshot={_format_bytes(mem.get('MemAvailable', 0))}, "
            f"free={_format_bytes(mem.get('MemFree', 0))}, "
            f"shared={_format_bytes(mem.get('Shmem', 0))}, "
            f"cached={_format_bytes(mem.get('Cached', 0) + mem.get('SReclaimable', 0))}\n"
            f"disk:\n{disk}\n"
            f"gpu:\n{gpu}\n"
            f"largest sysv shm segments:\n{_largest_sysv_shm_segments()}"
        )
    except Exception:
        logger.exception("failed to log machine status at startup")
