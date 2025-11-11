import os
import torch
import pathlib
import subprocess
import socket
import netifaces
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    from cuda import cuda, cudart

    import nvshmem
    import nvshmem.core
    from nvshmem.core.utils import _get_device
    import triton_dist
    from triton_dist.utils import init_nvshmem_by_torch_process_group

    is_triton_dist = True
except ImportError:
    is_triton_dist = False


def init_triton_dist(world_size: int):
    if is_triton_dist:
        setup_env()
        _TP_GROUP_GLOO = torch.distributed.new_group(ranks=list[int](range(world_size)), backend="gloo")
        init_nvshmem_by_torch_process_group(_TP_GROUP_GLOO)
    else:
        raise ValueError("Triton distributed is not installed")


def setup_env():
    # This function is used to setup the environment for the Triton distributed inference
    script_dir = pathlib.Path.cwd().resolve()
    nvshmem_home = find_nvshmem_home()
    assert nvshmem_home, "NVSHMEM_HOME not found"
    ompi_build = script_dir / "shmem/rocshmem_bind/ompi_build/install/ompi"
    ifname = get_ifname()

    os.environ.update(
        {
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
            "TORCH_CPP_LOG_LEVEL": "1",
            "NCCL_DEBUG": "ERROR",
            "NVSHMEM_HOME": nvshmem_home,
            "LD_LIBRARY_PATH": f"{nvshmem_home}/lib:{ompi_build}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
            "NVSHMEM_BOOTSTRAP": "UID",
            "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": ifname,
            "TRITON_CACHE_DIR": str(script_dir / "triton_cache"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", "") + f":{script_dir}/python",
            "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
            "NVSHMEM_SYMMETRIC_SIZE": os.environ.get("NVSHMEM_SYMMETRIC_SIZE", "1000000000"),
            "NVSHMEM_DIR": nvshmem_home,
            "NVSHMEM_IFNAME": ifname,
        }
    )

    pathlib.Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    # Unset unnecessary vars
    os.environ.pop("NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY", None)

    logger.info("NVSHMEM environment configured successfully.")
    logger.info(f"NVSHMEM_HOME = {nvshmem_home}")
    logger.info(f"NVSHMEM_IFNAME = {ifname}")
    logger.info(f"LD_LIBRARY_PATH includes {nvshmem_home}/lib")


def run_command(cmd):
    """Run shell command and return stdout, or empty string if fails."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError:
        return ""


def find_nvshmem_home():
    """Find NVSHMEM_HOME via environment, Python import, or ldconfig."""
    nvshmem_home = os.environ.get("NVSHMEM_HOME")
    if nvshmem_home:
        logger.info(f"Found NVSHMEM_HOME from environment variable: {nvshmem_home}")
        return nvshmem_home

    # Try from Python import
    nvshmem_home = run_command(
        'python -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))"'
    )
    if nvshmem_home:
        logger.info(f"Found NVSHMEM_HOME from Python nvidia-nvshmem-cu12: {nvshmem_home}")
        return nvshmem_home

    # Try from ldconfig
    nvshmem_home = run_command("ldconfig -p | grep 'libnvshmem_host' | awk '{print $NF}' | xargs dirname | head -n 1")
    if nvshmem_home:
        logger.info(f"Found NVSHMEM_HOME from ldconfig: {nvshmem_home}")
        return nvshmem_home

    logger.warning("NVSHMEM_HOME could not be determined.")
    return ""


def get_ifname():
    """Auto-detect network interface for NVSHMEM_BOOTSTRAP."""
    preferred = ["bond0", "eth0", "enp", "ens"]
    interfaces = netifaces.interfaces()
    # Try preferred
    for p in preferred:
        for iface in interfaces:
            if iface.startswith(p):
                return iface
    # Otherwise, pick first non-loopback active interface
    for iface in interfaces:
        if iface != "lo" and netifaces.AF_INET in netifaces.ifaddresses(iface):
            return iface
    return "eth0"  # fallback


if __name__ == "__main__":
    setup_env()
