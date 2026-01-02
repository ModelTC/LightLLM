import os
import time
import torch
import shutil
import subprocess
from functools import lru_cache
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def is_tesla():
    return "Tesla" in torch.cuda.get_device_name(0)


@lru_cache(maxsize=None)
def is_hopper():
    return (
        "H100" in torch.cuda.get_device_name(0)
        or "H200" in torch.cuda.get_device_name(0)
        or "H800" in torch.cuda.get_device_name(0)
        or "Hopper" in torch.cuda.get_device_name(0)
    )


@lru_cache(maxsize=None)
def is_4090():
    return "4090" in torch.cuda.get_device_name(0) or "RTX 4090" in torch.cuda.get_device_name(0)


@lru_cache(maxsize=None)
def get_device_sm_count():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["multiprocessor_count"]


@lru_cache(maxsize=None)
def get_device_sm_regs_num():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["max_num_regs"]


@lru_cache(maxsize=None)
def get_device_sm_shared_mem_num():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["max_shared_mem"]


@lru_cache(maxsize=None)
def get_device_warp_size():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["warpSize"]


def calcu_kernel_best_vsm_count(kernel, num_warps):
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared

    sm_count = get_device_sm_count()
    max_regs = get_device_sm_regs_num()
    shared_mem_max = get_device_sm_shared_mem_num()
    warp_size = get_device_warp_size()

    occupancy = max_regs // (n_regs * warp_size * num_warps)
    if size_smem > 0:
        occupancy = min(occupancy, shared_mem_max // size_smem)
    num_sm = sm_count * occupancy
    return num_sm


@lru_cache(maxsize=1)
def is_musa():
    return hasattr(torch.version, "musa") and torch.version.musa is not None


@lru_cache(maxsize=None)
def get_current_device_name():
    if torch.cuda.is_available() or is_musa():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        # 4090 trans to 4090 D
        if "4090" in gpu_name and "4090 D" not in gpu_name:
            gpu_name = gpu_name.replace("4090", "4090 D")

        gpu_name = gpu_name.replace(" ", "_")
        return gpu_name
    else:
        return None


@lru_cache(maxsize=None)
def init_p2p(device_index):
    """
    torch 调用跨卡的to操作后，triton编译的算子便能自动操作跨卡tensor。
    """
    num_gpus = torch.cuda.device_count()
    tensor = torch.zeros((1,))
    tensor = tensor.to(f"cuda:{device_index}")
    for j in range(num_gpus):
        tensor.to(f"cuda:{j}")

    torch.cuda.empty_cache()
    return


@lru_cache(maxsize=None)
def kv_trans_use_p2p():
    return not (os.getenv("DISABLE_KV_TRANS_USE_P2P", "False").upper() in ["1", "TRUE", "ON"])


def has_nvlink():
    try:
        # Call nvidia-smi to get the topology matrix
        result = subprocess.check_output(["nvidia-smi", "topo", "--matrix"])
        result = result.decode("utf-8")
        # Check if the output contains 'NVLink'
        return any(f"NV{i}" in result for i in range(1, 8))
    except FileNotFoundError:
        # nvidia-smi is not installed, assume no NVLink
        return False
    except subprocess.CalledProcessError:
        # If there's an error while executing nvidia-smi, assume no NVLink
        return False


def has_mtlink():
    try:
        # Call mthreads-gmi to get the topology matrix
        result = subprocess.check_output(["mthreads-gmi", "topo", "--matrix"])
        result = result.decode("utf-8")
        # Check if the output contains 'MTLink'
        return any(f"MT{i}" in result for i in range(1, 8))
    except FileNotFoundError:
        # mthreads-gmi is not installed, assume no MTLink
        return False
    except subprocess.CalledProcessError:
        # If there's an error while executing mthreads-gmi, assume no MTLink
        return False


def is_mps_running(verbose=False):
    result = subprocess.run(
        "ps -ef | grep '[n]vidia-cuda-mps-control'",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0


def stop_mps():
    if is_mps_running():
        result = subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True)
        logger.info("Stopping MPS...")
        if result.returncode == 0:
            logger.info("MPS stopped successfully.")
        else:
            logger.warning("Failed to stop MPS.")
    else:
        logger.info("MPS is not running, no need to stop.")


def enable_mps():
    if is_mps_running():
        logger.info("MPS is already running, no need to start.")
        return

    ret = os.system("nvidia-cuda-mps-control -d")

    time.sleep(10)
    if ret != 0:
        logger.warning("Failed to start MPS.")
        return
    if is_mps_running():
        logger.info("MPS started successfully.")
    return


def get_gpu_compute_mode(gpu_index=0):
    try:
        if not shutil.which("nvidia-smi"):
            logger.warning("nvidia-smi not found in PATH.")
            return None

        cmd = ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=compute_mode", "--format=csv,noheader"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            logger.warning(f"Failed to query compute mode: {result.stderr.strip()}")
            return None

        mode = result.stdout.strip()
        return mode

    except Exception as e:
        logger.warning(f"Exception occurred while checking GPU compute mode: {e}")
        return None


def set_gpu_exclusive_mode(gpu_index=0):
    logger.info(f"Setting GPU {gpu_index} to EXCLUSIVE_PROCESS mode...")
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index), "-c", "EXCLUSIVE_PROCESS"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        logger.info(f"GPU {gpu_index} set to EXCLUSIVE_PROCESS mode.")
        return True
    else:
        logger.warning(f"Failed to set EXCLUSIVE_PROCESS mode: {result.stderr.strip()}")
        return False


def set_gpu_default_mode(gpu_index=0):
    logger.info(f"Setting GPU {gpu_index} to DEFAULT mode...")
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index), "-c", "DEFAULT"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode == 0:
        logger.info(f"GPU {gpu_index} set to DEFAULT mode.")
        return True
    else:
        logger.warning(f"Failed to set DEFAULT mode: {result.stderr.strip()}")
        return False


def set_sm_limit(percent: int, gpu_index=0):
    """
    Sets CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to the given value if the GPU is in EXCLUSIVE_PROCESS mode.
    """
    if not (1 <= percent <= 100):
        logger.error("SM usage percentage must be between 1 and 100.")
        return False

    mode = get_gpu_compute_mode(gpu_index)
    if mode != "Exclusive_Process":
        logger.warning(f"Cannot set SM limit. GPU {gpu_index} is in '{mode}' mode, not 'Exclusive_Process'.")
        return False

    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percent)
    logger.info(f"Set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to {percent}% for GPU {gpu_index}.")
    return True


@lru_cache(maxsize=None)
def triton_support_tensor_descriptor() -> bool:
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor

        support_tma = torch.cuda.get_device_capability() >= (9, 0)
        if support_tma:
            logger.info("triton support tensor_descriptor")
            return True
        else:
            assert False
    except:
        logger.info("triton not support tensor_descriptor")
        return False


@lru_cache(maxsize=None)
def is_5090_gpu() -> bool:
    try:
        gpu_name = torch.cuda.get_device_name(0)
        if "5090" in gpu_name:
            return True
        else:
            return False
    except:
        return False
