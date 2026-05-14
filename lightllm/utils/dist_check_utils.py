"""
通过双卡 NCCL 任务，对可选的 all-reduce 快路径做环境探测。

每个后端用 ``torch.multiprocessing.spawn(..., nprocs=2)`` 起两个子进程：
初始化分布式后做一次真实集合通信，再退出。
"""

import socket
import threading
from typing import TYPE_CHECKING, Callable

from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.server.core.objs.start_args_type import StartArgs

logger = init_logger(__name__)

_CUSTOM_ALLREDUCE_WORLD_SIZES = (2, 4, 6, 8)
_TWO_GPU_CHECK_TIMEOUT_SECONDS = 60.0


def _start_two_gpu_check_timeout_watchdog(backend_name: str) -> threading.Event:
    """Each spawned rank runs its own watchdog thread; exits the process if the check does not finish in time."""

    import os
    import time

    probe_finished = threading.Event()

    def watchdog_main() -> None:
        time.sleep(_TWO_GPU_CHECK_TIMEOUT_SECONDS)
        if not probe_finished.is_set():
            logger.warning(
                "%s 2-GPU all-reduce capability check timed out after %.0fs; force exit.",
                backend_name,
                _TWO_GPU_CHECK_TIMEOUT_SECONDS,
            )
            os._exit(1)

    watchdog_thread = threading.Thread(target=watchdog_main, daemon=True)
    watchdog_thread.start()
    return probe_finished


def _should_run_allreduce_capability_check(args: "StartArgs") -> bool:
    if args.hardware_platform != "cuda":
        return False

    return (args.tp // args.dp) in _CUSTOM_ALLREDUCE_WORLD_SIZES


def _pick_free_tcp_port() -> int:
    socket_handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_handle.bind(("127.0.0.1", 0))
    free_port = socket_handle.getsockname()[1]
    socket_handle.close()
    return int(free_port)


def _flashinfer_two_gpu_check_worker(process_rank: int, init_tcp_port: int) -> None:
    probe_finished_event = _start_two_gpu_check_timeout_watchdog("FlashInfer")
    try:
        import torch
        import torch.distributed as dist

        cuda_device = torch.device(f"cuda:{process_rank}")
        torch.cuda.set_device(cuda_device)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{init_tcp_port}",
            world_size=2,
            rank=process_rank,
            device_id=cuda_device,
        )
        try:
            gloo_process_group = dist.new_group([0, 1], backend="gloo")
            from lightllm.distributed.flashinfer_all_reduce import FlashInferAllReduce

            flashinfer_all_reduce = FlashInferAllReduce(gloo_process_group, cuda_device)
            if flashinfer_all_reduce.disabled:
                raise RuntimeError("FlashInferAllReduce disabled")
            if process_rank == 0:
                input_tensor = torch.zeros(2, 64, device=cuda_device, dtype=torch.bfloat16)
            else:
                input_tensor = torch.ones(2, 64, device=cuda_device, dtype=torch.bfloat16)
            output_tensor = flashinfer_all_reduce.all_reduce(input_tensor)
            dist.barrier()
            expected_reduced = torch.ones(2, 64, device=cuda_device, dtype=torch.bfloat16)
            if not torch.allclose(output_tensor, expected_reduced):
                raise RuntimeError("FlashInfer allreduce value mismatch")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    finally:
        probe_finished_event.set()


def _symm_mem_two_gpu_check_worker(process_rank: int, init_tcp_port: int) -> None:
    probe_finished_event = _start_two_gpu_check_timeout_watchdog("SymmMem")
    try:
        import torch
        import torch.distributed as dist

        cuda_device = torch.device(f"cuda:{process_rank}")
        torch.cuda.set_device(cuda_device)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{init_tcp_port}",
            world_size=2,
            rank=process_rank,
            device_id=cuda_device,
        )
        try:
            nccl_process_group = dist.new_group([0, 1], backend="nccl")
            from lightllm.distributed.symm_mem_all_reduce import SymmMemAllreduce

            symm_mem_all_reduce = SymmMemAllreduce(nccl_process_group, cuda_device, dtype=torch.bfloat16)
            if symm_mem_all_reduce.disabled:
                raise RuntimeError("SymmMemAllreduce disabled")
            if process_rank == 0:
                activation_tensor = torch.zeros(8, 32, device=cuda_device, dtype=torch.bfloat16)
            else:
                activation_tensor = torch.ones(8, 32, device=cuda_device, dtype=torch.bfloat16)
            symm_mem_all_reduce.all_reduce(activation_tensor)
            dist.barrier()
            expected_reduced = torch.ones(8, 32, device=cuda_device, dtype=torch.bfloat16)
            if not torch.allclose(activation_tensor, expected_reduced):
                raise RuntimeError("SymmMem allreduce value mismatch")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    finally:
        probe_finished_event.set()


def _check_ok_two_gpu_all_reduce(worker_entry: Callable[[int, int], None], init_tcp_port: int) -> bool:
    import torch.multiprocessing as torch_mp

    try:
        torch_mp.spawn(worker_entry, args=(init_tcp_port,), nprocs=2, join=True)
        return True
    except Exception as error:
        error_str = str(error)
        error_str = error_str[-66:].replace("\n", "")
        logger.warning("2-GPU all-reduce capability check failed for %s: %s", worker_entry.__name__, error_str)
        return False


def auto_configure_allreduce_flags_from_args(args: "StartArgs") -> None:
    """
    用户若已通过 ``--disable_*`` 关闭某后端，则不再处理该后端。

    否则会按环境与并行规模，对每个后端做一次双进程 NCCL 探测；失败则将对应 ``disable_*`` 设为 True。

    会就地修改 ``args.disable_flashinfer_allreduce`` / ``args.disable_symm_mem_allreduce``。
    """
    if not _should_run_allreduce_capability_check(args):
        return

    if not args.disable_flashinfer_allreduce:
        if not _check_ok_two_gpu_all_reduce(_flashinfer_two_gpu_check_worker, _pick_free_tcp_port()):
            logger.info(
                "Auto-set disable_flashinfer_allreduce=True (2-GPU FlashInfer all-reduce capability check failed)."
            )
            args.disable_flashinfer_allreduce = True

    if not args.disable_symm_mem_allreduce:
        if not _check_ok_two_gpu_all_reduce(_symm_mem_two_gpu_check_worker, _pick_free_tcp_port()):
            logger.info("Auto-set disable_symm_mem_allreduce=True (2-GPU SymmMem all-reduce capability check failed).")
            args.disable_symm_mem_allreduce = True
