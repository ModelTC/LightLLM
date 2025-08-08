import torch
import triton
import triton.language as tl
from typing import List, Callable


def custom_cat(tensors):
    """
    直接调用 torch 的 cat操作，会造成多个流同步阻塞，用 custom_cat 进行替换。
    注意返回的 tensor 为 cpu pin_memory 类型, 只会在一些特殊的场景使用。
    """
    if not isinstance(tensors, (list, tuple)):
        raise ValueError("Input must be a list of tensors")

    assert tensors[0].is_cuda and len(tensors[0].shape) == 1
    sizes = [t.shape[0] for t in tensors]
    dest_size = sum(sizes)
    out_tensor = torch.empty((dest_size,), dtype=tensors[0].dtype, device="cpu", pin_memory=True)

    start_loc = 0
    for t, size in zip(tensors, sizes):
        out_tensor[start_loc : (start_loc + size)].copy_(t, non_blocking=True)
        start_loc += size
    torch.cuda.current_stream().synchronize()

    return out_tensor


def torch_cat_3(tensors: List[torch.Tensor], dim=0):
    if not tensors:
        raise ValueError("at least one tensor")

    ref = tensors[0]
    assert ref.ndim == 3
    dim = dim % ref.ndim

    out = torch.empty(
        [sum(t.size(dim) for t in tensors) if i == dim else ref.size(i) for i in range(ref.ndim)],
        dtype=ref.dtype,
        device=ref.device,
    )

    pos = 0
    for t in tensors:
        if (size := t.size(dim)) > 0:
            slices = [slice(None)] * ref.ndim
            slices[dim] = slice(pos, pos + size)
            tensor_copy_3dim(out[tuple(slices)], t)
            # out[tuple(slices)].copy_(t, non_blocking=True)
            pos += size

    return out


@triton.jit
def _tensor_copy_3dim(
    in_ptr,
    in_stride_0,
    in_stride_1,
    in_stride_2,
    out_ptr,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    head_num,
    head_dim,
    total_len,
    BLOCK_N: tl.constexpr,
):
    start_index = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offs_d = tl.arange(0, BLOCK_N)
    for cur_index in range(start_index, total_len, step=grid_num):
        for cur_head in tl.range(head_num, num_stages=3):
            in_tensor = tl.load(
                in_ptr + in_stride_0 * cur_index + in_stride_1 * cur_head + offs_d, mask=offs_d < head_dim, other=0
            )
            tl.store(
                out_ptr + out_stride_0 * cur_index + out_stride_1 * cur_head + offs_d, in_tensor, mask=offs_d < head_dim
            )
    return


@torch.no_grad()
def tensor_copy_3dim(dest_tensor: torch.Tensor, source_tensor: torch.Tensor):
    assert dest_tensor.shape == source_tensor.shape
    assert dest_tensor.ndim == 3
    assert source_tensor.stride(2) == 1 and dest_tensor.stride(2) == 1
    seq_len, head_num, head_dim = source_tensor.shape
    BLOCK_N = triton.next_power_of_2(head_dim)

    if BLOCK_N <= 256:
        num_warps = 1
    elif BLOCK_N <= 1024:
        num_warps = 4
    else:
        num_warps = 8

    if seq_len <= 16 * 1024:
        grid = (seq_len,)
    else:
        grid = (16 * 1024,)

    _tensor_copy_3dim[grid](
        source_tensor,
        *source_tensor.stride(),
        dest_tensor,
        *dest_tensor.stride(),
        head_num,
        head_dim,
        seq_len,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return


def pad2dim_tensor_to_new_batch(input: torch.Tensor, new_batch_size: int):
    assert input.ndim == 2
    origin_batch_size = input.shape[0]
    hidden = input.shape[1]
    out = torch.empty((new_batch_size, hidden), dtype=input.dtype, device=input.device, requires_grad=False)
    out[0:origin_batch_size, :] = input
    out[origin_batch_size:, :] = input[0:1, :]
    return out


def error(y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)

    SNR can be calcualted as following equation:

        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2

    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.

        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)

    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    y_pred = torch.flatten(y_pred).float()
    y_real = torch.flatten(y_real).float()

    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute snr loss for tensors with different shape. ({y_pred.shape} and {y_real.shape})"
        )

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)
    return snr.item()


def benchmark(func: Callable, shape: List[int], tflops: float, steps: int, *args, **kwargs):
    """
    A decorator function to assist in performance testing of CUDA operations.

    This function will:
    1. Automatically determine whether any parameters in the argument list,
       or the output of the `func`, are of type `torch.Tensor`.
    2. If so, calculate the memory usage of the input and output tensors
       on the GPU (based on their data type and `torch.numel()`).
    3. Establish a CUDA graph and attempt to execute `func` repeatedly for `steps` iterations.
    4. Record the execution time during these iterations.
    5. Use the information above to compute the compute performance (TFLOPS) and memory throughput.

    Args:
        func (function): The function to benchmark.
        shape (list of int): The problem shape.
        tflops (float): The computational workload (in TFLOPS) per call of `func`.
        steps (int): The number of times the function is executed during benchmarking.
        *args: Positional arguments to be passed to the `func`.
        **kwargs: Keyword arguments to be passed to the `func`.

    Returns:
        function result
    """

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking.")

    # Check for torch.Tensor in inputs and outputs
    input_tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    input_tensors += [value for value in kwargs.values() if isinstance(value, torch.Tensor)]

    def calculate_memory(tensor: torch.Tensor):
        """Calculate memory usage in bytes for a tensor."""
        return tensor.numel() * tensor.element_size()

    input_memory = sum(calculate_memory(t) for t in input_tensors)

    # Execute the function to inspect outputs
    with torch.no_grad():
        output = func(*args, **kwargs)

    output_memory = 0
    if isinstance(output, torch.Tensor):
        output_memory = calculate_memory(output)
    elif isinstance(output, (list, tuple)):
        output_memory = sum(calculate_memory(o) for o in output if isinstance(o, torch.Tensor))

    total_memory = input_memory + output_memory

    # Warm-up and CUDA graph creation
    for _ in range(10):  # Warm-up
        func(*args, **kwargs)

    torch.cuda.synchronize()  # Ensure no pending operations

    # Benchmark the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        func(*args, **kwargs)
    end_event.record()

    torch.cuda.synchronize()  # Ensure all operations are finished
    elapsed_time_ms = start_event.elapsed_time(end_event)  # Time in milliseconds

    # Calculate performance metrics
    elapsed_time_s = elapsed_time_ms / 1000  # Convert to seconds
    avg_time_per_step = elapsed_time_s / steps
    compute_performance = tflops / avg_time_per_step  # TFLOPS
    memory_throughput = (total_memory * steps / (1024 ** 3)) / elapsed_time_s  # GB/s

    # Print performance metrics
    print(f"Function: {func.__name__}{shape}")
    # print(f"Function: {func.__ne__}{shape}")
    print(f"Elapsed Time (total): {elapsed_time_s:.4f} seconds")
    print(f"Average Time Per Step: {avg_time_per_step * 1000:.3f} ms")
    print(f"Compute Performance: {compute_performance:.2f} TFLOPS")
    print(f"Memory Throughput: {memory_throughput:.2f} GB/s")
    print("")  # print a blank line.
