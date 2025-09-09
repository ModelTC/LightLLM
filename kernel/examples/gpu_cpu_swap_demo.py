import torch

torch.cuda.set_device(0)

data_size = 1024 * 2 * 128  # *1024*1024
dim = 8

gpu_tensor = torch.zeros((data_size, 1), dtype=torch.float16, device="cuda").normal_(0, 1)
gpu_tensor = gpu_tensor.view(-1, dim)
cpu_tensor = torch.zeros((data_size, 1), dtype=torch.float16, device="cpu").view(-1, dim).pin_memory()
cpu_tensor = cpu_tensor.view(-1, dim)

source_index = torch.range(0, gpu_tensor.shape[0] - 1, dtype=torch.int64, device="cuda")
dest_index = torch.range(gpu_tensor.shape[0] - 1, 0, step=-1, dtype=torch.int64, device="cuda")

print(gpu_tensor.shape, source_index.shape, cpu_tensor.shape[0])

grid_num = cpu_tensor.shape[0]
wrap_num = 1

stream = torch.cuda.Stream()
stream.cuda_stream
import time

torch.cuda.synchronize()
start = time.time()
from gpu_cpu_swap import swap_data_by_index

swap_data_by_index(cpu_tensor, dest_index, gpu_tensor, source_index)
torch.cuda.synchronize()
duration = time.time() - start
print(duration * 1000, "ms")
bandwidth = (data_size * 2) / (duration) / (1024 ** 3)  # MB/s
print(bandwidth, "GB/s")


stream = torch.cuda.Stream()
import time

torch.cuda.synchronize()
start = time.time()
from gpu_cpu_swap import swap_data_by_index

swap_data_by_index(cpu_tensor, dest_index, gpu_tensor, source_index)
torch.cuda.synchronize()

print((time.time() - start) * 1000, "ms")

print(torch.max(torch.abs(cpu_tensor.cuda()[dest_index] - gpu_tensor)))
