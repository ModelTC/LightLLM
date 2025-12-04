import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

torch.cuda.synchronize()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/"),
) as prof:
    # test cuda code
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
