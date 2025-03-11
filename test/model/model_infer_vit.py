import numpy as np
from multiprocessing import Queue
import multiprocessing
import os
import time

from lightllm.models.vit.model import VisionTransformer
from lightllm.utils.dist_utils import init_vision_distributed_env


def test_model_inference(world_size, weight_dir, quant_type=None):
    workers = []
    for rank_id in range(world_size):
        kvargs = {
            "vit_tp": world_size,
            "tp_rank_id": rank_id,
            "vit_rank_id": rank_id,
            "visual_gpu_ids": list(range(world_size)),
            "visual_nccl_port": 28766,
            "weight_dir": weight_dir,
            "data_type": "bf16",
            "quant_type": quant_type,
            "quant_cfg": None,
        }

        proc = multiprocessing.Process(target=tppart_model_infer, args=(kvargs,))
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()
    return


def tppart_model_infer(model_kvargs):
    import torch
    from lightllm.distributed import custom_comm_ops
    import torch.distributed as dist

    rank_id = model_kvargs["tp_rank_id"]
    init_vision_distributed_env(model_kvargs)
    custom_comm_ops.set_custom_reduce()

    torch.cuda.empty_cache()
    model_part = VisionTransformer(model_kvargs)
    test_data = torch.randn((13, 3, 448, 448)).cuda().to(torch.bfloat16)
    # warm up
    torch.cuda.synchronize()
    for i in range(10):
        model_part.forward(test_data)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(50):
        model_part.forward(test_data)
    torch.cuda.synchronize()
    end_time = time.time()
    if rank_id == 0:
        print("time total cost(ms):", (end_time - start_time) / 50 * 1000)

    return


if __name__ == "__main__":
    import torch

    world_size = 2
    weight_dir = "/nvme/models/InternVL2/InternVL2-8B/"
    torch.multiprocessing.set_start_method("spawn")
    test_model_inference(world_size, weight_dir, "none")
