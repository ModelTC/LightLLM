import numpy as np
from multiprocessing import Queue
import multiprocessing
import os
import time

from lightllm.models.vit.model import VisionTransformer


def test_model_inference(world_size, weight_dir, quant_type=None):
    workers = []
    for rank_id in range(world_size):
        kvargs = {
            "tp_rank": rank_id,
            "world_size": world_size,
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
    from lightllm.distributed import set_custom_reduce
    import torch.distributed as dist

    rank_id = model_kvargs["tp_rank"]
    world_size = model_kvargs["world_size"]

    torch.cuda.set_device(rank_id)
    os.environ["CURRENT_DEVICE_ID"] = str(rank_id)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:28765", rank=rank_id, world_size=world_size)
    set_custom_reduce()
    dist.barrier()
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
    weight_dir = "your_multimodal_vit_path"
    torch.multiprocessing.set_start_method("spawn")
    test_model_inference(world_size, weight_dir, "vllm-w8a8")
