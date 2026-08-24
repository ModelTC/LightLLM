import gc
import os
import time

import torch
from safetensors import safe_open
from tqdm import tqdm

import lightllm.utils.petrel_helper as utils
from lightllm.utils.dist_utils import get_current_device_id, get_global_rank


def _apply_weights(weights, pre_post_layer, transformer_layer_list):
    if pre_post_layer is not None:
        pre_post_layer.load_hf_weights(weights)
    if transformer_layer_list is not None:
        for layer in transformer_layer_list:
            layer.load_hf_weights(weights)


def load_func(file_, use_safetensors=False, pre_post_layer=None, transformer_layer_list=None, weight_dir=None):
    # fix bug for 多线程加载的时候，每个线程内部的cuda device 会切回 0， 修改后来保证不会出现bug
    torch.cuda.set_device(get_current_device_id())

    if use_safetensors:
        weights = safe_open(os.path.join(weight_dir, file_), "pt", "cpu")
        weights = {k: weights.get_tensor(k) for k in weights.keys()}
    else:
        weights = utils.PetrelHelper.load(os.path.join(weight_dir, file_), map_location="cpu")

    _apply_weights(weights, pre_post_layer, transformer_layer_list)
    del weights


def load_hf_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None, weight_dict=None):
    if isinstance(data_type, str):
        data_type = torch.float16 if data_type == "fp16" else torch.float32
    if pre_post_layer is not None and pre_post_layer.data_type_ != data_type:
        raise ValueError(
            f"pre/post layer dtype mismatch: expected {data_type}, got {pre_post_layer.data_type_}"
        )
    if transformer_layer_list is not None and transformer_layer_list[0].data_type_ != data_type:
        raise ValueError(
            "transformer layer dtype mismatch: "
            f"expected {data_type}, got {transformer_layer_list[0].data_type_}"
        )
    if weight_dict:
        _apply_weights(weight_dict, pre_post_layer, transformer_layer_list)
        del weight_dict
        return
    use_safetensors = True
    files = utils.PetrelHelper.list(weight_dir, extension="all")
    candidate_files = list(filter(lambda x: x.endswith(".safetensors"), files))
    if len(candidate_files) == 0:
        use_safetensors = False
        candidate_files = list(filter(lambda x: x.endswith(".bin"), files))
    if not candidate_files:
        raise RuntimeError(f"no .safetensors or .bin weight files found in {weight_dir}")
    candidate_files.sort()

    from lightllm.utils.envs_utils import get_env_start_args

    distributed_mode = get_env_start_args().distributed_weight_load
    if distributed_mode != "disabled":
        if not use_safetensors:
            raise RuntimeError("distributed weight loading only supports safetensors checkpoints")

        from lightllm.common.basemodel.layer_weights.distributed_safetensor_loader import (
            DistributedSafetensorLoader,
        )

        loader = DistributedSafetensorLoader(distributed_mode)
        iterator = enumerate(candidate_files)
        iterator = tqdm(
            iterator,
            total=len(candidate_files),
            desc=f"Distributed model weight loading ({distributed_mode})",
            disable=not loader.is_progress_rank,
        )
        for shard_index, file_ in iterator:
            shard = loader.load_file(os.path.join(weight_dir, file_), shard_index)
            if shard.expert_weights:
                _apply_weights(shard.expert_weights, pre_post_layer, transformer_layer_list)
            shard.wait_for_replicated_weights()
            if shard.replicated_weights:
                _apply_weights(shard.replicated_weights, pre_post_layer, transformer_layer_list)
            del shard
        gc.collect()
        if loader.is_progress_rank:
            print(f"Distributed model weight loading finished: {loader.summary()}", flush=True)
        return

    from functools import partial
    from multiprocessing.pool import ThreadPool as Pool

    partial_func = partial(
        load_func,
        use_safetensors=use_safetensors,
        pre_post_layer=pre_post_layer,
        transformer_layer_list=transformer_layer_list,
        weight_dir=weight_dir,
    )  # noqa
    worker = int(os.environ.get("LOADWORKER", 1))
    started_at = time.perf_counter()
    with Pool(worker) as p:
        iterator = p.imap_unordered(partial_func, candidate_files, chunksize=1)
        desc_str = f"pid {os.getpid()} Loading model weights with {worker} workers"
        iterator = tqdm(iterator, total=len(candidate_files), desc=desc_str, disable=get_global_rank() != 0)

        for _ in iterator:
            pass

    gc.collect()
    if get_global_rank() == 0:
        elapsed = time.perf_counter() - started_at
        print(f"Model weight loading finished: mode=legacy, workers={worker}, elapsed={elapsed:.2f}s", flush=True)
    return
