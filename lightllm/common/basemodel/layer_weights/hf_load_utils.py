import torch
import os
import gc
from safetensors import safe_open
import lightllm.utils.petrel_helper as utils
from lightllm.utils.dist_utils import get_current_device
from lightllm.utils.progress import ThreadProgress


def load_func(file_, use_safetensors=False, pre_post_layer=None, transformer_layer_list=None, weight_dir=None):
    # fix bug for 多线程加载的时候，每个线程内部的cuda device 会切回 0， 修改后来保证不会出现bug
    import torch.distributed as dist

    device = get_current_device()
    if device.type not in ("cpu", "meta"):
        getattr(torch, device.type).set_device(device)

    if use_safetensors:
        weights = safe_open(os.path.join(weight_dir, file_), "pt", "cpu")
        weights = {k: weights.get_tensor(k) for k in weights.keys()}
    else:
        weights = utils.PetrelHelper.load(os.path.join(weight_dir, file_), map_location="cpu")

    if pre_post_layer is not None:
        pre_post_layer.load_hf_weights(weights)
    if transformer_layer_list is not None:
        for layer in transformer_layer_list:
            layer.load_hf_weights(weights)
    del weights
    gc.collect()


def checkpoint_files(weight_dir):
    files = utils.PetrelHelper.list(weight_dir, extension="all")
    candidates = [name for name in files if name.endswith(".safetensors")]
    use_safetensors = bool(candidates)
    if not candidates:
        candidates = [name for name in files if name.endswith(".bin")]
    assert candidates, "can only support pytorch tensor and safetensors format for weights."
    return use_safetensors, candidates


def checkpoint_weight_names(weight_dir, weight_dict=None):
    """Inspect the selected checkpoint layout before allocating model weights."""
    if weight_dict:
        return frozenset(weight_dict)
    use_safetensors, files = checkpoint_files(weight_dir)
    names = set()
    for name in files:
        path = os.path.join(weight_dir, name)
        if use_safetensors:
            with safe_open(path, "pt", "cpu") as weights:
                names.update(weights.keys())
        else:
            names.update(utils.PetrelHelper.load(path, map_location="meta"))
    return frozenset(names)


def load_hf_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None, weight_dict=None):
    if isinstance(data_type, str):
        data_type = torch.float16 if data_type == "fp16" else torch.float32
    if pre_post_layer is not None:
        assert pre_post_layer.data_type_ == data_type, "type is not right"
    if transformer_layer_list is not None:
        assert transformer_layer_list[0].data_type_ == data_type, "type is not right"
    if weight_dict:
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weight_dict)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weight_dict)
        del weight_dict
        return
    use_safetensors, candidate_files = checkpoint_files(weight_dir)
    from functools import partial
    from concurrent.futures import ThreadPoolExecutor, as_completed

    partial_func = partial(
        load_func,
        use_safetensors=use_safetensors,
        pre_post_layer=pre_post_layer,
        transformer_layer_list=transformer_layer_list,
        weight_dir=weight_dir,
    )  # noqa
    worker = int(os.environ.get("LOADWORKER", 1))
    with ThreadPoolExecutor(max_workers=worker) as pool:
        futures = [pool.submit(partial_func, file) for file in candidate_files]
        desc_str = f"pid {os.getpid()} Loading model weights with {worker} workers"
        for future in ThreadProgress(as_completed(futures), total=len(candidate_files), desc=desc_str):
            future.result()

    return
