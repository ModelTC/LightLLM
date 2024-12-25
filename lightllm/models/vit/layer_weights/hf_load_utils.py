import torch
import os
import gc
from safetensors import safe_open
import lightllm.utils.petrel_helper as utils


def load_func(file_, use_safetensors=False, pre_post_layer=None, transformer_layer_list=None, weight_dir=None):
    if use_safetensors:
        weights = safe_open(os.path.join(weight_dir, file_), "pt", "cpu")
        weights = {k: weights.get_tensor(k) for k in weights.keys()}
    else:
        weights = utils.PetrelHelper.load(os.path.join(weight_dir, file_), map_location="cpu")
        new_weight = {}
        for k, v in weights.items():
            if "language_model." in k:
                new_weight[k[len("language_model.") :]] = v
            else:
                new_weight[k] = v
        del weights
        weights = new_weight
    if pre_post_layer is not None:
        pre_post_layer.load_hf_weights(weights)
    if transformer_layer_list is not None:
        for layer in transformer_layer_list:
            layer.load_hf_weights(weights)
    del weights
    gc.collect()


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
    use_safetensors = True
    files = utils.PetrelHelper.list(weight_dir, extension="all")
    candidate_files = list(filter(lambda x: x.endswith(".safetensors"), files))
    if len(candidate_files) == 0:
        use_safetensors = False
        candidate_files = list(sorted(filter(lambda x: x.endswith(".bin"), files)))
        candidate_files = candidate_files[0:45] + [candidate_files[-1]]
        print(candidate_files)
    assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."
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
    with Pool(worker) as p:
        _ = p.map(partial_func, candidate_files)
    return
