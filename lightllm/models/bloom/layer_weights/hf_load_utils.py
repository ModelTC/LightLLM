import torch
import os
import gc
from safetensors import safe_open


def load_hf_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None, weight_dict=None):
    data_type = torch.float16 if data_type == 'fp16' else torch.float32
    if pre_post_layer is not None:
        assert pre_post_layer.data_type_ == data_type, "type is not right"
    if transformer_layer_list is not None:
        assert transformer_layer_list[0].data_type_ == data_type, "type is not right"
    if weight_dict:
        new_w = {}
        for k,v in weight_dict.items():
            if "transformer." in k:
                new_w[k[len("transformer."):]] = v
            else:
                new_w[k] = v
        del weight_dict
        weight_dict = new_w
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weight_dict)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weight_dict)
        del weight_dict
        return
    use_safetensors = True
    files = os.listdir(weight_dir)
    candidate_files = list(filter(lambda x : x.endswith('.safetensors'), files))
    if len(candidate_files) == 0:
        use_safetensors = False
        candidate_files = list(filter(lambda x : x.endswith('.bin'), files))
    assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."
    for file_ in candidate_files:
        if use_safetensors:
            weights = safe_open(os.path.join(weight_dir, file_), 'pt', 'cpu')
            weights = {k: weights.get_tensor(k) for k in weights.keys()}
        else:
            weights = torch.load(os.path.join(weight_dir, file_), 'cpu')
        new_w = {}
        for k,v in weights.items():
            if "transformer." in k:
                new_w[k[len("transformer."):]] = v
            else:
                new_w[k] = v
        del weights
        weights = new_w
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weights)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weights)
        del weights
        gc.collect()
    return
