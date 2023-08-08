import torch
import os
import gc


def load_hf_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None):
    data_type = torch.float16 if data_type == 'fp16' else torch.float32
    if pre_post_layer is not None:
        assert pre_post_layer.data_type_ == data_type, "type is not right"
    if transformer_layer_list is not None:
        assert transformer_layer_list[0].data_type_ == data_type, "type is not right"
    for file_ in os.listdir(weight_dir):
        if not file_.endswith(".bin"):
            continue
        weights = torch.load(os.path.join(weight_dir, file_), 'cpu')
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weights)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weights)
        del weights
        gc.collect()
    return
