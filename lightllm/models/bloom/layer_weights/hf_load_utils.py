import torch
import os
import gc


def load_hf_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None):
    data_type = torch.float16 if data_type == 'fp16' else torch.float32
    for file_ in os.listdir(weight_dir):
        if not file_.endswith(".bin"):
            continue
        weights = torch.load(os.path.join(weight_dir, file_), 'cpu')
        new_weights = {}
        for key, v in weights.items():
            # print(f"name:{key} : shape:{v.shape}")
            if "transformer." in key:
                new_weights[key[len("transformer."):]] = v
            else:
                new_weights[key] = v

        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(new_weights)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(new_weights)
        del weights
        del new_weights
        gc.collect()
    for layer in transformer_layer_list:
        layer.init_hf_alibi()
    return
