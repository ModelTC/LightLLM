import torch


def load_ft_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None):
    data_type = torch.float16 if data_type == 'fp16' else torch.float32
    if pre_post_layer is not None:
        pre_post_layer.load_ft_weights(weight_dir=weight_dir)
    if transformer_layer_list is not None:
        for trans_layer in transformer_layer_list:
            trans_layer.load_ft_weights(weight_dir=weight_dir)
    return
