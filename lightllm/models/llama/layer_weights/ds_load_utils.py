import collections
import torch
import os
import gc

def load_ds_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None, weight_dict=None, prefix="", num_layer=0):
    if weight_dict:
        return weight_dict
    files = os.listdir(weight_dir)
    candidate_files = sorted(list(filter(lambda x : x.endswith('.pt') and x.startswith('layer'), files)))
    assert len(candidate_files) != 0, "can only support pytorch tensor format for weights."
    if weight_dict:
        weights_all = weight_dict
    else:
        weights_all = {}
        for file_ in candidate_files:
            file_split = file_.split('-')
            layer_num = int(file_split[0].split('_')[-1])
            rank_num = int(file_split[0].split('_')[-1])
            weights = torch.load(os.path.join(weight_dir, file_), 'cpu')
            for k,v in weights.items():
                if layer_num >=3 and layer_num < 3 + num_layer:
                    k = prefix + str(layer_num - 3) + '.' + k
                if layer_num == num_layer + 5:
                    k = 'lm_head.weight'
                if layer_num == num_layer + 4:
                    k = 'model.norm.weight'
                if layer_num == 1:
                    k = 'model.embed_tokens.weight'
                if k not in weights_all:
                    weights_all[k] = v 
                else:
                    if 'q_proj' in k or 'k_proj' in k or 'v_proj' in k or 'gate_proj' in k or 'up_proj' in k:
                        weights_all[k] = torch.cat([weights_all[k], v], dim=0)
                    elif 'o_proj' in k or 'down_proj' in k:
                        weights_all[k] = torch.cat([weights_all[k], v], dim=1)
                    else:
                        weights_all[k] = v
    if pre_post_layer is not None:
        pre_post_layer.load_hf_weights(weights_all)
    if transformer_layer_list is not None:
        for layer in transformer_layer_list:
            layer.load_hf_weights(weights_all)
    del weights_all
    gc.collect()
    return

if __name__ == '__main__':
    load_ds_weight('fp16', '/nvme/baishihao/llama7b', prefix='model.layers.', num_layer=32)