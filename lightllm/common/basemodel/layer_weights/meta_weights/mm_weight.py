import os
import torch
from .base_weight import BaseWeightTpl
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


def generate_scale_name(name):
    weight_scale_name = ".".join(name.split(".")[:-1] + ["weight_scale"])
    input_scale_name = ".".join(name.split(".")[:-1] + ["input_scale"])
    return weight_scale_name, input_scale_name


STATIC_QUANT = os.getenv("STATIC_QUANT", "0").upper() in ["1", "TRUE", "ON"]


class MMWeightTpl(BaseWeightTpl):
    def __init__(self, data_type):
        super().__init__()
        self.data_type_ = data_type
        self.quant_method = None
        self.weight = None
        self.bias = None
        self.weight_scale = None
        self.input_scale = None

    def set_quant_method(self, quant_method):
        self.quant_method = quant_method

    def mm(self, input_tensor, out=None, use_custom_tensor_mananger=True):
        if self.quant_method is not None:
            return self.quant_method.apply(
                input_tensor, self.weight, self.bias, out, use_custom_tensor_mananger=use_custom_tensor_mananger
            )
        if out is None:
            shape = (input_tensor.shape[0], self.weight.shape[1])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.bias is None:
            return torch.mm(input_tensor, self.weight, out=out)
        return torch.addmm(self.bias, input_tensor, self.weight, out=out)

    def _post_load_weights(self):
        if self.quant_method is not None:
            if STATIC_QUANT:
                if all(w is not None for w in [self.weight, self.weight_scale, self.input_scale]):
                    self.weight = self.quant_method.quantize((self.weight, self.weight_scale, self.input_scale))
            else:
                self.weight = self.quant_method.quantize(self.weight.to(self.data_type_).cuda(self.device_id_))
            return
        self.weight = self.weight.to(self.data_type_).transpose(0, 1).cuda(self.device_id_)


class MMWeight(MMWeightTpl):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(data_type)
        self.start = split_n_embed * self.tp_rank_
        self.end = split_n_embed * (self.tp_rank_ + 1)
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.weight_scale_name, self.input_scale_name = generate_scale_name(weight_name)

    def verify_load(self):
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.bias_name is not None:
            load_ok = load_ok and self.bias is not None
        return load_ok


class ROWMMWeight(MMWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)

    def load_hf_weights(self, weights):
        weight = None
        weight_scale = None
        input_scale = None
        if self.weight_name in weights:
            weight = weights[self.weight_name]
            self.weight = weight[self.start : self.end]
        if self.bias_name in weights:
            bias = weights[self.bias_name].to(self.data_type_)[self.start : self.end]
            self.bias = bias.cuda(self.device_id_)

        if STATIC_QUANT and self.weight_scale_name in weights:
            weight_scale = weights[self.weight_scale_name].to(torch.float)[self.start : self.end]
            self.weight_scale = weight_scale.cuda()

        if STATIC_QUANT and self.input_scale_name in weights:
            input_scale = weights[self.input_scale_name].to(torch.float)
            self.input_scale = input_scale.cuda()

        if weight is None and weight_scale is None and input_scale is None:
            return
        self._post_load_weights()
        return


class ROWMMWeightNoTP(ROWMMWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)
        self.start = 0
        self.end = split_n_embed


class COLMMWeight(MMWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)

    def load_hf_weights(self, weights):
        weight = None
        weight_scale = None
        input_scale = None
        if self.weight_name in weights:
            weight = weights[self.weight_name]
            self.weight = weight[:, self.start : self.end]
        if self.bias_name in weights:
            bias = weights[self.bias_name]
            self.bias = (bias / self.world_size_).to(self.data_type_).cuda(self.device_id_)

        if STATIC_QUANT and self.weight_scale_name in weights:
            weight_scale = weights[self.weight_scale_name].to(torch.float)
            self.weight_scale = weight_scale.cuda()

        if STATIC_QUANT and self.input_scale_name in weights:
            input_scale = weights[self.input_scale_name].to(torch.float)
            self.input_scale = input_scale.cuda()

        if weight is None and weight_scale is None and input_scale is None:
            return
        self._post_load_weights()
        return


class COLMMWeightNoTp(MMWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)
        self.start = 0
        self.end = split_n_embed

    def load_hf_weights(self, weights):
        weight = None
        if self.weight_name in weights:
            weight = weights[self.weight_name].to(self.data_type_)
            self.weight = weight[:, self.start : self.end]
        if self.bias_name in weights:
            bias = weights[self.bias_name]
            self.bias = bias.to(self.data_type_).cuda(self.tp_rank_)
        if weight is None:
            return
        self._post_load_weights()
        return


class MultiMMWeight(MMWeightTpl):
    def __init__(self, weight_names, data_type, split_n_embeds, bias_names=[]):
        super().__init__(data_type)
        if isinstance(split_n_embeds, int):
            self.split_n_embeds = [split_n_embeds] * len(weight_names)
        else:
            self.split_n_embeds = split_n_embeds

        self.starts = [i * self.tp_rank_ for i in self.split_n_embeds]
        self.ends = [i * (self.tp_rank_ + 1) for i in self.split_n_embeds]
        self.weight_names = weight_names
        self.bias_names = bias_names
        self.weight_scale_names = []
        self.input_scale_names = []
        for weight_name in weight_names:
            weight_scale_name, input_scale_name = generate_scale_name(weight_name)
            self.weight_scale_names.append(weight_scale_name)
            self.input_scale_names.append(input_scale_name)

        self.weights = [None] * len(self.weight_names)
        self.biases = [None] * len(self.bias_names)
        self.input_scales = [None] * len(self.weight_names)
        self.weight_scales = [None] * len(self.weight_names)
        self.has_bias = all(b is not None for b in self.bias_names) and len(bias_names) > 0

    def verify_load(self):
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.has_bias:
            load_ok = load_ok and self.bias is not None
        return load_ok


class MultiROWMMWeight(MultiMMWeight):
    def __init__(self, weight_names, data_type, split_n_embed, bias_names=[]):
        super().__init__(weight_names, data_type, split_n_embed, bias_names)

    def _fuse(self):
        if self.weight is None and all(w is not None for w in self.weights):
            self.weight = torch.cat(self.weights, dim=0)
            self._post_load_weights()

        if self.weight_scale is None and all(w is not None for w in self.weight_scales):
            self.weight_scale = torch.cat(self.weight_scales, dim=0).cuda()
            self._post_load_weights()

        if self.input_scale is None and all(w is not None for w in self.input_scales):
            input_scales = torch.stack(self.input_scales, dim=0)
            self.input_scale = torch.max(input_scales).cuda()
            self._post_load_weights()

        if self.has_bias:
            if self.bias is None and all(b is not None for b in self.biases):
                self.bias = torch.cat(self.biases, dim=0).cuda(self.device_id_)
        return self

    def load_hf_weights(self, weights):
        weight = None
        for i in range(len(self.weight_names)):
            if self.weight_names[i] in weights:
                weight = weights[self.weight_names[i]]
                self.weights[i] = weight[self.starts[i] : self.ends[i]]
            if self.has_bias and self.bias_names[i] in weights:
                bias = weights[self.bias_names[i]].to(self.data_type_)
                self.biases[i] = bias[self.starts[i] : self.ends[i]]
            if STATIC_QUANT and self.weight_scale_names[i] in weights:
                weight_scale = weights[self.weight_scale_names[i]][self.starts[i] : self.ends[i]]
                self.weight_scales[i] = weight_scale.to(torch.float)
            if STATIC_QUANT and self.input_scale_names[i] in weights:
                input_scale = weights[self.input_scale_names[i]].to(torch.float)
                self.input_scales[i] = input_scale

        self._fuse()
        return


class MultiROWMMWeightNoTP(MultiROWMMWeight):
    def __init__(self, weight_names, data_type, split_n_embed, bias_names=[]):
        super().__init__(weight_names, data_type, split_n_embed, bias_names)
        self.starts = [0 for i in self.split_n_embeds]
        self.ends = [i for i in self.split_n_embeds]


class MultiCOLMMWeight(MultiROWMMWeight):
    def __init__(self, weight_names, data_type, split_n_embed, bias_names=[]):
        super().__init__(weight_names, data_type, split_n_embed, bias_names)

    def load_hf_weights(self, weights):
        weight = None
        for i in range(len(self.weight_names)):
            if self.weight_names[i] in weights:
                weight = weights[self.weight_names[i]]
                self.weights[i] = weight[:, self.starts[i] : self.ends[i]]
            if self.has_bias and self.bias_names[i] in weights:
                bias = weights[self.bias_names[i]].to(self.data_type_)
                self.biases[i] = bias[:, self.starts[i] : self.ends[i]]
            if STATIC_QUANT and self.weight_scale_names[i] in weights:
                weight_scale = weights[self.weight_scale_names[i]]
                self.weight_scales[i] = weight_scale.to(torch.float)
            if STATIC_QUANT and self.input_scale_names[i] in weights:
                input_scale = weights[self.input_scale_names[i]].to(torch.float)
                self.input_scales[i] = input_scale
        self._fuse()
        return


class MultiCOLMMWeightNoTp(MultiROWMMWeightNoTP):
    def __init__(self, weight_names, data_type, split_n_embed, bias_names=[]):
        super().__init__(weight_names, data_type, split_n_embed, bias_names)

    def load_hf_weights(self, weights):
        weight = None
        for i in range(len(self.weight_names)):
            if self.weight_names[i] in weights:
                weight = weights[self.weight_names[i]].to(self.data_type_)
                self.weights[i] = weight[:, self.starts[i] : self.ends[i]]
            if self.has_bias and self.bias_names[i] in weights:
                bias = weights[self.bias_names[i]].to(self.data_type_)
                self.biases[i] = bias[:, self.starts[i] : self.ends[i]]
        self._fuse()
        return


class BMMWeightTpl(BaseWeightTpl):
    def __init__(self, data_type):
        super().__init__()
        self.data_type_ = data_type
        self.quant_method = None
        self.weight = None
        self.bias = None

    def set_quant_method(self, quant_method):
        self.quant_method = None

    def bmm(self, input_tensor, out=None, use_custom_tensor_mananger=True):
        if self.quant_method is not None:
            return self.quant_method.apply(input_tensor, self.weight, self.bias, out)
        if out is None:
            shape = (input_tensor.shape[0], input_tensor.shape[1], self.weight.shape[2])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.bias is None:
            return torch.bmm(input_tensor, self.weight, out=out)
        return torch.addbmm(self.bias, input_tensor, self.weight, out=out)

    def _post_load_weights(self):
        self.weight = self.weight.cuda(self.device_id_)


class BMMWeight(BMMWeightTpl):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(data_type)
        self.start = split_n_embed * self.tp_rank_
        self.end = split_n_embed * (self.tp_rank_ + 1)
        self.weight_name = weight_name
        self.bias_name = bias_name

    def verify_load(self):
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.bias_name is not None:
            load_ok = load_ok and self.bias is not None
        return load_ok


class ROWBMMWeight(BMMWeight):
    load_hf_weights = ROWMMWeight.load_hf_weights

    def __init__(
        self,
        weight_name,
        data_type,
        split_n_embed,
        bias_name=None,
    ):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)


class ROWBMMWeightNoTp(BMMWeight):
    load_hf_weights = ROWMMWeight.load_hf_weights

    def __init__(
        self,
        weight_name,
        data_type,
        split_n_embed,
        bias_name=None,
    ):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)
        self.start = 0
        self.end = split_n_embed


class COLBMMWeight(BMMWeight):
    load_hf_weights = COLMMWeight.load_hf_weights

    def __init__(
        self,
        weight_name,
        data_type,
        split_n_embed,
        bias_name=None,
    ):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)

    def _post_load_weights(self):
        self.weight = self.weight.transpose(0, 1).cuda(self.device_id_)


class COLBMMWeightNoTp(BMMWeight):
    load_hf_weights = COLMMWeightNoTp.load_hf_weights

    def __init__(
        self,
        weight_name,
        data_type,
        split_n_embed,
        bias_name=None,
    ):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)
        self.start = 0
        self.end = split_n_embed

    def _post_load_weights(self):
        self.weight = self.weight.transpose(0, 1).cuda(self.tp_rank_)
