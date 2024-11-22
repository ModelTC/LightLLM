from .base_weight import BaseWeightTpl


class NormWeight(BaseWeightTpl):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__()
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.data_type_ = data_type
        self.weight = None
        self.bias = None

    def load_hf_weights(self, weights):
        if self.weight_name in weights:
            self.weight = weights[self.weight_name].to(self.data_type_).cuda(self.tp_rank_)
        if self.bias_name in weights:
            self.bias = weights[self.bias_name].to(self.data_type_).cuda(self.tp_rank_)

    def verify_load(self):
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.bias_name is not None:
            load_ok = load_ok and self.bias is not None
        return load_ok


class GEMMANormWeight(NormWeight):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)

    def load_hf_weights(self, weights):
        if self.weight_name in weights:
            self.weight = (weights[self.weight_name] + 1).to(self.data_type_).cuda(self.tp_rank_)


class TpNormWeight(NormWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)
        self.split_n_embed = split_n_embed

    def load_hf_weights(self, weights):
        start = self.offset + self.split_n_embed * self.tp_rank_
        end = self.offset + self.split_n_embed * (self.tp_rank_ + 1)

        if self.weight_name in weights:
            self.weight = weights[self.weight_name][start:end].to(self.data_type_).cuda(self.tp_rank_)
        if self.bias_name in weights:
            self.bias = weights[self.bias_name][start:end].to(self.data_type_).cuda(self.tp_rank_)
