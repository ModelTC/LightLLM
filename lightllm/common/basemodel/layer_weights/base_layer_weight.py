import torch
import numpy as np


class BaseLayerWeight:
    def __init__(self):
        pass

    def load_hf_weights(self, weights):
        """
        load weights
        """
        pass


    def init_static_params(self):
        """
        design for some static init params, many model dont need do this.
        """
        pass

    def verify_load(self):
        """
        verify all load is ok
        """
        raise Exception("must verify weights load ok")
        pass

    def _cuda(self, cpu_tensor):
        return cpu_tensor.contiguous().to(self.data_type_).cuda()
