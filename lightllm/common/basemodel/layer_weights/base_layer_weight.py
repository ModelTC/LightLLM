import torch
import numpy as np
import threading


class BaseLayerWeight:
    def __init__(self):
        self.tp_rank_ = None
        self.lock = threading.Lock()

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
        if self.tp_rank_ is None:
            return cpu_tensor.contiguous().to(self.data_type_).cuda()
        else:
            return cpu_tensor.contiguous().to(self.data_type_).cuda(self.tp_rank_)

    def _try_cat_to(self, source_tensor_names, dest_name, cat_dim, handle_func=None):
        if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
            with self.lock:
                if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
                    assert all(
                        not getattr(self, name, None).is_cuda for name in source_tensor_names
                    ), "all not cuda tensor"
                    tensors = [getattr(self, name, None) for name in source_tensor_names]
                    ans = torch.cat(tensors, dim=cat_dim)
                    if handle_func is not None:
                        ans = handle_func(ans)
                    else:
                        ans = self._cuda(ans)
                    setattr(self, dest_name, ans)
                    for name in source_tensor_names:
                        delattr(self, name)
        return
