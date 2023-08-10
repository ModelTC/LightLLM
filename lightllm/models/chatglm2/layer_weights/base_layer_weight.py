import torch
import numpy as np


class BaseLayerWeight:
    def __init__(self):
        self.data_type_ = "fp32"

    def load_to_torch(self, path):
        numpy_type = {"fp32": np.float32, "fp16": np.float16}[self.data_type_]
        torch_type = {"fp32": torch.float32, "fp16": torch.float16}[self.data_type_]
        return torch.from_numpy(np.fromfile(path, dtype=numpy_type)).to(torch_type)
