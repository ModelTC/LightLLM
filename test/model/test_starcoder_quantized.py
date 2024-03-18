import os
import sys
import unittest
from model_infer import test_model_inference
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestStarcoderInfer(unittest.TestCase):
    def test_starcoder_infer(self):
        from lightllm.models.starcoder_wquant.model import StarcoderTpPartModelWQuant

        test_model_inference(
            world_size=1,
            model_dir="/path/xxxx",
            model_class=StarcoderTpPartModelWQuant,
            batch_size=2,
            input_len=10,
            output_len=10,
            mode=["triton_int8weight"],
        )
        return


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    unittest.main()
