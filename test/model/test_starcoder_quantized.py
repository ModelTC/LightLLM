import os
import sys
import unittest
from model_infer import test_model_inference
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestStarcoderInfer(unittest.TestCase):

    def test_starcoder_infer(self):
        from lightllm.models.starcoder_quantized.model import StarcoderTpPartModelQuantized
        test_model_inference(world_size=1,
                             model_dir="/data/wanzihao/092001_3p/",
                             model_class=partial(StarcoderTpPartModelQuantized, mode=['int8weight']),
                             batch_size=2,
                             input_len=10,
                             output_len=10)
        return


if __name__ == '__main__':
    unittest.main()
