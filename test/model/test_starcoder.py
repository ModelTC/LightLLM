import os
import sys
import unittest
from model_infer import test_model_inference
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))))


class TestStarcoderInfer(unittest.TestCase):

    def test_starcoder_infer(self):
        from lightllm.models.starcoder.model import StarcoderTpPartModel
        test_model_inference(world_size=1,
                             model_dir="/nvme/baishihao/yuewen_0720_15b5/",
                             model_class=StarcoderTpPartModel,
                             batch_size=20,
                             input_len=1024,
                             output_len=1024)
        return


if __name__ == '__main__':
    unittest.main()
