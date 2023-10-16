import os
import sys
import unittest
from model_infer_multimodal import test_multimodal_inference
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))))


class TestLlamaMultiModalInfer(unittest.TestCase):

    def test_llama_infer(self):
        from lightllm.models.llama_multimodal.model import LlamaTpPartMulitModal
        test_multimodal_inference(world_size=4,
                                  model_dir="/path/to/llama-7b",
                                  model_class=LlamaTpPartMulitModal,
                                  batch_size=10,
                                  input_len=1024,
                                  output_len=1024,
                                  # (pad_len, pad_dim_size, offset)
                                  repad_embeds_args=(36, 4096, 5))
        return


if __name__ == '__main__':
    unittest.main()
