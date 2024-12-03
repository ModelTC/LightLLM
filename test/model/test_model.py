import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
from model_infer import test_model_inference
from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder2.model import Starcoder2TpPartModel
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.chatglm2.model import ChatGlm2TpPartModel
from lightllm.models.internlm.model import InternlmTpPartModel
from lightllm.models.stablelm.model import StablelmTpPartModel
from lightllm.models.internlm2.model import Internlm2TpPartModel
from lightllm.models.mistral.model import MistralTpPartModel
from lightllm.models.minicpm.model import MiniCPMTpPartModel
from lightllm.models.gemma_2b.model import Gemma_2bTpPartModel
from lightllm.models.phi3.model import Phi3TpPartModel
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.cohere.model import CohereTpPartModel
from lightllm.models.mixtral.model import MixtralTpPartModel
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.utils.config_utils import get_dtype
from lightllm.utils.config_utils import get_config_json

def get_model(weight_dir):
    model_cfg = get_config_json(weight_dir)
    model_type = model_cfg["model_type"]
    if model_type == "bloom":
        model_cls = BloomTpPartModel
    elif model_type == "llama":
        model_cls = LlamaTpPartModel
    elif model_type == "qwen":
        model_cls = QWenTpPartModel
    elif model_type == "gpt_bigcode":
        model_cls = StarcoderTpPartModel
    elif model_type == "starcoder2":
        model_cls = Starcoder2TpPartModel
    elif model_type == "chatglm":
        model_cls = ChatGlm2TpPartModel
    elif model_type == "internlm":
        model_cls = InternlmTpPartModel
    elif model_type == "internlm2":
        model_cls = Internlm2TpPartModel
    elif model_type == "mistral":
        model_cls = MistralTpPartModel
    elif model_type == "stablelm":
        model_cls = StablelmTpPartModel
    elif model_type == "mixtral":
        model_cls = MixtralTpPartModel
    elif model_type == "minicpm" or model_cfg["architectures"][0] == "MiniCPMForCausalLM":
        model_cls = MiniCPMTpPartModel
    elif model_type == "qwen2":
        model_cls = Qwen2TpPartModel
    elif model_type == "gemma":
        model_cls = Gemma_2bTpPartModel
    elif model_type == "cohere":
        model_cls = CohereTpPartModel
    elif model_type == "phi3":
        model_cls = Phi3TpPartModel
    elif model_type == "deepseek_v2":
        model_cls = Deepseek2TpPartModel

    return model_cls


class TestModelInfer(unittest.TestCase):
    def test_model_infer(self):
        model_dir = "/nvme/ci_performance/models/DeepSeek-V2-Lite-Chat/"
        model_class = get_model(model_dir)
        data_type = get_dtype(model_dir)
        mode = "triton_gqa_flashdecoding"
        world_size = 1
        batch_size = 1
        input_len = 1024
        output_len = 128
        disable_cudagraph = False
        graph_max_batch_size = batch_size
        graph_max_len_in_batch = 2048
        quant_type = None
        quant_cfg = None
        extra_model_kvargs = {
            "weight_dir": model_dir,
            "mode": mode,
            "data_type": data_type,
            "disable_cudagraph": disable_cudagraph,
            "graph_max_batch_size": graph_max_batch_size,
            "graph_max_len_in_batch": graph_max_len_in_batch,
            "quant_type": quant_type,
            "quant_cfg": quant_cfg,  # path for mixed quantization config.
        }

        test_model_inference(
            world_size=world_size,
            model_class=model_class,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            extra_model_kvargs=extra_model_kvargs,
        )
        return


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    unittest.main()
