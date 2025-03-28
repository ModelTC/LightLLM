import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
from model_infer import test_model_inference
from lightllm.server.api_cli import make_argument_parser
from lightllm.utils.envs_utils import set_env_start_args, get_env_start_args
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
    elif model_type in ["deepseek_v2", "deepseek_v3"]:
        model_cls = Deepseek2TpPartModel

    return model_cls


class TestModelInfer(unittest.TestCase):
    def test_model_infer(self):
        args = get_env_start_args()
        model_dir = args.model_dir
        model_class = get_model(model_dir)
        test_model_inference(args, model_class)
        return


if __name__ == "__main__":
    import torch

    parser = make_argument_parser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--input_len", type=int, default=64, help="input sequence length")
    parser.add_argument("--output_len", type=int, default=128, help="output sequence length")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    )
    args = parser.parse_args()
    set_env_start_args(args)
    torch.multiprocessing.set_start_method("spawn")
    unittest.main(argv=["first-arg-is-ignored"])
