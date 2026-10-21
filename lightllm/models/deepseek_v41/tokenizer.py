import importlib.util
import os

from lightllm.models.deepseek_v4.tokenizer import DeepSeekV4Tokenizer


class DeepSeekV41Tokenizer(DeepSeekV4Tokenizer):
    supported_reasoning_efforts = ("low", "high", "max", None, *range(1, 101))

    def _get_encoding_module(self):
        if self._encoding_module is None:
            path = os.path.join(self.model_dir, "encoding", "encoding.py")
            spec = importlib.util.spec_from_file_location("lightllm_deepseek_v41_encoding", path)
            self._encoding_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._encoding_module)
        return self._encoding_module
