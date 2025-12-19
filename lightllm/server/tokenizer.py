# Adapted from vllm/transformers_utils/tokenizer.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers.configuration_utils import PretrainedConfig
from lightllm.utils.log_utils import init_logger
from ..models.tarsier2.model import Tarsier2Tokenizer

logger = init_logger(__name__)
from ..models.llava.model import LlavaTokenizer
from ..models.qwen_vl.model import QWenVLTokenizer
from ..models.qwen2_vl.model import QWen2VLTokenizer
from ..models.qwen3_vl.model import QWen3VLTokenizer
from ..models.internvl.model import InternvlTokenizer
from ..models.gemma3.model import Gemma3Tokenizer

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "llama" in tokenizer_name.lower() and kwargs.get("use_fast", True):
        logger.info(
            "For some LLaMA-based models, initializing the fast tokenizer may "
            "take a long time. To eliminate the initialization time, consider "
            f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer."
        )
        # tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        # tokenizer = convert_slow_tokenizer(tokenizer)
        # return tokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code, *args, **kwargs)
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments, using slow mode.
        # you can try pip install protobuf==3.20.0 to try repair
        logger.warning(f"load fast tokenizer fail: {str(e)}")
        kwargs["use_fast"] = False
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code, *args, **kwargs)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.info(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    model_cfg, _ = PretrainedConfig.get_config_dict(tokenizer_name)
    model_type = model_cfg.get("model_type", "")
    if model_cfg["architectures"][0] == "TarsierForConditionalGeneration":
        from ..models.qwen2_vl.vision_process import Qwen2VLImageProcessor

        image_processor = Qwen2VLImageProcessor.from_pretrained(tokenizer_name)
        tokenizer = Tarsier2Tokenizer(tokenizer=tokenizer, image_processor=image_processor, model_cfg=model_cfg)
    elif model_type == "llava" or model_type == "internlmxcomposer2":
        tokenizer = LlavaTokenizer(tokenizer, model_cfg)
    elif model_type == "qwen" and "visual" in model_cfg:
        tokenizer = QWenVLTokenizer(tokenizer, model_cfg)
    elif model_type in ["qwen2_vl", "qwen2_5_vl"] and "vision_config" in model_cfg:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(tokenizer_name)
        tokenizer = QWen2VLTokenizer(
            tokenizer=tokenizer, image_processor=processor.image_processor, model_cfg=model_cfg
        )
    elif model_type in ["qwen3_vl", "qwen3_vl_moe"] and "vision_config" in model_cfg:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(tokenizer_name)
        tokenizer = QWen3VLTokenizer(
            tokenizer=tokenizer, image_processor=processor.image_processor, model_cfg=model_cfg
        )
    elif model_type == "internvl_chat":
        tokenizer = InternvlTokenizer(tokenizer, model_cfg, weight_dir=tokenizer_name)
    elif model_type == "gemma3":
        tokenizer = Gemma3Tokenizer(tokenizer, model_cfg)

    return tokenizer
