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

from lightllm.utils.model_config import load_model_config_dict
from typing import Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

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

    model_cfg = load_model_config_dict(tokenizer_name, trust_remote_code=trust_remote_code, **kwargs)

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

    model_type = model_cfg.get("model_type", "")
    # DeepSeek-V3.2 custom tokenizer mode: wraps the HF tokenizer with
    # a Python-based apply_chat_template that uses encoding_dsv32.py.
    if model_type == "deepseek_v32":
        from ..models.deepseek3_2.model import DeepSeekV32Tokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote_code, *args, **kwargs
        )
        logger.info("Using DeepSeek-V3.2 tokenizer mode with Python-based chat template encoding.")
        return DeepSeekV32Tokenizer(hf_tokenizer)

    if (model_cfg.get("architectures") or [""])[0] == "TarsierForConditionalGeneration":
        from ..models.tarsier2.model import Tarsier2Tokenizer
        from ..models.qwen2_vl.vision_process import Qwen2VLImageProcessor

        image_processor = Qwen2VLImageProcessor.from_pretrained(tokenizer_name)
        tokenizer = Tarsier2Tokenizer(tokenizer=tokenizer, image_processor=image_processor, model_cfg=model_cfg)
    elif model_type == "llava" or model_type == "internlmxcomposer2":
        from ..models.llava.model import LlavaTokenizer

        tokenizer = LlavaTokenizer(tokenizer, model_cfg)
    elif model_type == "qwen" and "visual" in model_cfg:
        from ..models.qwen_vl.model import QWenVLTokenizer

        tokenizer = QWenVLTokenizer(tokenizer, model_cfg)
    elif model_type in ["qwen2_vl", "qwen2_5_vl"] and "vision_config" in model_cfg:
        from ..models.qwen2_vl.model import QWen2VLTokenizer
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(tokenizer_name)
        tokenizer = QWen2VLTokenizer(
            tokenizer=tokenizer, image_processor=processor.image_processor, model_cfg=model_cfg
        )
    elif model_type in ["qwen3_vl", "qwen3_vl_moe"] and "vision_config" in model_cfg:
        from ..models.qwen3_vl.model import QWen3VLTokenizer
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(tokenizer_name)
        tokenizer = QWen3VLTokenizer(
            tokenizer=tokenizer, image_processor=processor.image_processor, model_cfg=model_cfg
        )
    elif model_type in ["qwen3_5", "qwen3_5_moe"] and "vision_config" in model_cfg:
        from transformers import AutoProcessor
        from ..models.qwen3_5.model import QWen3_5Tokenizer

        processor = AutoProcessor.from_pretrained(tokenizer_name)
        tokenizer = QWen3_5Tokenizer(
            tokenizer=tokenizer, image_processor=processor.image_processor, model_cfg=model_cfg
        )
    elif model_cfg.get("thinker_config") is not None:
        from ..models.qwen3_omni_moe_thinker.model import QWen3OmniTokenizer
        from transformers import AutoProcessor

        model_cfg = model_cfg["thinker_config"]
        processor = AutoProcessor.from_pretrained(tokenizer_name)
        tokenizer = QWen3OmniTokenizer(tokenizer, processor=processor, model_cfg=model_cfg)
    elif model_type == "internvl_chat":
        from ..models.internvl.model import InternvlTokenizer

        tokenizer = InternvlTokenizer(tokenizer, model_cfg, weight_dir=tokenizer_name)
    elif model_type == "gemma3":
        from ..models.gemma3.model import Gemma3Tokenizer

        tokenizer = Gemma3Tokenizer(tokenizer, model_cfg)
    elif model_type == "gemma4":
        from ..models.gemma4.tokenizer import Gemma4Tokenizer

        image_processor = None
        if "vision_config" in model_cfg and model_cfg["vision_config"] is not None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(tokenizer_name)
            image_processor = processor.image_processor
        tokenizer = Gemma4Tokenizer(tokenizer, model_cfg, image_processor=image_processor)

    return tokenizer
