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
from lightllm.utils.config_utils import ModelPaths, get_model_config
from lightllm.utils.gguf_tokenizer_utils import load_tokenizer_from_gguf
from lightllm.utils.log_utils import init_logger
from ..models.tarsier2.model import Tarsier2Tokenizer

logger = init_logger(__name__)
from ..models.llava.model import LlavaTokenizer
from ..models.qwen_vl.model import QWenVLTokenizer
from ..models.qwen2_vl.model import QWen2VLTokenizer
from ..models.qwen3_vl.model import QWen3VLTokenizer
from ..models.internvl.model import InternvlTokenizer
from ..models.gemma3.model import Gemma3Tokenizer
from ..models.qwen3_omni_moe_thinker.model import QWen3OmniTokenizer

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def _load_hf_tokenizer(
    path: str,
    trust_remote_code: bool,
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, *args, **kwargs)
    except TypeError as e:
        logger.warning(f"load fast tokenizer fail: {str(e)}")
        kwargs["use_fast"] = False
        return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, *args, **kwargs)


def _load_base_tokenizer(
    load_path: str,
    from_gguf: bool,
    model_cfg: dict,
    trust_remote_code: bool,
    tokenizer_mode: str,
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """ Load the base tokenizer based on AutoTokenizer. """
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    
    if from_gguf:
        logger.info(f"Loading tokenizer from GGUF file: {load_path}")
        return load_tokenizer_from_gguf(load_path, model_cfg, *args, **kwargs)

    if "llama" in load_path.lower() and kwargs.get("use_fast", True):
        logger.info(
            "For some LLaMA-based models, initializing the fast tokenizer may "
            "take a long time. To eliminate the initialization time, consider "
            f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer."
        )
    
    tokenizer = _load_hf_tokenizer(load_path, trust_remote_code, *args, **kwargs)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.info(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    return tokenizer


def _wrap_tokenizer(
    base_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    model_cfg: dict,
    processor_dir: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    model_type = model_cfg.get("model_type", "")
    # DeepSeek-V3.2 custom tokenizer mode: wraps the HF tokenizer with
    # a Python-based apply_chat_template that uses encoding_dsv32.py.
    if model_type == "deepseek_v32":
        from ..models.deepseek3_2.model import DeepSeekV32Tokenizer

        logger.info("Using DeepSeek-V3.2 tokenizer mode with Python-based chat template encoding.")
        return DeepSeekV32Tokenizer(base_tokenizer)

    if model_cfg["architectures"][0] == "TarsierForConditionalGeneration":
        from ..models.qwen2_vl.vision_process import Qwen2VLImageProcessor, load_image_processor

        image_processor = load_image_processor(processor_dir, Qwen2VLImageProcessor)
        return Tarsier2Tokenizer(tokenizer=base_tokenizer, image_processor=image_processor, model_cfg=model_cfg)

    if model_type == "llava" or model_type == "internlmxcomposer2":
        return LlavaTokenizer(base_tokenizer, model_cfg)

    if model_type == "qwen" and "visual" in model_cfg:
        return QWenVLTokenizer(base_tokenizer, model_cfg)

    if model_type in ["qwen2_vl", "qwen2_5_vl"] and "vision_config" in model_cfg:
        from ..models.qwen2_vl.vision_process import Qwen2VLImageProcessor, load_image_processor

        image_processor = load_image_processor(processor_dir, Qwen2VLImageProcessor)
        return QWen2VLTokenizer(
            tokenizer=base_tokenizer, image_processor=image_processor, model_cfg=model_cfg
        )

    if model_type in ["qwen3_vl", "qwen3_vl_moe"] and "vision_config" in model_cfg:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(processor_dir)
        return QWen3VLTokenizer(
            tokenizer=base_tokenizer, image_processor=processor.image_processor, model_cfg=model_cfg
        )

    if model_type in ["qwen3_5", "qwen3_5_moe"] and "vision_config" in model_cfg:
        from transformers import AutoProcessor
        from ..models.qwen3_5.model import QWen3_5Tokenizer

        processor = AutoProcessor.from_pretrained(processor_dir)
        return QWen3_5Tokenizer(
            tokenizer=base_tokenizer, image_processor=processor.image_processor, model_cfg=model_cfg
        )

    if model_cfg.get("thinker_config") is not None:
        from transformers import AutoProcessor

        model_cfg = model_cfg["thinker_config"]
        processor = AutoProcessor.from_pretrained(processor_dir)
        return QWen3OmniTokenizer(base_tokenizer, processor=processor, model_cfg=model_cfg)

    if model_type == "internvl_chat":
        return InternvlTokenizer(base_tokenizer, model_cfg, weight_dir=processor_dir)

    if model_type == "gemma3":
        return Gemma3Tokenizer(base_tokenizer, model_cfg)

    return base_tokenizer


def get_tokenizer(
    paths: ModelPaths,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Load base tokenizer (HF or GGUF), then wrap for model-specific behavior if needed."""
    model_cfg = get_model_config(paths)
    load_path, from_gguf = paths.tokenizer_load_path

    base_tokenizer = _load_base_tokenizer(
        load_path=load_path,
        from_gguf=from_gguf,
        model_cfg=model_cfg,
        trust_remote_code=trust_remote_code,
        tokenizer_mode=tokenizer_mode,
        *args,
        **kwargs,
    )

    return _wrap_tokenizer(
        base_tokenizer=base_tokenizer,
        model_cfg=model_cfg,
        processor_dir=paths.processor_dir,
    )
