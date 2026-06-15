from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.integrations.ggml import (
    GGUF_TO_FAST_CONVERTERS,
    GGUF_TOKENIZER_MAPPING,
    convert_gguf_tokenizer,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING, tokenizer_class_from_name
from typing import Any, Dict, Optional, Tuple, Union

from lightllm.common.basemodel.layer_weights.gguf_load_utils import LightLLMGGUFReader, get_gguf_reader
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def _build_gguf_tokenizer_field_map() -> Dict[str, Tuple[str, str]]:
    """ Build a mapping from GGUF tokenizer field name to HuggingFace tokenizer field name. """
    field_map: Dict[str, Tuple[str, str]] = {}
    # Build mapping for tokenizer fields
    for gguf_suffix, hf_key in GGUF_TOKENIZER_MAPPING["tokenizer"].items():
        # e.g., "ggml.model" -> ("tokenizer", "model_type")
        field_map[f"tokenizer.{gguf_suffix}"] = ("tokenizer", hf_key)
    # Build mapping for tokenizer_config fields
    for gguf_suffix, hf_key in GGUF_TOKENIZER_MAPPING["tokenizer_config"].items():
        # e.g., "ggml.model" -> ("tokenizer_config", "model_type")
        gguf_key = f"tokenizer.{gguf_suffix}"
        if gguf_key not in field_map:
            field_map[gguf_key] = ("tokenizer_config", hf_key)
    # Add special case for add_bos_token
    field_map["tokenizer.ggml.add_bos_token"] = ("tokenizer_config", "add_bos_token")

    return field_map


def _parse_gguf_tokenizer_from_reader(reader: LightLLMGGUFReader) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Parse tokenizer metadata from a GGUFReader using ReaderField.contents(). """
    tokenizer: Dict[str, Any] = {}
    tokenizer_config: Dict[str, Any] = {}

    tokenizer_field_map = _build_gguf_tokenizer_field_map()
    for gguf_key, (bucket, hf_key) in tokenizer_field_map.items():
        value = reader.read_field(gguf_key)
        if value is None:
            continue
        # Add value to tokenizer or tokenizer_config
        if bucket == "tokenizer":
            tokenizer[hf_key] = value
        else:
            tokenizer_config[hf_key] = value

    if "model_type" not in tokenizer_config and tokenizer.get("tokenizer_type") is not None:
        tokenizer_config["model_type"] = tokenizer["tokenizer_type"]

    for id_key in ("bos_token_id", "eos_token_id", "unk_token_id", "pad_token_id"):
        token_id = tokenizer.get(id_key)
        if token_id is not None:
            tokenizer_config[id_key] = token_id

    return tokenizer, tokenizer_config


def _architecture_to_converter_key(architecture: str) -> str:
    """ Convert GGUF architecture to HuggingFace converter key. """
    arch = architecture.replace("-", "_")
    if arch in GGUF_TO_FAST_CONVERTERS:
        return arch
    # e.g., qwen2moe / qwen3moe -> qwen2_moe / qwen3_moe
    if arch.endswith("moe") and not arch.endswith("_moe"):
        alt = f"{arch[:-3]}_moe"
        if alt in GGUF_TO_FAST_CONVERTERS:
            return alt

    return arch


def _resolve_gguf_tokenizer_architecture(model_config: Dict[str, Any], reader: LightLLMGGUFReader) -> str:
    """ Resolve GGUF tokenizer architecture from GGUF reader or model config. """
    architecture = None
    # Read architecture from GGUF reader
    arch = reader.read_field("general.architecture")
    if isinstance(arch, (list, tuple)):
        arch = arch[0]
    if isinstance(arch, str):
        architecture = arch

    # Read architecture from model config
    if not architecture:
        architecture = model_config.get("model_type")

    if not architecture:
        raise ValueError(
            "Can't resolve GGUF tokenizer architecture: "
            "missing general.architecture in GGUF and model_type in config"
        )

    return _architecture_to_converter_key(architecture)


def _build_tokenizer_init_kwargs(
    tokenizer_dict: Dict[str, Any],
    tokenizer_config: Dict[str, Any],
    additional_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """ Build tokenizer initialization kwargs from tokenizer dictionary and configuration. """
    # Merge tokenizer configuration and additional kwargs
    init_kwargs = {**tokenizer_config, **additional_kwargs}

    tokens = tokenizer_dict.get("tokens") or []
    for id_key, token_key in (
        ("bos_token_id", "bos_token"),
        ("eos_token_id", "eos_token"),
        ("unk_token_id", "unk_token"),
        ("pad_token_id", "pad_token"),
    ):
        token_id = tokenizer_dict.get(id_key)
        if token_id is None:
            token_id = init_kwargs.get(id_key)

        if token_id is not None and token_id < len(tokens):
            init_kwargs.setdefault(token_key, tokens[token_id])

    return init_kwargs


def _get_hf_tokenizer_class_from_config(
    model_config: Dict[str, Any],
    use_fast: bool = True,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """ Resolve the same tokenizer class AutoTokenizer would use for this config. """
    # Get tokenizer class name from tokenizer_class key of model config
    tokenizer_class_name: str = model_config.get("tokenizer_class")
    if tokenizer_class_name:
        candidates = []
        if use_fast and not tokenizer_class_name.endswith("Fast"):
            candidates.append(f"{tokenizer_class_name}Fast")
        candidates.append(tokenizer_class_name)
        for candidate in candidates:
            tokenizer_class = tokenizer_class_from_name(candidate)
            if tokenizer_class is not None:
                return tokenizer_class

    # Get HuggingFace tokenizer class from model_type key of model config
    model_type = model_config.get("model_type")
    # CONFIG_MAPPING: model_type -> config class
    if not model_type or model_type not in CONFIG_MAPPING:
        return PreTrainedTokenizerFast

    config_class = CONFIG_MAPPING[model_type]
    # TOKENIZER_MAPPING: config class -> (tokenizer class, fast tokenizer class)
    if config_class not in TOKENIZER_MAPPING:
        return PreTrainedTokenizerFast

    tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[config_class]
    if use_fast and tokenizer_class_fast is not None:
        return tokenizer_class_fast

    if tokenizer_class_py is not None:
        return tokenizer_class_py

    return PreTrainedTokenizerFast


def load_tokenizer_from_gguf(
    gguf_path: str,
    model_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """ Load the tokenizer from GGUF file. """
    if model_config is None:
        raise ValueError("model_config is required when loading tokenizer from GGUF")

    reader = get_gguf_reader(gguf_path)
    tokenizer_dict, tokenizer_config = _parse_gguf_tokenizer_from_reader(reader)
    architecture = _resolve_gguf_tokenizer_architecture(model_config, reader=reader)

    if "tokens" not in tokenizer_dict:
        raise ValueError(f"GGUF file does not contain tokenizer.ggml.tokens: {gguf_path}")
        
    if architecture not in GGUF_TO_FAST_CONVERTERS:
        supported = ", ".join(sorted(GGUF_TO_FAST_CONVERTERS.keys()))
        raise ValueError(
            f"unsupported GGUF tokenizer architecture {architecture!r} for {gguf_path}; "
            f"supported: {supported}"
        )

    logger.info(
        f"loading tokenizer from GGUF ReaderField metadata: {gguf_path} "
        f"(architecture={architecture}, vocab_size={len(tokenizer_dict['tokens'])})"
    )

    # Convert GGUF tokenizer to HuggingFace tokenizer
    fast_tokenizer, additional_kwargs = convert_gguf_tokenizer(architecture, tokenizer_dict)

    init_kwargs = _build_tokenizer_init_kwargs(tokenizer_dict, tokenizer_config, additional_kwargs)

    tokenizer_kwargs = {k: v for k, v in kwargs.items() if k != "trust_remote_code"}
    init_kwargs.update(tokenizer_kwargs)

    use_fast = kwargs.pop("use_fast", True)
    # Get HuggingFace tokenizer class from model config
    tokenizer_class = _get_hf_tokenizer_class_from_config(model_config, use_fast=use_fast)

    logger.info(f"GGUF tokenizer class: {tokenizer_class.__name__}")

    # Initialize HuggingFace tokenizer
    return tokenizer_class(tokenizer_object=fast_tokenizer, **init_kwargs)
