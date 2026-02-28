import json
import os
from typing import Optional, List
from functools import lru_cache
from .envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_config_json(model_path: str):
    with open(os.path.join(model_path, "config.json"), "r") as file:
        json_obj = json.load(file)
    return json_obj


def _get_config_llm_keyvalue(model_path: str, key_name: list[str]):
    config_json = get_config_json(model_path)
    for key in key_name:
        try:
            value = config_json[key]
        except:
            # for some multimodal model
            try:
                value = config_json["llm_config"][key]
            except:
                value = config_json.get("text_config", {}).get(key)
        if config_json.get("thinker_config") is not None:
            value = config_json.get("thinker_config", {}).get("text_config").get(key)
        if value is not None:
            return value

    logger.error(f"cannot get {key_name} from config.json, return None")

    return None


def get_hidden_size(model_path: str) -> Optional[int]:
    hidden_size = _get_config_llm_keyvalue(model_path=model_path, key_name=["hidden_size", "n_embd", "n_embed"])
    if isinstance(hidden_size, int):
        return hidden_size
    return None


@lru_cache(maxsize=None)
def get_num_key_value_heads(model_path: str) -> int:
    num_key_value_heads = _get_config_llm_keyvalue(model_path=model_path, key_name=["num_key_value_heads"])
    if isinstance(num_key_value_heads, int):
        return num_key_value_heads
    return None


@lru_cache(maxsize=None)
def get_num_attention_heads(model_path: str) -> int:
    num_attention_heads = _get_config_llm_keyvalue(model_path=model_path, key_name=["num_attention_heads"])
    if isinstance(num_attention_heads, int):
        return num_attention_heads
    return None


@lru_cache(maxsize=None)
def get_head_dim(model_path: str) -> int:
    head_dim = _get_config_llm_keyvalue(model_path=model_path, key_name=["head_dim"])
    if isinstance(head_dim, int):
        return head_dim

    # calcu head_dim
    head_dim = get_hidden_size(model_path=model_path) // get_num_attention_heads(model_path=model_path)

    return head_dim


@lru_cache(maxsize=None)
def get_layer_num(model_path: str) -> int:
    num_hidden_layers = _get_config_llm_keyvalue(model_path=model_path, key_name=["num_hidden_layers"])
    if isinstance(num_hidden_layers, int):
        return num_hidden_layers
    return None


def get_eos_token_ids(model_path: str) -> Optional[List[int]]:
    try:
        # qwen3-omini special eos_token_id
        config_json = get_config_json(model_path)
        assert config_json["architectures"][0] == "Qwen3OmniMoeForConditionalGeneration"
        return [151645]
    except:
        pass

    # Qwen3.5 checkpoints can have an eos_token_id in config that differs from
    # tokenizer.eos_token_id. In practice tokenizer.eos_token_id is the reliable
    # stop id (<|im_end|>) for detokenization/stop behavior.
    try:
        config_json = get_config_json(model_path)
        model_type = config_json.get("model_type") or config_json.get("text_config", {}).get("model_type")
        if model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
            if tokenizer.eos_token_id is not None:
                return [int(tokenizer.eos_token_id)]
    except Exception:
        # Fall back to config-based lookup below.
        pass

    eos_token_id = _get_config_llm_keyvalue(model_path=model_path, key_name=["eos_token_id"])
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    if isinstance(eos_token_id, list):
        return eos_token_id

    assert False, "error eos_token_id format in config.json"
    return


def get_model_architectures(model_path: str):
    try:
        config_json = get_config_json(model_path)
        arch = config_json["architectures"][0]
        return arch
    except:
        logger.error("can not get architectures from config.json, return unknown_architecture")
        return "unknown_architecture"


def get_vocab_size(model_path: str):
    try:
        config_json = get_config_json(model_path)
        # qwen3-omini special
        if "thinker_config" in config_json:
            config_json = config_json["thinker_config"]
        if "llm_config" in config_json:
            vocab_size = int(config_json["llm_config"]["vocab_size"])
            return vocab_size
        elif "text_config" in config_json:
            vocab_size = int(config_json["text_config"]["vocab_size"])
            return vocab_size
        vocab_size = config_json["vocab_size"]
        if not isinstance(vocab_size, int):
            vocab_size = int(vocab_size)
        return vocab_size
    except:
        logger.error("can not get vocab_size from config.json, return 0")
        return 0


def get_dtype(model_path: str):
    torch_dtype = _get_config_llm_keyvalue(model_path=model_path, key_name=["torch_dtype", "dtype", "model_dtype"])
    if torch_dtype is None:
        logger.warning("torch_dtype not in config.json, use float16 as default")
        return "float16"
    else:
        return torch_dtype


@lru_cache(maxsize=None)
def get_fixed_kv_len():
    start_args = get_env_start_args()
    model_cfg = get_config_json(start_args.model_dir)
    if "prompt_cache_token_ids" in model_cfg:
        return len(model_cfg["prompt_cache_token_ids"])
    else:
        return 0
