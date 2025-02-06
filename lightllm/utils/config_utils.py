import json
import os
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_config_json(model_path: str):
    with open(os.path.join(model_path, "config.json"), "r") as file:
        json_obj = json.load(file)
    return json_obj


def get_eos_token_ids(model_path: str):
    config_json = get_config_json(model_path)
    try:
        eos_token_id = config_json["eos_token_id"]
    except:
        # for some multimode model.
        eos_token_id = config_json["llm_config"]["eos_token_id"]

    if isinstance(eos_token_id, int):
        return [eos_token_id]
    if isinstance(eos_token_id, list):
        return eos_token_id
    assert False, "error eos_token_id format in config.json"


def get_vocab_size(model_path: str):
    try:
        config_json = get_config_json(model_path)
        vocab_size = config_json["vocab_size"]
        if not isinstance(vocab_size, int):
            vocab_size = int(vocab_size)
        return vocab_size
    except:
        logger.error("can not get vocab_size from config.json, return 0")
        return 0


def get_dtype(model_path: str):
    config_json = get_config_json(model_path)
    try:
        torch_dtype = config_json["torch_dtype"]
        return torch_dtype
    except:
        logger.warning("torch_dtype not in config.json, use float16 as default")
        return "float16"
