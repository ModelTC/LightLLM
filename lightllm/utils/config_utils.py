import json
import os


def get_config_json(model_path: str):
    with open(os.path.join(model_path, "config.json"), "r") as file:
        json_obj = json.load(file)
    return json_obj


def get_eos_token_ids(model_path: str):
    config_json = get_config_json(model_path)
    eos_token_id = config_json["eos_token_id"]
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    if isinstance(eos_token_id, list):
        return eos_token_id
    assert False, "error eos_token_id format in config.json"
