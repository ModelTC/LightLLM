import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from functools import lru_cache
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import get_current_device_name

logger = init_logger(__name__)


class KernelConfigs(ABC):

    kernel_name: str = "unknown_kernel"

    @classmethod
    def get_config_file_name(cls, params: Dict[str, Any]) -> str:
        json_str = json.dumps(params, sort_keys=True)
        json_str = json_str.replace(" ", "").replace("\n", "").replace('"', "").replace(":", "=")
        filename = json_str
        device_name = get_current_device_name().replace(" ", "_")
        return f"{filename}_{device_name}.json"

    @classmethod
    @lru_cache(maxsize=None)
    def get_the_config(cls, params: Dict[str, Any]) -> Optional[dict]:
        assert cls != KernelConfigs, "base class can not call this classmethod"

        json_file_name = KernelConfigs.get_config_file_name(params)
        config_dir_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "all_kernel_configs", cls.kernel_name
        )

        config_file_path = os.path.join(config_dir_path, json_file_name)

        if os.path.exists(config_file_path):
            with open(config_file_path, mode="r") as file:
                return json.load(file)
        else:
            logger.warning(
                f"can not find config_path {config_file_path} kernel name {cls.kernel_name} use default kernel setting"
            )
            return None

    @classmethod
    def store_config(cls, params: Dict[str, Any], dest_json: dict):
        assert cls != KernelConfigs, "base class can not call this classmethod"

        json_file_name = KernelConfigs.get_config_file_name(params)
        config_dir_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "all_kernel_configs", cls.kernel_name
        )
        os.makedirs(config_dir_path, exist_ok=True)
        config_file_path = os.path.join(config_dir_path, json_file_name)
        with open(config_file_path, mode="w") as file:
            json.dump(dest_json, file)
        return

    @classmethod
    @abstractmethod
    def try_to_get_best_config(cls, *args, **kwargs) -> dict:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def save_config(cls, *args, **kwargs) -> None:
        raise NotImplementedError()
