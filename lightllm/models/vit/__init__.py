import os
import importlib.util
from lightllm.utils.log_utils import init_logger
from lightllm.models.internvl.img_process import load_image as default_load_image
from lightllm.models.internvl.img_process import get_image_patch as default_get_image_patch

logger = init_logger(__name__)


def get_load_image_func(weight_dir):
    global load_image
    pre_process_path = os.path.join(weight_dir, "pre_process.py")
    if os.path.exists(pre_process_path):
        logger.info(f"Found pre_process.py in {weight_dir}, attempting to load load_image from it.")
        spec = importlib.util.spec_from_file_location("pre_process", pre_process_path)
        pre_process = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pre_process)
        if hasattr(pre_process, "load_image"):
            logger.info("load_image function replaced by the one in pre_process.py.")
            return pre_process.load_image
        else:
            logger.info("load_image function not found in pre_process.py.")
    else:
        logger.info(f"pre_process.py not found in {weight_dir}, using default load_image.")

    return default_load_image


def get_image_patch_func(weight_dir):
    global load_image
    pre_process_path = os.path.join(weight_dir, "pre_process.py")
    if os.path.exists(pre_process_path):
        logger.info(f"Found pre_process.py in {weight_dir}, attempting to load get_image_patch from it.")
        spec = importlib.util.spec_from_file_location("pre_process", pre_process_path)
        pre_process = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pre_process)
        if hasattr(pre_process, "get_image_patch"):
            logger.info("load_image function replaced by the one in pre_process.py.")
            return pre_process.get_image_patch
        else:
            logger.info("get_image_patch function not found in pre_process.py.")
    else:
        logger.info(f"pre_process.py not found in {weight_dir}, using default get_image_patch.")

    return default_get_image_patch
