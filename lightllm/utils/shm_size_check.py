import ctypes
import os
import shutil
import time
import threading
from lightllm.server.core.objs.req import ChunkedPrefillReq, TokenHealingReq
from lightllm.server.multimodal_params import ImageItem
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.config_utils import get_hidden_size
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def check_recommended_shm_size(args):
    try:
        shm_size, recommended_shm_size, is_shm_sufficient = _check_shm_size(args)
        if not is_shm_sufficient:
            _start_shm_size_warning_thread(shm_size, recommended_shm_size)
        else:
            logger.info(
                f"SHM check: Available={shm_size:.2f} GB,"
                f"Recommended={recommended_shm_size:.2f} GB."
                f"Sufficient: {is_shm_sufficient}",
            )
    except BaseException as e:
        logger.error(f"check_recommended_shm_size error: {str(e)}")


def _check_shm_size(args):
    RED = "\033[91m"
    ENDC = "\033[0m"
    shm_size = _get_system_shm_size_gb()
    required_size = _get_recommended_shm_size_gb(args)
    if shm_size < required_size:
        logger.warning(f"{RED}Available shm size {shm_size:.2f}G is less than required_size {required_size:.2f}G{ENDC}")
        return shm_size, required_size, False
    else:  # shm_size >= required_size
        return shm_size, required_size, True


def _start_shm_size_warning_thread(shm_size, required_shm_size):
    def _periodic_shm_warning(shm_size, required_shm_size):
        RED = "\033[91m"
        ENDC = "\033[0m"
        while True:
            logger.warning(
                f"{RED}Insufficient shared memory (SHM) available."
                f"Required: {required_shm_size:.2f}G, Available: {shm_size:.2f}G.\n"
                "If running in Docker, you can increase SHM size with the `--shm-size` flag, "
                f"like so: `docker run --shm-size=30g [your_image]`{ENDC}",
            )
            time.sleep(120)  # 每 120 秒打印一次警告日志

    shm_warning_thread = threading.Thread(
        target=_periodic_shm_warning,
        args=(
            shm_size,
            required_shm_size,
        ),
        daemon=True,
    )
    shm_warning_thread.start()


def _get_system_shm_size_gb():
    """
    获取 /dev/shm 的总大小(以GB为单位)。
    """
    try:
        shm_path = "/dev/shm"
        if not os.path.exists(shm_path):
            logger.error(f"{shm_path} not exist, this may indicate a system or Docker configuration anomaly.")
            return 0

        # shutil.disk_usage 返回 (total, used, free)
        total_bytes = shutil.disk_usage(shm_path).total
        total_gb = total_bytes / (1024 ** 3)
        return total_gb
    except Exception as e:
        logger.error(f"Error getting /dev/shm size: {e}")
        return 0


def _get_recommended_shm_size_gb(args, max_image_resolution=(3940, 2160), dtype_size=2):
    """
    获取所需的 /dev/shm 大小(以GB为单位)。
    """
    tokenizer = get_tokenizer(args.model_dir, trust_remote_code=True)

    # 估算input_token和logprob占用shm大小，由于是double和int64，所以固定占用8个字节
    input_token_logprob_size_bytes = args.running_max_req_size * 8 * 2 * args.max_req_total_len

    # 估算Req所需的shm大小
    if args.token_healing_mode:
        req_class_size = ctypes.sizeof(TokenHealingReq)
    else:
        req_class_size = ctypes.sizeof(ChunkedPrefillReq)
    req_shm_size_bytes = req_class_size * args.running_max_req_size

    if not args.enable_multimodal:
        total_recommended_shm_size_gb = (req_shm_size_bytes + input_token_logprob_size_bytes) / (1024 ** 3) + 2
    else:
        # 存储图片数据所需的shm大小
        num_channels = 3
        image_width, image_height = max_image_resolution
        image_size_bytes = image_width * image_height * num_channels

        # 假设加载最大分辨率图片时，通过 tokenizer 得到最多的 image_tokens
        if not hasattr(tokenizer, "get_image_token_length"):
            logger.error("Tokenizer must have a 'get_image_token_length' method for multimodal models.")
            return float("inf")

        fake_image_item = ImageItem(
            type="image_size",
            data=max_image_resolution,
        )
        fake_image_item.image_w = fake_image_item._data[0]
        fake_image_item.image_h = fake_image_item._data[1]
        # for internvl model shm check
        fake_image_item.extra_params["image_patch_max_num"] = 12
        max_image_tokens = tokenizer.get_image_token_length(fake_image_item)

        # 估算图片 token 所需的资源
        hidden_size = get_hidden_size(args.model_dir)
        if hidden_size is None:
            logger.warning(
                "Model config not contain 'hidden_size', " "using 4096 by default to calculate recommended shm size."
            )
            image_token_size_bytes = max_image_tokens * 4096 * dtype_size
        else:
            image_token_size_bytes = max_image_tokens * hidden_size * dtype_size

        total_recommended_shm_size_gb = (
            args.cache_capacity * (image_size_bytes + image_token_size_bytes)
            + req_shm_size_bytes
            + input_token_logprob_size_bytes
        )

        total_recommended_shm_size_gb = total_recommended_shm_size_gb / (1024 ** 3) + 2

    return total_recommended_shm_size_gb
