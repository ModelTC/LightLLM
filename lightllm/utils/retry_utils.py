import time
import functools
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def retry(max_attempts=3, wait_time=1):
    """
    被修饰的函数调用失败需要自己抛异常
    :param max_attempts: 最大重试次数
    :param wait_time: 每次重试之间的等待时间（秒）
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logger.info(f"try {func.__name__} {attempts}/{max_attempts} fail: {str(e)}")
                    if attempts < max_attempts:
                        time.sleep(wait_time)
            raise Exception(f"{func.__name__} try all failed")

        return wrapper

    return decorator
