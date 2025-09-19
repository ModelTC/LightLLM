from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class ServerBusyError(Exception):
    """Custom exception for server busy/overload situations"""

    def __init__(self, message="Server is busy, please try again later", status_code=503):
        """
        Initialize the ServerBusyError

        Args:
            message (str): Error message to display
            status_code (int): HTTP status code (default 503 Service Unavailable)
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code  # HTTP 503 Service Unavailable

    def __str__(self):
        """String representation of the error"""
        return f"{self.message} (Status code: {self.status_code})"


def log_exception(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except BaseException as e:
            logger.exception(str(e))
            raise e
        return result

    return wrapper
