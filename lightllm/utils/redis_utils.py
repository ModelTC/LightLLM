import subprocess
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def start_redis_service(args):
    """launch redis service"""

    config_server_host = args.config_server_host
    redis_port = args.config_server_visual_redis_port
    try:
        subprocess.run(
            ["redis-cli", "-h", config_server_host, "-p", str(redis_port), "FLUSHALL", "ASYNC"], check=False, timeout=2
        )
        subprocess.run(
            ["redis-cli", "-h", config_server_host, "-p", str(redis_port), "SHUTDOWN", "NOSAVE"], check=False, timeout=2
        )
    except Exception:
        pass

    try:
        redis_command = [
            "redis-server",
            "--port",
            str(redis_port),
            "--bind",
            f"{config_server_host}",
            "--daemonize",
            "no",
            "--logfile",
            "/dev/stdout",
            "--loglevel",
            "notice",
            "--save",
            '""',  # 不触发 RDB 快照
            "--appendonly",
            "no",  # 关闭 AOF
        ]

        logger.info(f"Starting Redis service on port {redis_port}")
        redis_process = subprocess.Popen(redis_command)

        import redis
        import time

        max_wait = 10
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                r = redis.Redis(host=args.config_server_host, port=redis_port, socket_connect_timeout=1)
                r.ping()
                logger.info(f"Redis service started successfully on port {redis_port}")
                del r
                break
            except Exception as e:
                logger.error(f"Error occurred while checking Redis service: {e}")
                time.sleep(0.5)
                if redis_process.poll() is not None:
                    logger.error("Redis service failed to start")
                    return None
        else:
            logger.error("Redis service startup timeout")
            if redis_process.poll() is None:
                redis_process.terminate()
            return None

        return redis_process

    except Exception as e:
        logger.error(f"Failed to start Redis service: {e}")
        return None
