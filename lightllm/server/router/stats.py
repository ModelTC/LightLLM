from lightllm.utils.log_utils import init_logger
from lightllm.server.core.objs import StartArgs

logger = init_logger(__name__)


class RouterStatics:
    def __init__(self, args: StartArgs):
        self.busy_token_used_ratio = args.router_token_ratio
        self.ema_req_out_len = 2048
        self.ema_params = 0.04

    def update(self, req_out_len: int):
        # 过滤掉输出特别短的情况，防止计算得过于短，导致调度频繁引发暂停，导致系统吞吐下降。
        req_out_len = max(req_out_len, 64)
        self.ema_req_out_len = int(self.ema_req_out_len * (1 - self.ema_params) + req_out_len * self.ema_params)
        self.ema_req_out_len = max(64, self.ema_req_out_len)

    def log_str(self) -> str:
        return (
            f"RouterStatics busy_token_used_ratio: {self.busy_token_used_ratio} "
            f"ema_req_out_put_len: {self.ema_req_out_len}"
        )
