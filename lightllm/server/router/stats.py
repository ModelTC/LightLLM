import time
from lightllm.utils.log_utils import init_logger
from lightllm.server.core.objs import StartArgs

logger = init_logger(__name__)


class SystemStatusReporter:
    """统计 token 吞吐和 prefix cache 命中情况，并周期性上报到 prometheus 监控指标。"""

    def __init__(self, args: StartArgs, metric_client=None):
        self.enabled = not args.disable_log_stats
        self.interval = max(5, args.log_stats_interval)
        if args.log_stats_interval < 5:
            logger.warning(f"log_stats_interval={args.log_stats_interval}s is below minimum, using 5s")
        self.metric_client = metric_client

        # 窗口期计数器（每个上报周期重置）
        self.last_report_time = time.time()
        self.prompt_tokens = 0
        self.output_tokens = 0

        # 全局计数器（不重置，用于计算全局 cache 命中率）
        self.global_input_total = 0
        self.global_gpu_cache_total = 0

    def count_prompt_tokens(self, num_tokens: int):
        if self.metric_client is not None:
            self.metric_client.counter_inc_by("lightllm_prompt_tokens_total", num_tokens)
        if self.enabled:
            self.prompt_tokens += num_tokens

    def count_output_tokens(self, num_tokens: int):
        if self.metric_client is not None:
            self.metric_client.counter_inc_by("lightllm_generation_tokens_total", num_tokens)
        if self.enabled:
            self.output_tokens += num_tokens

    def on_request_completed(self, input_len: int, gpu_cache_len: int):
        if self.enabled:
            self.global_input_total += input_len
            self.global_gpu_cache_total += gpu_cache_len

    def maybe_report(self, running_batch):
        if not self.enabled:
            return
        now = time.time()
        elapsed = now - self.last_report_time
        if elapsed < self.interval:
            return

        output_tps = self.output_tokens / elapsed
        running = len(running_batch.reqs) if running_batch is not None else 0
        global_gpu_cache_hit_rate = (
            (self.global_gpu_cache_total / self.global_input_total) if self.global_input_total > 0 else 0.0
        )

        if self.metric_client is not None:
            self.metric_client.gauge_set("lightllm_cache_hit_rate", global_gpu_cache_hit_rate)
            self.metric_client.gauge_set("lightllm_gen_throughput", output_tps)
            self.metric_client.gauge_set("lightllm_num_running_reqs", running)

        # 重置窗口期计数器
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.last_report_time = now


class RouterStatics:
    def __init__(self, args: StartArgs):
        self.busy_token_used_ratio = args.router_token_ratio
        self.ema_req_out_len = 2048
        self.cur_ema_params = 0.5
        self.min_ema_params = 0.04

    def update(self, req_out_len: int):
        # 过滤掉输出特别短的情况，防止计算得过于短，导致调度频繁引发暂停，导致系统吞吐下降。
        req_out_len = max(req_out_len, 64)
        self.ema_req_out_len = int(self.ema_req_out_len * (1 - self.cur_ema_params) + req_out_len * self.cur_ema_params)
        self.ema_req_out_len = max(64, self.ema_req_out_len)
        # 不断的调整ema 的计算参数，这样可以在早期，快速将 ema_req_out_len 调整到接近
        # 当前分布的水平，然后后期趋于稳定调整。
        self.cur_ema_params = max(self.min_ema_params, self.cur_ema_params * 0.8)

    def log_str(self) -> str:
        return (
            f"RouterStatics busy_token_used_ratio: {self.busy_token_used_ratio} "
            f"ema_req_out_put_len: {self.ema_req_out_len}"
        )
