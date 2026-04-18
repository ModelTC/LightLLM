import time
import logging
from lightllm.server.core.objs import StartArgs
from lightllm.utils.log_utils import init_system_status_logger

logger = logging.getLogger(__name__)


class SystemStatusReporter:
    def __init__(self, args, max_total_token_num, dp_size_in_node, metric_client=None):
        self.enabled = not args.disable_log_stats
        self.interval = max(5, args.log_stats_interval)
        if args.log_stats_interval < 5:
            logger.warning(f"log_stats_interval={args.log_stats_interval}s is below minimum, using 5s")
        self.max_total_token_num = max_total_token_num
        self.dp_size_in_node = dp_size_in_node
        self.status_logger = init_system_status_logger("router")
        self.metric_client = metric_client

        # Accumulation counters (reset each interval)
        self.last_print_time = time.time()
        self.prompt_tokens = 0
        self.output_tokens = 0

        # Windowed counters for cache hit (reset each interval)
        self.window_input_total = 0
        self.window_gpu_cache_total = 0
        self.window_cpu_cache_total = 0

        # Global counters (never reset, for lifetime stats)
        self.global_input_total = 0
        self.global_gpu_cache_total = 0
        self.global_cpu_cache_total = 0
        self.global_mtp_output_total = 0
        self.global_mtp_accepted_total = 0

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

    def on_request_completed(
        self,
        input_len: int,
        output_len: int,
        gpu_cache_len: int,
        cpu_cache_len: int,
        mtp_accepted: int,
    ):
        if self.enabled:
            self.window_input_total += input_len
            self.window_gpu_cache_total += gpu_cache_len
            self.window_cpu_cache_total += cpu_cache_len
            self.global_input_total += input_len
            self.global_gpu_cache_total += gpu_cache_len
            self.global_cpu_cache_total += cpu_cache_len
            self.global_mtp_output_total += output_len
            self.global_mtp_accepted_total += mtp_accepted

    def maybe_print(
        self,
        running_batch,
        req_queue,
        read_only_statics_mem_manager,
        paused_req_num=0,
        radix_cache_client=None,
        disable_dynamic_prompt_cache=False,
    ):
        if not self.enabled:
            return
        now = time.time()
        elapsed = now - self.last_print_time
        if elapsed < self.interval:
            return

        total_tps = (self.prompt_tokens + self.output_tokens) / elapsed
        input_tps = self.prompt_tokens / elapsed
        output_tps = self.output_tokens / elapsed

        running = len(running_batch.reqs) if running_batch else 0
        queued = req_queue.get_wait_req_num()

        # Memory utilization (average across dp)
        # kv_used: physical KV memory usage (includes prefix cache tree occupancy)
        # kv_used_no_cache: effective usage excluding unrefed prefix cache tokens
        kv_used_list = []
        kv_used_no_cache_list = []
        for dp_i in range(self.dp_size_in_node):
            unrefed = read_only_statics_mem_manager.get_unrefed_token_num(dp_i)
            used = self.max_total_token_num - unrefed
            kv_used_list.append(used / self.max_total_token_num)
            if not disable_dynamic_prompt_cache and radix_cache_client is not None:
                cache_unrefed = radix_cache_client.get_unrefed_tokens_num(dp_i)
                active_used = max(0, used - cache_unrefed)
                kv_used_no_cache_list.append(active_used / self.max_total_token_num)
            else:
                kv_used_no_cache_list.append(used / self.max_total_token_num)
        avg_kv_used = sum(kv_used_list) / len(kv_used_list)
        avg_kv_used_no_cache = sum(kv_used_no_cache_list) / len(kv_used_no_cache_list)

        # Windowed prefix cache hit rates (this interval only)
        window_gpu_cache_hit_rate = (
            (self.window_gpu_cache_total / self.window_input_total * 100) if self.window_input_total > 0 else 0.0
        )
        window_cpu_cache_hit_rate = (
            (self.window_cpu_cache_total / self.window_input_total * 100) if self.window_input_total > 0 else 0.0
        )
        # Global prefix cache hit rates (lifetime)
        global_gpu_cache_hit_rate = (
            (self.global_gpu_cache_total / self.global_input_total * 100) if self.global_input_total > 0 else 0.0
        )
        global_cpu_cache_hit_rate = (
            (self.global_cpu_cache_total / self.global_input_total * 100) if self.global_input_total > 0 else 0.0
        )

        kv_pct = avg_kv_used * 100
        kv_pct_no_cache = avg_kv_used_no_cache * 100

        # Avg MTP accepted length (only shown when MTP is active)
        mtp_suffix = ""
        if self.global_mtp_accepted_total > 0:
            decode_steps = self.global_mtp_output_total - self.global_mtp_accepted_total
            avg_mtp_len = self.global_mtp_output_total / max(decode_steps, 1)
            mtp_suffix = f", MTP {avg_mtp_len:.2f}"

        self.status_logger.info(
            f"TPS {total_tps:.1f} (in {input_tps:.1f}, out {output_tps:.1f}), "
            f"REQ {running}run, {queued}wait, {paused_req_num}pause, "
            f"KV CACHE {kv_pct:.1f}% (active {kv_pct_no_cache:.1f}%), "
            f"GPU CACHE HIT {window_gpu_cache_hit_rate:.1f}% (global {global_gpu_cache_hit_rate:.1f}%), "
            f"CPU CACHE HIT {window_cpu_cache_hit_rate:.1f}% (global {global_cpu_cache_hit_rate:.1f}%)"
            f"{mtp_suffix}"
        )

        if self.metric_client is not None:
            self.metric_client.gauge_set("lightllm_cache_hit_rate", global_gpu_cache_hit_rate / 100.0)
            self.metric_client.gauge_set("lightllm_gen_throughput", output_tps)
            self.metric_client.gauge_set("lightllm_num_running_reqs", running)

        # Reset windowed counters
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.window_input_total = 0
        self.window_gpu_cache_total = 0
        self.window_cpu_cache_total = 0
        self.last_print_time = now


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
