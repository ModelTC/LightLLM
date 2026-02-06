import time
from lightllm.utils.log_utils import init_system_status_logger


class SystemStatusReporter:
    def __init__(self, args, max_total_token_num, dp_size_in_node):
        self.enabled = not args.disable_log_stats
        self.interval = max(5, args.log_stats_interval)
        self.max_total_token_num = max_total_token_num
        self.dp_size_in_node = dp_size_in_node
        self.status_logger = init_system_status_logger("router")

        # Accumulation counters (reset each interval)
        self.last_print_time = time.time()
        self.prompt_tokens = 0
        self.output_tokens = 0

        # Global counters (never reset, for lifetime stats)
        self.global_input_total = 0
        self.global_cache_total = 0
        self.global_mtp_output_total = 0
        self.global_mtp_accepted_total = 0

    def count_prompt_tokens(self, num_tokens: int):
        if self.enabled:
            self.prompt_tokens += num_tokens

    def count_output_tokens(self, num_tokens: int):
        if self.enabled:
            self.output_tokens += num_tokens

    def on_request_completed(self, input_len: int, output_len: int, cache_len: int, mtp_accepted: int):
        if self.enabled:
            self.global_input_total += input_len
            self.global_cache_total += cache_len
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
                kv_used_no_cache_list.append((used - cache_unrefed) / self.max_total_token_num)
            else:
                kv_used_no_cache_list.append(used / self.max_total_token_num)
        avg_kv_used = sum(kv_used_list) / len(kv_used_list)
        avg_kv_used_no_cache = sum(kv_used_no_cache_list) / len(kv_used_no_cache_list)

        # Global prefix cache hit rate
        cache_hit_rate = (
            (self.global_cache_total / self.global_input_total * 100) if self.global_input_total > 0 else 0.0
        )

        # Avg MTP accepted length (global, 1.0 when MTP is off)
        decode_steps = self.global_mtp_output_total - self.global_mtp_accepted_total
        avg_mtp_len = self.global_mtp_output_total / max(decode_steps, 1) if self.global_mtp_output_total > 0 else 1.0

        kv_pct = avg_kv_used * 100
        kv_pct_no_cache = avg_kv_used_no_cache * 100

        self.status_logger.info(
            f"Throughput {total_tps:>7.1f} tok/s (in {input_tps:.1f}, out {output_tps:.1f}) | "
            f"Reqs {running} run, {queued} wait, {paused_req_num} pause | "
            f"KV Cache {kv_pct:.1f}% (active {kv_pct_no_cache:.1f}%) | "
            f"Prefix Hit {cache_hit_rate:.1f}% | "
            f"mtp {avg_mtp_len:.2f}"
        )

        # Reset windowed counters
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.last_print_time = now
