import time
import logging
import subprocess
from typing import Dict
from lightllm.server.core.objs import StartArgs
from lightllm.utils.log_utils import init_system_status_logger

logger = logging.getLogger(__name__)


class SystemStatusReporter:
    def __init__(self, args, max_total_token_num, dp_size_in_node):
        self.enabled = not args.disable_log_stats
        self.interval = max(5, args.log_stats_interval)
        if args.log_stats_interval < 5:
            logger.warning(f"log_stats_interval={args.log_stats_interval}s is below minimum, using 5s")
        self.max_total_token_num = max_total_token_num
        self.dp_size_in_node = dp_size_in_node
        self.status_logger = init_system_status_logger("router")

        self.last_print_time = time.time()
        self.prompt_tokens = 0
        self.output_tokens = 0

        self.window_input_total = 0
        self.window_cache_total = 0

        self.global_input_total = 0
        self.global_cache_total = 0
        self.global_mtp_output_total = 0
        self.global_mtp_accepted_total = 0

        # Per-req shm_cur_output_len snapshot at the previous window boundary,
        # used to compute the windowed output-token count without per-tick scans.
        self._req_last_output_len: Dict[int, int] = {}

    def count_prompt_tokens(self, num_tokens: int):
        if self.enabled:
            self.prompt_tokens += num_tokens

    def discard_req(self, req):
        """Settle a finished/aborted req's tail output tokens (those produced after the last
        window-boundary sweep) and drop its tracking entry."""
        if not self.enabled:
            return
        cur_out_len = req.shm_cur_output_len
        prev_out_len = self._req_last_output_len.pop(req.request_id, 0)
        if cur_out_len > prev_out_len:
            self.output_tokens += cur_out_len - prev_out_len

    def on_request_completed(self, input_len: int, output_len: int, cache_len: int, mtp_accepted: int):
        if self.enabled:
            self.window_input_total += input_len
            self.window_cache_total += cache_len
            self.global_input_total += input_len
            self.global_cache_total += cache_len
            self.global_mtp_output_total += output_len
            self.global_mtp_accepted_total += mtp_accepted

    def _get_gpu_status_for_debug(self) -> str:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError) as e:
            return f"gpu=unavailable({e.__class__.__name__})"

        gpu_infos = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            gpu_index, util, mem_used, mem_total = parts
            try:
                mem_used_mb = float(mem_used)
                mem_total_mb = float(mem_total)
                mem_ratio = mem_used_mb / mem_total_mb * 100 if mem_total_mb > 0 else 0.0
                mem_used_gb = mem_used_mb / 1024
                mem_total_gb = mem_total_mb / 1024
                gpu_infos.append(
                    f"{gpu_index}(util={float(util):.0f}%,mem={mem_ratio:.1f}%,"
                    f"used={mem_used_gb:.1f}GiB/{mem_total_gb:.1f}GiB)"
                )
            except ValueError:
                continue
        if not gpu_infos:
            return "gpu=unavailable(empty)"
        return "gpu=[" + ";".join(gpu_infos) + "]"

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

        # Single bulk sweep at the window boundary: account for output tokens produced
        # by every still-running req since the previous boundary, and refresh their
        # snapshots. Reqs that finished in this window already settled via discard_req.
        if running_batch is not None:
            for req in running_batch.reqs:
                cur_out_len = req.shm_cur_output_len
                prev_out_len = self._req_last_output_len.get(req.request_id, 0)
                if cur_out_len > prev_out_len:
                    self.output_tokens += cur_out_len - prev_out_len
                    self._req_last_output_len[req.request_id] = cur_out_len

        total_tps = (self.prompt_tokens + self.output_tokens) / elapsed
        input_tps = self.prompt_tokens / elapsed
        output_tps = self.output_tokens / elapsed

        running = len(running_batch.reqs) if running_batch else 0
        queued = req_queue.get_wait_req_num()

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

        window_cache_hit_rate = (
            (self.window_cache_total / self.window_input_total * 100) if self.window_input_total > 0 else 0.0
        )
        global_cache_hit_rate = (
            (self.global_cache_total / self.global_input_total * 100) if self.global_input_total > 0 else 0.0
        )

        kv_pct = avg_kv_used * 100
        kv_pct_no_cache = avg_kv_used_no_cache * 100

        log_parts = [
            f"router_status(window={elapsed:.1f}s)",
            f"throughput(total={total_tps:.1f},input={input_tps:.1f},output={output_tps:.1f})",
            f"req(running={running},waiting={queued},paused={paused_req_num})",
            f"kv(used={kv_pct_no_cache:.1f}%)",
            f"gpu_cache_hit(window={window_cache_hit_rate:.1f}%,global={global_cache_hit_rate:.1f}%)",
        ]

        if self.global_mtp_accepted_total > 0:
            decode_steps = self.global_mtp_output_total - self.global_mtp_accepted_total
            avg_mtp_len = self.global_mtp_output_total / max(decode_steps, 1)
            log_parts.append(
                f"mtp(avg_tokens_per_step={avg_mtp_len:.2f},"
                f"accepted={self.global_mtp_accepted_total},output={self.global_mtp_output_total})"
            )

        self.status_logger.info(" | ".join(log_parts))
        if logger.isEnabledFor(logging.DEBUG):
            kv_unrefed_prefix_cache_pct = max(0.0, kv_pct - kv_pct_no_cache)
            debug_parts = [
                "router_status_debug",
                f"kv_physical={kv_pct:.1f}%",
                f"kv_unrefed_prefix_cache={kv_unrefed_prefix_cache_pct:.1f}%",
                f"throughput_tokens(input={self.prompt_tokens},output={self.output_tokens})",
                f"gpu_cache_tokens(window={self.window_cache_total}/{self.window_input_total},"
                f"global={self.global_cache_total}/{self.global_input_total})",
                f"tracked_output_reqs={len(self._req_last_output_len)}",
                self._get_gpu_status_for_debug(),
            ]
            logger.debug(" | ".join(debug_parts))

        self.prompt_tokens = 0
        self.output_tokens = 0
        self.window_input_total = 0
        self.window_cache_total = 0
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
