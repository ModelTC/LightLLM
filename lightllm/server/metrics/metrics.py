import os
import time
from prometheus_client import CollectorRegistry, Histogram, Counter, Gauge
from prometheus_client import push_to_gateway
from prometheus_client.exposition import basic_auth_handler

MONITOR_INFO = {
    "lightllm_request_count": "The total number of requests",
    "lightllm_request_success": "The number of successful requests",
    "lightllm_request_failure": "The number of failed requests",
    "lightllm_request_duration": "Duration of the request (s)",
    "lightllm_request_validation_duration": "Validation time of the request",
    "lightllm_request_inference_duration": "Inference time of the request",
    "lightllm_request_mean_time_per_token_duration": "Per token time of the request",
    "lightllm_request_first_token_duration": "First token time of the request",
    "lightllm_request_input_length": "Length of the input tokens",
    "lightllm_request_generated_tokens": "Number of generated tokens",
    "lightllm_request_max_new_tokens": "Max new token",
    "lightllm_batch_next_size": "Batch size of the next new batch",
    "lightllm_batch_current_size": "Current batch size",
    "lightllm_batch_pause_size": "The number of pause requests",
    "lightllm_queue_size": "Queue size",
    "lightllm_request_queue_duration_bucket": "Queue duration of requests",
    "lightllm_batch_inference_count": "The number of prefill steps / decode steps",
    "lightllm_batch_inference_duration_bucket": "Inference time of prefill step / decode step",
    "lightllm_cache_length": "Length of tokens which hit prompt cache",
    "lightllm_cache_ratio": "cache length / input_length",
    "lightllm_batch_current_max_tokens": "dynamic max token used for current batch",
    "lightllm_request_mtp_avg_token_per_step": "Average number of tokens per step",
}


def my_auth_handler(url, method, timeout, headers, data):
    username = os.getenv("USERNAME", None)
    password = os.getenv("PASSWORD", None)
    if username is None or password is None:
        raise ValueError("USERNAME and PASSWORD must be set when the auth is opened.")
    return basic_auth_handler(url, method, timeout, headers, data, username, password)


class Monitor:
    def __init__(self, args):
        duration_buckets = []
        value = 0.001
        n_duration_buckets = 35
        for _ in range(n_duration_buckets):
            value *= 1.5
            duration_buckets.append(value)
        self.duration_buckets = duration_buckets
        self.monitor_registry = {}
        self.gateway_url = args.metric_gateway
        self.registry = CollectorRegistry()
        self.job_name = args.job_name
        self.grouping_key = {}
        if args.grouping_key:
            for item in args.grouping_key:
                key, value = item.split("=")
                self.grouping_key[key] = value
        self.auth = args.enable_monitor_auth
        self.init_metrics(args)

    def init_metrics(self, args):

        self.create_histogram("lightllm_request_duration", self.duration_buckets)
        self.create_histogram("lightllm_request_validation_duration", self.duration_buckets)
        self.create_counter("lightllm_request_count")
        self.create_counter("lightllm_request_success")
        self.create_counter("lightllm_request_failure")
        self.create_counter("lightllm_batch_inference_count", labelnames=["method"])

        max_req_input_len = args.max_req_total_len
        input_len_buckets = [max_req_input_len / 100.0 * (i + 1) for i in range(-1, 100)]
        self.create_histogram("lightllm_request_input_length", input_len_buckets)
        self.create_histogram("lightllm_cache_length", input_len_buckets)

        max_req_total_len = args.max_req_total_len
        generate_tokens_buckets = [max_req_total_len / 100.0 * (i + 1) for i in range(-1, 100)]
        self.create_histogram("lightllm_request_max_new_tokens", generate_tokens_buckets)
        self.create_histogram("lightllm_request_generated_tokens", generate_tokens_buckets)

        self.create_histogram("lightllm_request_inference_duration", self.duration_buckets)
        self.create_histogram("lightllm_request_mean_time_per_token_duration", self.duration_buckets)
        self.create_histogram("lightllm_request_first_token_duration", self.duration_buckets)
        self.create_histogram("lightllm_request_queue_duration_bucket", self.duration_buckets)
        self.create_histogram("lightllm_batch_inference_duration_bucket", self.duration_buckets, labelnames=["method"])
        self.gateway_url = args.metric_gateway

        self.create_gauge("lightllm_queue_size")
        self.create_gauge("lightllm_batch_current_size")
        self.create_gauge("lightllm_batch_pause_size")
        self.create_gauge("lightllm_batch_current_max_tokens")
        batch_size_buckets = [i + 1 for i in range(0, 128)]
        self.create_histogram("lightllm_batch_next_size", batch_size_buckets)

        ratio_buckets = [(i + 1) / 10.0 for i in range(-1, 10)]
        self.create_histogram("lightllm_cache_ratio", ratio_buckets)

        mtp_avg_token_per_step_buckets = [i / 10.0 + 1.0 for i in range(0, 10 * args.mtp_step)]
        self.create_histogram("lightllm_request_mtp_avg_token_per_step", mtp_avg_token_per_step_buckets)

    def create_histogram(self, name, buckets, labelnames=None):
        if labelnames is None:
            histogram = Histogram(name, MONITOR_INFO[name], buckets=buckets, registry=self.registry)
        else:
            histogram = Histogram(
                name, MONITOR_INFO[name], labelnames=labelnames, buckets=buckets, registry=self.registry
            )
        self.monitor_registry[name] = histogram

    def create_counter(self, name, labelnames=None):
        if labelnames is None:
            histogram = Counter(name, MONITOR_INFO[name], registry=self.registry)
        else:
            histogram = Counter(name, MONITOR_INFO[name], labelnames=labelnames, registry=self.registry)
        self.monitor_registry[name] = histogram

    def create_gauge(self, name):
        gauge = Gauge(name, MONITOR_INFO[name], registry=self.registry)
        self.monitor_registry[name] = gauge

    def counter_inc(self, name, label=None):
        if label is None:
            self.monitor_registry[name].inc()
        else:
            self.monitor_registry[name].labels(method=label).inc()

    def histogram_observe(self, name, value, label=None):
        if label is None:
            self.monitor_registry[name].observe(value)
        else:
            self.monitor_registry[name].labels(method=label).observe(value)

    def gauge_set(self, name, value):
        self.monitor_registry[name].set(value)

    def push_metrices(self):
        if self.gateway_url is not None:
            if self.auth:
                push_to_gateway(
                    self.gateway_url,
                    job=self.job_name,
                    grouping_key=self.grouping_key,
                    registry=self.registry,
                    handler=my_auth_handler,
                )
            else:
                push_to_gateway(
                    self.gateway_url, job=self.job_name, grouping_key=self.grouping_key, registry=self.registry
                )
