import time
from prometheus_client import Histogram, Counter, Gauge, start_http_server, push_to_gateway

MONITOR_INFO = {
    "lightllm_request_count": "The total number of requests",
    "lightllm_request_success": "The total number of requests",
    "lightllm_request_duration": "Duration of the request",
    "lightllm_request_validation_duration": "Validation time of the request",
    "lightllm_request_inference_duration": "Inference time of the request",
    "lightllm_request_mean_time_per_token_duration": "Per token time of the request",
    "lightllm_request_first_token_duration": "First token time of the request",
    "lightllm_request_input_length": "Length of the input tokens",
    "lightllm_request_generated_tokens": "Number of generated tokens",
    "lightllm_request_max_new_tokens": "Max new token",
    "lightllm_batch_next_size": "Batch size",
    "lightllm_batch_current_size": "Current batch size",
    "lightllm_batch_pause_size": "The number of pause requests",
    "lightllm_queue_size": "Queue size"
}

class Monitor:
    
    def __init__(self):
        duration_buckets = []
        value = 0.001
        n_duration_buckets = 35
        for _ in range(n_duration_buckets):
            value *= 1.5
            duration_buckets.append(value)
        self.duration_buckets = duration_buckets
        self.monitor_registry = {}

    def init_api_server_monitor(self, args):
        
        self.create_histogram("lightllm_request_duration", self.duration_buckets)
        self.create_histogram("lightllm_request_validation_duration", self.duration_buckets)
        self.create_counter("lightllm_request_count")
        self.create_counter("lightllm_request_success")

    def init_httpserver_monitor(self, args):
        max_req_input_len = args.max_req_input_len
        input_len_buckets = [max_req_input_len / 100. * (i + 1) for i in range(0, 100)]
        self.create_histogram("lightllm_request_input_length", input_len_buckets)
        
        max_req_total_len = args.max_req_total_len
        generate_tokens_buckets = [max_req_total_len / 100. * (i + 1) for i in range(0, 100)]
        self.create_histogram("lightllm_request_max_new_tokens", generate_tokens_buckets)
        self.create_histogram("lightllm_request_generated_tokens", generate_tokens_buckets)

        self.create_histogram("lightllm_request_inference_duration", self.duration_buckets)
        self.create_histogram("lightllm_request_mean_time_per_token_duration", self.duration_buckets)
        self.create_histogram("lightllm_request_first_token_duration", self.duration_buckets)

    def init_router_monitor(self):
        self.create_gauge("lightllm_queue_size")
        self.create_gauge("lightllm_batch_current_size")
        self.create_gauge("lightllm_batch_pause_size")
        batch_size_buckets = [i + 1 for i in range(0, 1024)]
        self.create_histogram("lightllm_batch_next_size", batch_size_buckets)

    def create_histogram(self, name, buckets):
        histogram = Histogram(name, MONITOR_INFO[name], buckets=buckets)
        self.monitor_registry[name] = histogram 

    def create_counter(self, name):
        histogram = Counter(name, MONITOR_INFO[name])
        self.monitor_registry[name] = histogram 

    def create_gauge(self, name):
        gauge = Gauge(name,  MONITOR_INFO[name])
        self.monitor_registry[name] = gauge 

    def counter_inc(self, name):
        self.monitor_registry[name].inc()

    def histogram_observe(self, name, value):
        self.monitor_registry[name].observe(value) 

    def gauge_set(self, name, value):
        self.monitor_registry[name].set(value) 

    def histogram_timer(self, name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time() 
                result = func(*args, **kwargs)  
                elapsed_time = time.time() - start_time  
                self.monitor_registry[name].observe(elapsed_time) 
                return result
            return wrapper
        return decorator

monitor = Monitor()