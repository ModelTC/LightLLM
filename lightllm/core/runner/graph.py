"""Batch execution policy; model recipes and CUDA capture mechanisms stay separate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeStep:
    batch_size: int
    graph: object = None
    capture: bool = False

    def prepare(self, batch):
        batch.is_cuda_graph_capture = self.capture
        batch.is_cuda_graph_replay = self.graph is not None and not self.capture


class GraphRunner:
    def __init__(self):
        self.decode = None
        self.prefill = None

    def close(self):
        """Release captured work after the caller has drained device execution."""
        for name in ("prefill", "decode"):
            graph = getattr(self, name)
            if graph is not None:
                graph.close()
                setattr(self, name, None)

    def plan_decode(self, batch_size, max_kv_length):
        graph = self.decode
        if graph is None or not graph.can_run(batch_size=batch_size, max_len_in_batch=max_kv_length):
            return DecodeStep(batch_size)
        size = graph.find_closest_graph_batch_size(batch_size=batch_size)
        return DecodeStep(size, graph, graph.need_capture(size))

    def decode_metadata_capacity(self, batch_size, max_kv_length):
        graph = self.decode
        if graph is not None and graph.can_run(batch_size, max_kv_length):
            return graph.graph_max_len_in_batch
        return max_kv_length

    def run_decode(self, step, forward, *batches):
        if step.graph is None:
            return forward(*batches)
        kwargs = {"infer_state1": batches[1]} if len(batches) == 2 else {}
        if step.capture:
            return step.graph.capture_decode(forward, batches[0], **kwargs)
        return step.graph.replay(batches[0], **kwargs)

    def run_prefill(self, forward, inputs, batch):
        graph, count = self.prefill, batch.input_ids.shape[0]
        if graph is not None and graph.can_run(count):
            if graph.find_closest_graph_handle_token_num(count) != count:
                raise ValueError("prefill inputs must be padded before Graph execution")
            if graph.need_capture(count):
                return graph.capture_prefill(forward, inputs, batch)
            return graph.replay(inputs, batch)

        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        g_cache_manager.cache_env_in()
        try:
            return forward(inputs, batch)
        finally:
            g_cache_manager.cache_env_out()
