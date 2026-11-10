from lightllm.common.attention.api import GraphBinding


class FiaGraphBinding(GraphBinding):
    def __init__(self, plan, runtime):
        self.plan, self.runtime = plan, runtime
        self.update_stream = runtime.create_stream()
        self.completion = runtime.create_event()
        self.pending = False

    def wait_for_update(self):
        if self.pending:
            self.completion.synchronize()
            self.pending = False

    def after_submit(self):
        import torch_npu

        api = self.runtime.api
        with self.runtime.use_stream(self.update_stream):
            for handle, event, kwargs, workspace, output in self.plan.tasks:
                api.graph_task_update_begin(self.update_stream, handle)
                torch_npu.npu_fused_infer_attention_score.out(**kwargs, workspace=workspace, out=output)
                api.graph_task_update_end(self.update_stream)
                event.record(self.update_stream)
            self.completion.record(self.update_stream)
        self.pending = True
        return (self.completion,)

    def close(self):
        self.wait_for_update()
        self.plan.tasks.clear()
