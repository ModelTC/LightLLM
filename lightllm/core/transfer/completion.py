"""A transfer has separate source and destination completion points."""

from dataclasses import dataclass
from typing import Protocol


class Completion(Protocol):
    def query(self) -> bool:
        ...

    def synchronize(self) -> None:
        ...


@dataclass(frozen=True)
class TransferCompletion:
    source_reusable: Completion
    destination_ready: Completion
    # Bound views and indices remain alive for the duration of asynchronous work.
    keepalive: object = None


class TransferFailure(RuntimeError):
    def __init__(self, cause: Exception, quiesced: Completion, keepalive):
        super().__init__(f"copy submission failed: {cause}")
        self.quiesced = quiesced
        self.keepalive = keepalive


class Transfer:
    def __init__(self, runtime=None):
        if runtime is None:
            from lightllm.core.device.cuda import CudaRuntime

            runtime = CudaRuntime()
        self.runtime = runtime

    def submit(self, plan, *, stream=None, destination_barrier=None):
        """A barrier may publish a destination only after all TP shards arrive.

        On partial submission failure there is no ready destination. The failure's
        fence must complete before either side's resources may be recycled.
        """
        with self.runtime.use_stream(stream):
            try:
                plan.run()
                source = self.runtime.record_event()
                if destination_barrier is None:
                    destination = source
                else:
                    destination_barrier()
                    destination = self.runtime.record_event()
            except Exception as exc:
                quiesced = self.runtime.record_event()
                raise TransferFailure(exc, quiesced, plan) from exc
        return TransferCompletion(source, destination, plan)
