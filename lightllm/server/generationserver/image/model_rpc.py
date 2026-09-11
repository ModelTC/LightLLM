"""Image-model worker process."""

from __future__ import annotations

import base64
import inspect
import io
import time
from typing import Protocol

import rpyc
import torch
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer

from lightllm.models.registry import get_generation_model_class
from lightllm.server.activation_transfer import KVActivation
from lightllm.server.generationserver.kv_materializer import PagedKVMaterializer
from lightllm.server.generationserver.image.protocol import (
    GeneratedArtifact,
    ImageModelTask,
    ImageOutcome,
    ImageWorkerResult,
)
from lightllm.server.generationserver.worker import start_model_worker
from lightllm.utils.config_utils import get_config_json
from lightllm.utils.envs_utils import get_unique_server_name, set_env_start_args
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import start_parent_check_thread

logger = init_logger(__name__)


class ImageGenerationModel(Protocol):
    """Small model-owned interface consumed by the image RPC worker."""

    def generate(self, *, activations: dict, generation) -> list:
        ...

    def finish_task(self) -> None:
        ...


class ImageModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        kvargs = obtain(kvargs)
        self.args = kvargs["args"]
        set_env_start_args(self.args)
        self.rank = 0
        self.device_id = 0
        torch.cuda.set_device(self.device_id)

        self.kv_materializer = PagedKVMaterializer.from_shared(
            self.rank,
            device_index=self.device_id,
        )
        model_class = get_generation_model_class(
            get_config_json(self.args.model_dir),
            "image",
        )
        self.model: ImageGenerationModel = model_class(
            self.args,
            prefix_layout=self.kv_materializer.prefix_layout,
            device_id=self.device_id,
        )

    @torch.no_grad()
    def exposed_run_task(self, task):
        task = obtain(task)
        if not isinstance(task, ImageModelTask):
            raise TypeError(f"invalid generation model task: {type(task)!r}")
        started = time.perf_counter()
        artifacts = []
        outcome = ImageOutcome.SUCCESS
        error_info = None
        safe_to_release = True
        try:
            model_activations = {
                role: self.kv_materializer.gather(activation)
                if isinstance(activation, KVActivation)
                else activation
                for role, activation in task.activations.items()
            }
            artifacts = [
                self._to_artifact(output)
                for output in self.model.generate(
                    activations=model_activations,
                    generation=task.generation,
                )
            ]
        except Exception as exc:
            logger.exception("generation task %s failed", task.generation.generation_id)
            outcome = ImageOutcome.ERROR
            error_info = repr(exc)
        finally:
            try:
                if hasattr(self, "model"):
                    self.model.finish_task()
                torch.cuda.synchronize(self.device_id)
            except Exception as exc:
                safe_to_release = False
                outcome = ImageOutcome.RESTART_REQUIRED
                suffix = repr(exc)
                error_info = suffix if error_info is None else f"{error_info}; {suffix}"

        return ImageWorkerResult(
            generation_id=task.generation.generation_id,
            rank=self.rank,
            outcome=outcome,
            safe_to_release=safe_to_release,
            artifacts=artifacts
            if outcome is ImageOutcome.SUCCESS and self.rank == 0
            else [],
            generation_ms=(time.perf_counter() - started) * 1000.0,
            error_info=error_info,
        )

    @staticmethod
    def _to_artifact(value) -> GeneratedArtifact:
        if isinstance(value, str):
            encoded = value.partition(",")[2] if value.startswith("data:") else value
            # Validate backend output before it crosses the process boundary.
            base64.b64decode(encoded, validate=True)
        elif isinstance(value, bytes):
            encoded = base64.b64encode(value).decode("ascii")
        else:
            try:
                from PIL import Image

                if not isinstance(value, Image.Image):
                    raise TypeError(f"unsupported image output: {type(value)!r}")
                buffer = io.BytesIO()
                value.convert("RGB").save(buffer, format="JPEG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            except ImportError as exc:
                raise TypeError(f"unsupported image output: {type(value)!r}") from exc
        return GeneratedArtifact("image/jpeg", encoded)


def _init_env(socket_path: str, success_event):
    graceful_registry(inspect.currentframe().f_code.co_name)
    import setproctitle

    setproctitle.setproctitle(
        f"lightllm::{get_unique_server_name()}::generation_model_infer"
    )
    start_parent_check_thread()
    import lightllm.utils.rpyc_fix_utils as _  # noqa: F401

    server = ThreadedServer(
        ImageModelRpcServer(),
        socket_path=socket_path,
        protocol_config={"allow_pickle": True},
    )
    success_event.set()
    server.start()


async def start_model_process():
    return await start_model_worker(
        _init_env,
        socket_prefix="lightllm_generation_infer",
        worker_name="generation model",
    )
