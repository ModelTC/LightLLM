"""Model execution owned by the autonomous image/video server."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from lightllm.server.generationserver.image.protocol import (
    ImageGenerationInput,
    ImageModelTask,
    ImageOutcome,
    ImageWorkerResult,
)

from lightllm.server.activation_transfer import (
    ActivationPayload,
    ConsumerExecutionError as GenerationBackendError,
)


@dataclass(frozen=True)
class ImageGenerationResult:
    generation_id: str
    image: object
    generation_ms: float = 0.0


class ImageBackend:
    def __init__(self, model_rpcs: list) -> None:
        self._model_rpcs = model_rpcs
        self._active: set[str] = set()

    async def generate(
        self,
        generation: ImageGenerationInput,
        activations: dict[str, ActivationPayload],
    ) -> ImageGenerationResult:
        if not isinstance(generation, ImageGenerationInput):
            raise TypeError(f"unsupported image task: {type(generation)!r}")
        if not self._model_rpcs:
            raise GenerationBackendError(
                "image worker is not initialized",
                safe_to_release=True,
            )

        self._active.add(generation.generation_id)
        try:
            try:
                model_task = ImageModelTask(
                    generation=generation,
                    activations=dict(activations),
                )
            except Exception as exc:
                raise GenerationBackendError(
                    f"failed to prepare image inputs: {exc!r}",
                    safe_to_release=True,
                ) from exc
            results = await asyncio.gather(
                *(rpc.run_task(model_task) for rpc in self._model_rpcs),
                return_exceptions=True,
            )
            workers = self._validate_results(generation, results)
            if any(not worker.safe_to_release for worker in workers):
                raise GenerationBackendError(
                    "an image worker did not return a CUDA-safe acknowledgement",
                    safe_to_release=False,
                )
            errors = [
                worker.error_info or f"image worker rank {worker.rank} failed"
                for worker in workers
                if worker.outcome is not ImageOutcome.SUCCESS
            ]
            if errors:
                raise GenerationBackendError(
                    "; ".join(errors),
                    safe_to_release=True,
                )
            try:
                rank_zero = workers[0]
                return ImageGenerationResult(
                    generation_id=generation.generation_id,
                    image=list(rank_zero.artifacts),
                    generation_ms=max(worker.generation_ms for worker in workers),
                )
            except Exception as exc:
                raise GenerationBackendError(
                    f"failed to prepare image output: {exc!r}",
                    safe_to_release=True,
                ) from exc
        finally:
            self._active.discard(generation.generation_id)

    async def cancel(self, task_id: str) -> None:
        if task_id in self._active:
            raise GenerationBackendError(
                "image worker RPC cannot be cancelled safely",
                safe_to_release=False,
            )

    def _validate_results(
        self,
        generation: ImageGenerationInput,
        results: list,
    ) -> list[ImageWorkerResult]:
        if len(results) != len(self._model_rpcs):
            raise GenerationBackendError(
                "image worker result count mismatch",
                safe_to_release=False,
            )
        workers = []
        for rank, result in enumerate(results):
            if isinstance(result, BaseException):
                raise GenerationBackendError(
                    f"image worker rank {rank} RPC failed: {result!r}",
                    safe_to_release=False,
                )
            if not isinstance(result, ImageWorkerResult):
                raise GenerationBackendError(
                    f"image worker rank {rank} returned {type(result)!r}",
                    safe_to_release=False,
                )
            if (
                result.rank != rank
                or result.generation_id != generation.generation_id
            ):
                raise GenerationBackendError(
                    f"stale image worker result from rank {rank}",
                    safe_to_release=False,
                )
            workers.append(result)
        return workers
