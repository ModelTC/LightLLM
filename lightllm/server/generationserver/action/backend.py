"""Model execution owned by the autonomous VLA server."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

import torch
from .protocol import (
    ActionModelTask,
    ActionOutcome,
    ActionRequest,
    ActionTaskIdentity,
    ActionWorkerAck,
)

from lightllm.server.activation_transfer import (
    ConsumerExecutionError as GenerationBackendError,
    KVActivation,
)


class ActionProcessor(Protocol):
    """Environment-facing state/action conversion used by ActionBackend."""

    def normalize_state(self, state, *, domain: str | None = None) -> torch.Tensor:
        ...

    def postprocess_actions(
        self, actions, *, raw_state, domain: str | None = None
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True, kw_only=True)
class ActionGenerationTask:
    task_id: str
    # Number of future control steps produced by this request.
    action_horizon: int
    # Robot-visible width returned to the caller. The checkpoint may use a
    # larger fixed projection width internally.
    action_dimension: int
    # ABA-safe identity copied into every TP worker ACK.
    identity: ActionTaskIdentity
    # Original action payload, retained for state/noise and post-processing.
    action_request: ActionRequest
    # Validated per-request snapshot after applying bounded request overrides.
    request_config: dict
    # CPU RNG seed used only when the caller did not provide explicit noise.
    seed: int

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        for value, name in (
            (self.action_horizon, "action_horizon"),
            (self.action_dimension, "action_dimension"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        # SamplingParams uses -1 as its default seed. torch.Generator keeps a
        # stable signed-to-uint64 mapping, so preserve that existing policy.
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("action seed must be an integer")
        if (
            self.request_config["action_horizon"] != self.action_horizon
            or self.request_config["action_dim"] != self.action_dimension
        ):
            raise ValueError("action request config and public shape disagree")


@dataclass(frozen=True)
class ActionGenerationResult:
    task_id: str
    actions: torch.Tensor
    # CUDA time measured inside the Action Expert worker.
    action_expert_ms: float = 0.0
    # End-to-end backend time: preparation, RPC, synchronization and postprocess.
    backend_ms: float = 0.0


class ActionBackend:
    def __init__(
        self,
        model_rpcs: list,
        processor: ActionProcessor,
        *,
        tp_size: int,
    ) -> None:
        # The server fills this shared list in place as TP workers start.
        self._model_rpcs = model_rpcs
        self._processor = processor
        # One ACK is required from each target-model TP rank before its shared
        # Prefix KV consumer lease can be released.
        self._tp_size = tp_size
        # RPyC execution cannot be interrupted safely; membership means the
        # task may still be reading shared Prefix KV on CUDA.
        self._active: set[str] = set()

    async def generate(
        self,
        task: ActionGenerationTask,
        prefix: KVActivation,
    ) -> ActionGenerationResult:
        if not isinstance(task, ActionGenerationTask):
            raise TypeError(f"unsupported action task: {type(task)!r}")
        if not isinstance(prefix, KVActivation):
            raise TypeError("action generation requires one KVActivation")
        if len(self._model_rpcs) != self._tp_size:
            raise GenerationBackendError(
                "action workers are not initialized",
                safe_to_release=True,
            )

        self._active.add(task.task_id)
        started = time.perf_counter()
        try:
            # Action uses one target VLM prefix; unlike image CFG there is no
            # conditional/unconditional prefix pair.
            try:
                model_task = self._prepare_model_task(task, prefix)
            except Exception as exc:
                raise GenerationBackendError(
                    f"failed to prepare action inputs: {exc!r}",
                    safe_to_release=True,
                ) from exc
            worker_results = await asyncio.gather(
                *(rpc.run_task(model_task) for rpc in self._model_rpcs),
                return_exceptions=True,
            )
            acks = self._validate_acks(task, worker_results)
            # safe_to_release is a CUDA completion guarantee, not merely a
            # successful RPC return. One unsafe rank pins the entire lease.
            if any(not ack.safe_to_release for ack in acks.values()):
                raise GenerationBackendError(
                    "an action worker did not return a CUDA-safe acknowledgement",
                    safe_to_release=False,
                )
            worker_errors = [
                ack.error_info or f"action worker rank {ack.rank} failed"
                for ack in acks.values()
                if ack.outcome is not ActionOutcome.SUCCESS
            ]
            if worker_errors:
                raise GenerationBackendError(
                    "; ".join(worker_errors),
                    safe_to_release=True,
                )
            output_ack = acks[0]
            if output_ack.actions is None:
                raise GenerationBackendError(
                    "action rank 0 returned no output",
                    safe_to_release=True,
                )
            # raw_state stays in robot coordinates and is used only for action
            # unnormalization/relative-action adapters after inference.
            raw_state = task.action_request.raw_state
            if raw_state is None:
                raw_state = task.action_request.state
            processor_args = {"raw_state": raw_state}
            if task.action_request.domain is not None:
                processor_args["domain"] = task.action_request.domain
            try:
                actions = self._processor.postprocess_actions(
                    output_ack.actions,
                    **processor_args,
                )
                # The public HTTP action path is one observation per request.
                if actions.ndim == 3 and actions.shape[0] == 1:
                    actions = actions[0]
            except Exception as exc:
                raise GenerationBackendError(
                    f"failed to postprocess action output: {exc!r}",
                    safe_to_release=True,
                ) from exc
            return ActionGenerationResult(
                task_id=task.task_id,
                actions=actions,
                action_expert_ms=output_ack.action_expert_ms,
                backend_ms=(time.perf_counter() - started) * 1000.0,
            )
        finally:
            self._active.discard(task.task_id)

    async def cancel(self, task_id: str) -> None:
        if task_id in self._active:
            raise GenerationBackendError(
                "action worker RPC cannot be cancelled safely",
                safe_to_release=False,
            )

    def _prepare_model_task(
        self,
        task: ActionGenerationTask,
        prefix: KVActivation,
    ) -> ActionModelTask:
        # Continuous-state policies consume this tensor in the action model;
        # prefix-state policies already encoded state while building Prefix KV.
        state = None
        if task.request_config["action_expert_uses_state"]:
            assert task.action_request.state is not None
            if task.action_request.domain is None:
                state = self._processor.normalize_state(task.action_request.state)
            else:
                state = self._processor.normalize_state(
                    task.action_request.state,
                    domain=task.action_request.domain,
                )
            if state.ndim == 1:
                state = state.unsqueeze(0)

        # Noise always uses the checkpoint projection width:
        # [batch=1, action_horizon, max_action_dim]. action_dimension only
        # slices the final robot-visible output.
        if task.action_request.noise is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(task.seed)
            noise = torch.randn(
                1,
                task.action_horizon,
                task.request_config["max_action_dim"],
                generator=generator,
                dtype=torch.float32,
            )
        else:
            noise = torch.as_tensor(task.action_request.noise, dtype=torch.float32)
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)

        return ActionModelTask(
            identity=task.identity,
            prefix_kv=prefix,
            state=state,
            noisy_actions=noise,
            action_horizon=task.action_horizon,
            action_dim=task.action_dimension,
            num_denoise_steps=task.request_config["num_denoise_steps"],
        )

    def _validate_acks(
        self,
        task: ActionGenerationTask,
        results: list,
    ) -> dict[int, ActionWorkerAck]:
        # Rank-keyed ACKs make duplicate, stale and missing TP replies explicit.
        acks: dict[int, ActionWorkerAck] = {}
        for result in results:
            if isinstance(result, BaseException):
                raise GenerationBackendError(
                    f"action worker RPC failed: {result!r}",
                    safe_to_release=False,
                )
            if not isinstance(result, ActionWorkerAck):
                raise GenerationBackendError(
                    f"invalid action worker ACK: {type(result)!r}",
                    safe_to_release=False,
                )
            if result.identity != task.identity or result.rank in acks:
                raise GenerationBackendError(
                    f"stale or duplicate action worker ACK from rank {result.rank}",
                    safe_to_release=False,
                )
            acks[result.rank] = result
        missing = set(range(self._tp_size)).difference(acks)
        if missing:
            raise GenerationBackendError(
                f"missing action worker ACKs from ranks {sorted(missing)}",
                safe_to_release=False,
            )
        return acks
