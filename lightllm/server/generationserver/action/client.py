"""Action transport and persistent-context delivery cleanup."""

from __future__ import annotations

import asyncio

from lightllm.utils.log_utils import init_logger

from .protocol import (
    ActionCancel,
    ActionContextTask,
    ActionRequest,
    ActionResult,
    PrefixContextOp,
    PrefixContextRef,
)
from lightllm.server.generationserver.client import GenerationClient


logger = init_logger(__name__)


class ActionDispatchNotSent(RuntimeError):
    """The send failed before ownership reached GenerationServer."""


class ActionGenerationClient(GenerationClient):
    result_type = ActionResult
    dispatch_error = ActionDispatchNotSent
    label = "action generation"

    def _make_cancel(self, correlation_id: str, reason: str):
        return ActionCancel(correlation_id=correlation_id, reason=reason)

    async def close_context(
        self,
        identity,
        *,
        request_id: str | None = None,
        timeout: float = 5.0,
    ):
        """Release a committed context whose result has no public owner."""

        request = ActionRequest(
            state=None,
            request_id=request_id,
            timeout=timeout,
            prefix_context=PrefixContextRef(
                op=PrefixContextOp.CLOSE,
                identity=identity,
            ),
        )
        result = await self.submit(
            ActionContextTask(action_request=request),
            timeout=timeout,
        )
        if not result.ok:
            raise RuntimeError(
                result.error_info or "GenerationServer rejected context cleanup"
            )
        return result

    async def _release_undelivered_result(self, result) -> None:
        if not (
            isinstance(result, ActionResult)
            and result.ok
            and result.committed_new_context
            and result.prefix_context is not None
        ):
            return
        await self.close_context(
            result.prefix_context,
            request_id=result.request_id,
        )

    def _handle_stale_result(self, result) -> None:
        if not result.committed_new_context:
            return
        cleanup = asyncio.create_task(self._release_undelivered_result(result))
        cleanup.add_done_callback(self._consume_background_task)

    @staticmethod
    def _consume_background_task(task: asyncio.Task) -> None:
        try:
            task.result()
        except BaseException:
            logger.exception("failed to release an undelivered action context")
