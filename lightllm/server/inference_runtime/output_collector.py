from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from lightllm.server.core.objs import FinishStatus

from .output_plan import OutputPlan


class OutputEventKind(str, Enum):
    TOKEN = "token"
    ACTION = "action"


@dataclass(frozen=True)
class CollectedOutput:
    """One public output event with an explicit token/non-token type."""

    request_id: int
    text: str
    metadata: dict[str, Any]
    finish_status: FinishStatus
    kind: OutputEventKind

    @property
    def is_token(self) -> bool:
        return self.kind is OutputEventKind.TOKEN

    def as_legacy_tuple(self):
        # Preserve ordinary text metadata exactly.  Only the new non-token
        # action event needs an explicit discriminator for legacy tuple users.
        metadata = self.metadata
        if not self.is_token:
            metadata = dict(metadata)
            metadata.setdefault("output_event", self.kind.value)
            metadata.setdefault("is_token", False)
        return self.request_id, self.text, metadata, self.finish_status


class OutputCollector:
    """Join independently finishing text and action branches.

    The collector drains the text iterator exactly in the order supplied by
    detokenization, leaving the existing text decoding strategy unchanged.
    """

    def __init__(
        self,
        *,
        plan: OutputPlan,
        request_id: int,
        text_events: AsyncIterator[tuple[int, str, dict, FinishStatus]] | None,
        action_result: Callable[[], Awaitable[Any]] | None,
        mark_output_consumed: Callable[[], None] | None = None,
        mark_output_discarded: Callable[[], None] | None = None,
    ) -> None:
        self.plan = plan
        self.request_id = request_id
        self.text_events = text_events
        self.action_result = action_result
        self.mark_output_consumed = mark_output_consumed
        self.mark_output_discarded = mark_output_discarded
        self._terminal_delivered = False

        if plan.wants_text != (text_events is not None):
            raise ValueError("text iterator does not match the output plan")
        if plan.wants_action != (action_result is not None):
            raise ValueError("action result callback does not match the output plan")

    @staticmethod
    def _action_dict(action_result: Any) -> dict[str, Any]:
        to_dict = getattr(action_result, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if isinstance(action_result, dict):
            return dict(action_result)
        raise TypeError(f"unsupported action result type: {type(action_result)!r}")

    def _mark_consumed(self) -> None:
        if self._terminal_delivered:
            return
        if self.mark_output_consumed is not None:
            self.mark_output_consumed()
        self._terminal_delivered = True

    async def _close_text_events(self) -> None:
        if self.text_events is None:
            return
        close = getattr(self.text_events, "aclose", None)
        if close is not None:
            with contextlib.suppress(RuntimeError):
                await close()

    async def collect(self) -> AsyncGenerator[CollectedOutput, None]:
        action_task: asyncio.Task | None = None
        if self.action_result is not None:
            action_task = asyncio.create_task(self.action_result())

        try:
            if not self.plan.wants_text:
                assert action_task is not None
                raw_response = await action_task
                response = self._action_dict(raw_response)
                metadata = {"action_response": response}
                context_owner_identity = getattr(
                    raw_response,
                    "context_owner_identity",
                    None,
                )
                if context_owner_identity is not None:
                    metadata["_action_context_owner_identity"] = (
                        context_owner_identity
                    )
                yield CollectedOutput(
                    request_id=self.request_id,
                    text="",
                    metadata=metadata,
                    finish_status=FinishStatus(FinishStatus.FINISHED_STOP),
                    kind=OutputEventKind.ACTION,
                )
                # The consumer has resumed the generator after receiving the
                # terminal payload, so shared output is now safe to recycle.
                self._mark_consumed()
                return

            assert self.text_events is not None
            if not self.plan.wants_action:
                # The legacy text iterator already handles n/best_of and may
                # therefore emit several terminal records.  Pass every record
                # through untouched and only claim ownership after exhaustion.
                async for sub_req_id, text, metadata, finish_status in self.text_events:
                    yield CollectedOutput(
                        request_id=sub_req_id,
                        text=text,
                        metadata=metadata,
                        finish_status=finish_status,
                        kind=OutputEventKind.TOKEN,
                    )
                self._mark_consumed()
                return

            terminal: tuple[int, str, dict, FinishStatus] | None = None
            async for event in self.text_events:
                sub_req_id, text, metadata, finish_status = event
                if finish_status.is_finished():
                    if terminal is not None:
                        raise RuntimeError("text branch emitted more than one terminal event")
                    terminal = event
                    # Aggregate completion must wait for the action branch.
                    if self.plan.wants_action:
                        continue
                else:
                    yield CollectedOutput(
                        request_id=sub_req_id,
                        text=text,
                        metadata=metadata,
                        finish_status=finish_status,
                        kind=OutputEventKind.TOKEN,
                    )

            if terminal is None:
                raise RuntimeError("text branch ended without a terminal event")

            sub_req_id, text, metadata, finish_status = terminal
            metadata = dict(metadata)
            if action_task is not None:
                raw_response = await action_task
                metadata["action_response"] = self._action_dict(raw_response)
                context_owner_identity = getattr(
                    raw_response,
                    "context_owner_identity",
                    None,
                )
                if context_owner_identity is not None:
                    metadata["_action_context_owner_identity"] = (
                        context_owner_identity
                    )
            yield CollectedOutput(
                request_id=sub_req_id,
                text=text,
                metadata=metadata,
                finish_status=finish_status,
                kind=OutputEventKind.TOKEN,
            )
            self._mark_consumed()
        finally:
            try:
                if action_task is not None and not action_task.done():
                    action_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await action_task
            finally:
                try:
                    await self._close_text_events()
                finally:
                    if not self._terminal_delivered and self.mark_output_discarded is not None:
                        self.mark_output_discarded()
