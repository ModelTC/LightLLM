import asyncio
import contextlib

from lightllm.server.core.objs import FinishStatus
from lightllm.server.inference_runtime import OutputPlan
from lightllm.server.inference_runtime.output_collector import OutputCollector


def _finish(value=FinishStatus.NO_FINISH):
    return FinishStatus(value)


async def _text_events(events):
    for event in events:
        yield event


def test_action_terminal_is_consumed_only_after_consumer_resumes_generator():
    consumed = []
    discarded = []

    async def run():
        async def action_result():
            return {"actions": [[1.0]]}

        iterator = OutputCollector(
            plan=OutputPlan.action_only(),
            request_id=17,
            text_events=None,
            action_result=action_result,
            mark_output_consumed=lambda: consumed.append(True),
            mark_output_discarded=lambda: discarded.append(True),
        ).collect()

        terminal = await anext(iterator)
        assert terminal.finish_status.is_finished()
        assert consumed == []
        assert discarded == []

        try:
            await anext(iterator)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("collector emitted an event after aggregate finish")

    asyncio.run(run())
    assert consumed == [True]
    assert discarded == []


def test_closing_after_text_action_terminal_discards_instead_of_consuming():
    consumed = []
    discarded = []
    events = [
        (19, "partial", {"id": 1}, _finish()),
        (19, "done", {"id": 2}, _finish(FinishStatus.FINISHED_STOP)),
    ]

    async def run():
        async def action_result():
            return {"actions": [[2.0]]}

        iterator = OutputCollector(
            plan=OutputPlan.text_and_action(),
            request_id=19,
            text_events=_text_events(events),
            action_result=action_result,
            mark_output_consumed=lambda: consumed.append(True),
            mark_output_discarded=lambda: discarded.append(True),
        ).collect()

        assert (await anext(iterator)).text == "partial"
        terminal = await anext(iterator)
        assert terminal.text == "done"
        assert terminal.metadata["action_response"]["actions"] == [[2.0]]
        assert consumed == []

        await iterator.aclose()

    asyncio.run(run())
    assert consumed == []
    assert discarded == [True]


def test_cancellation_while_waiting_for_action_discards_output():
    consumed = []
    discarded = []

    async def run():
        action_started = asyncio.Event()
        never_finishes = asyncio.Event()

        async def action_result():
            action_started.set()
            await never_finishes.wait()

        iterator = OutputCollector(
            plan=OutputPlan.action_only(),
            request_id=23,
            text_events=None,
            action_result=action_result,
            mark_output_consumed=lambda: consumed.append(True),
            mark_output_discarded=lambda: discarded.append(True),
        ).collect()

        pending_output = asyncio.create_task(anext(iterator))
        await action_started.wait()
        pending_output.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending_output

    asyncio.run(run())
    assert consumed == []
    assert discarded == [True]
