import asyncio
from types import SimpleNamespace

from lightllm.server.actionserver.objs import ActionTaskIdentity
from lightllm.server.core.objs import FinishStatus
from lightllm.server.inference_runtime import OutputPlan
from lightllm.server.inference_runtime.output_collector import (
    OutputCollector,
    OutputEventKind,
)


def _finish(value=FinishStatus.NO_FINISH):
    return FinishStatus(value)


async def _text_events(events):
    for event in events:
        yield event


def test_text_and_action_preserve_arbitrary_token_delta_order():
    consumed = []
    events = [
        (7, "a", {"id": 1}, _finish()),
        (7, "b", {"id": 2}, _finish()),
        (7, "c", {"id": 3}, _finish(FinishStatus.FINISHED_STOP)),
    ]

    async def run():
        async def action_result():
            await asyncio.sleep(0)
            return {"actions": [[1.0]], "finish_status": "finished"}

        collector = OutputCollector(
            plan=OutputPlan.text_and_action(),
            request_id=7,
            text_events=_text_events(events),
            action_result=action_result,
            mark_output_consumed=lambda: consumed.append(True),
        )
        return [event async for event in collector.collect()]

    output = asyncio.run(run())
    assert [event.text for event in output] == ["a", "b", "c"]
    assert [event.metadata["id"] for event in output] == [1, 2, 3]
    assert all(event.kind is OutputEventKind.TOKEN for event in output)
    assert "action_response" not in output[0].metadata
    assert output[-1].metadata["action_response"]["actions"] == [[1.0]]
    assert output[-1].finish_status.is_finished()
    assert consumed == [True]


def test_action_only_is_a_typed_non_token_event():
    async def run():
        async def action_result():
            return {"actions": [[2.0]], "finish_status": "finished"}

        collector = OutputCollector(
            plan=OutputPlan.action_only(),
            request_id=11,
            text_events=None,
            action_result=action_result,
        )
        return [event async for event in collector.collect()]

    output = asyncio.run(run())
    assert len(output) == 1
    event = output[0]
    assert event.kind is OutputEventKind.ACTION
    assert not event.is_token
    assert event.text == ""
    assert event.finish_status.is_finished()
    assert event.as_legacy_tuple()[2]["output_event"] == "action"


def test_generator_close_discards_output_instead_of_claiming_consumption():
    consumed = []
    discarded = []

    async def run():
        async def action_result():
            return {"actions": [[3.0]], "finish_status": "finished"}

        collector = OutputCollector(
            plan=OutputPlan.action_only(),
            request_id=13,
            text_events=None,
            action_result=action_result,
            mark_output_consumed=lambda: consumed.append(True),
            mark_output_discarded=lambda: discarded.append(True),
        )
        iterator = collector.collect()
        await anext(iterator)
        await iterator.aclose()

    asyncio.run(run())
    assert consumed == []
    assert discarded == [True]


def test_persistent_action_carries_private_owner_token_outside_public_payload():
    owner = ActionTaskIdentity(2, 19, 19)

    async def run():
        async def action_result():
            return SimpleNamespace(
                context_owner_identity=owner,
                to_dict=lambda: {
                    "actions": [[4.0]],
                    "finish_status": "finished",
                },
            )

        collector = OutputCollector(
            plan=OutputPlan.action_only(),
            request_id=19,
            text_events=None,
            action_result=action_result,
        )
        return [event async for event in collector.collect()]

    event = asyncio.run(run())[0]
    assert event.metadata["action_response"] == {
        "actions": [[4.0]],
        "finish_status": "finished",
    }
    assert event.metadata["_action_context_owner_identity"] == owner
