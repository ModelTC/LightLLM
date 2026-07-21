import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.pi0.config import Pi0VLAConfig
from lightllm.server.actionserver.api import _ordered_images
from lightllm.server.actionserver.kv_memory import (
    ActionPrefixContextCache,
    ScopedKVMemoryView,
)
from lightllm.server.actionserver.lifecycle import (
    ActionTaskRegistry,
    TaskSubmissionDecision,
)
from lightllm.server.actionserver.manager import ActionManager, _ManagedActionTask
from lightllm.server.actionserver.objs import (
    ActionContextOwnerDisposition,
    ActionExpertTask,
    ActionOutcome,
    ActionReleaseDecision,
    ActionRequest,
    ActionStatus,
    ActionTaskIdentity,
    ActionWorkerAck,
    PrefixContextIdentity,
)
from lightllm.server.actionserver.shared_store import ActionOutputStore
from lightllm.utils.envs_utils import get_unique_server_name


PI0_DIR = "/mtc/baishihao/vla/lerobot_models_for_vla/pi0_base"


@pytest.fixture
def action_store(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "unit_action_store")
    get_unique_server_name.cache_clear()
    store = ActionOutputStore(
        max_requests=2,
        max_horizon=4,
        max_action_dim=3,
        consumer_ranks=2,
        initialize=True,
    )
    yield store
    for value in vars(store).values():
        if hasattr(value, "close_shm"):
            try:
                value.close_shm()
            except FileNotFoundError:
                pass
    get_unique_server_name.cache_clear()


def test_output_is_recyclable_only_after_every_vlm_rank_consumes_it(action_store):
    identity = ActionTaskIdentity(slot_index=0, generation=1, task_id=101)
    expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    assert action_store.begin_task(identity, horizon=4, action_dim=3)
    assert action_store.set_context_version(identity, 9)
    action_store.write_output(
        0,
        expected,
        timing_ms=(1.0, 2.0, 3.0),
        identity=identity,
    )
    assert action_store.get_status(0) is ActionStatus.HAS_OUTPUT

    action_store.mark_worker_ack(identity, 0)
    with pytest.raises(RuntimeError, match="all action workers"):
        action_store.acknowledge(0, identity=identity)
    action_store.mark_worker_ack(identity, 1)
    assert action_store.get_release_decision(0, identity=identity) is ActionReleaseDecision.RELEASE

    action_store.mark_rank_consumed(0, 0, identity=identity)
    with pytest.raises(RuntimeError, match="all TP ranks"):
        action_store.acknowledge(0, identity=identity)
    action_store.mark_rank_consumed(0, 1, identity=identity)
    action_store.acknowledge(0, identity=identity)
    assert action_store.get_status(0) is ActionStatus.DONE

    with pytest.raises(RuntimeError, match="every target rank retires"):
        action_store.release_slot(identity)
    assert action_store.mark_rank_done_observed(0, 0, identity=identity)
    assert not action_store.mark_rank_retired(0, 0, identity=identity)
    assert action_store.mark_rank_done_observed(0, 1, identity=identity)
    assert action_store.all_ranks_done_observed(0)
    assert action_store.mark_rank_retired(0, 0, identity=identity)
    with pytest.raises(RuntimeError, match="every target rank retires"):
        action_store.release_slot(identity)
    assert action_store.mark_rank_retired(0, 1, identity=identity)
    assert action_store.all_ranks_retired(0)

    response = action_store.read_response(0, "request-0", identity=identity)
    torch.testing.assert_close(response.actions, expected)
    assert response.outcome is ActionOutcome.SUCCESS
    assert response.prefix_context_version == 9
    assert response.policy_timing == {
        "queue_ms": 1.0,
        "action_expert_ms": 2.0,
        "total_ms": 3.0,
    }
    assert action_store.release_slot(identity)
    assert not action_store.release_slot(identity)


def test_persistent_context_slot_waits_for_http_and_every_target_owner_ack(
    action_store,
):
    identity = ActionTaskIdentity(slot_index=0, generation=2, task_id=202)
    assert action_store.begin_task(identity, horizon=1, action_dim=1)
    assert action_store.require_context_owner_ack(identity)
    assert (
        action_store.get_context_owner_disposition(identity)
        is ActionContextOwnerDisposition.PENDING
    )
    action_store.write_output(
        0,
        torch.ones(1, 1),
        timing_ms=(0.0, 0.0, 0.0),
        identity=identity,
    )
    action_store.mark_worker_ack(identity, 0)
    action_store.mark_worker_ack(identity, 1)
    action_store.mark_rank_consumed(0, 0, identity=identity)
    action_store.mark_rank_consumed(0, 1, identity=identity)
    action_store.acknowledge(0, identity=identity)
    action_store.mark_rank_done_observed(0, 0, identity=identity)
    action_store.mark_rank_done_observed(0, 1, identity=identity)
    action_store.mark_rank_retired(0, 0, identity=identity)
    action_store.mark_rank_retired(0, 1, identity=identity)

    with pytest.raises(RuntimeError, match="owner disposition ACK"):
        action_store.release_slot(identity)
    assert action_store.mark_context_owner_disposition(
        identity,
        ActionContextOwnerDisposition.DELIVERED,
    )
    assert action_store.mark_context_owner_rank_acked(identity, 0)
    with pytest.raises(RuntimeError, match="owner disposition ACK"):
        action_store.release_slot(identity)
    assert action_store.mark_context_owner_rank_acked(identity, 1)
    assert action_store.context_owner_release_ready(identity)
    assert action_store.release_slot(identity)


def test_action_kv_view_rejects_writes_outside_router_lease():
    shared = SimpleNamespace(
        kv_buffer=torch.zeros(2, 8, 2, 4),
        req_to_token_indexs=torch.zeros(4, 8, dtype=torch.int32),
        head_num=1,
        head_dim=4,
        layer_num=2,
        dtype=torch.float32,
    )

    def get_att_input_params(layer_index):
        layer = shared.kv_buffer[layer_index]
        return layer[:, : shared.head_num], layer[:, shared.head_num :]

    def copy_kv_to_mem_manager(*, layer_index, mem_index, kv):
        shared.kv_buffer[layer_index, mem_index.long()] = kv

    shared.get_att_input_params = get_att_input_params
    shared.operator = SimpleNamespace(copy_kv_to_mem_manager=copy_kv_to_mem_manager)
    view = ScopedKVMemoryView(shared)
    identity = ActionTaskIdentity(slot_index=0, generation=1, task_id=1)
    view.begin_task_mapping(
        identity=identity,
        target_req_indexes=torch.tensor([0]),
        action_req_indexes=torch.tensor([0]),
        prefix_seq_lens=torch.tensor([1]),
        prefix_mem_indexes=torch.tensor([0]),
        scratch_mem_indexes=torch.tensor([2, 5]),
    )
    lease = view.get_scratch_write_indexes(2)
    value = torch.ones(2, 2, 4)

    view.operator.copy_kv_to_mem_manager(layer_index=0, mem_index=lease, kv=value)
    torch.testing.assert_close(view.kv_buffer[0, lease], value)
    with pytest.raises(RuntimeError, match="outside scratch"):
        view.operator.copy_kv_to_mem_manager(layer_index=0, mem_index=torch.tensor([1, 5]), kv=value)
    view.end_task_mapping(identity)
    with pytest.raises(RuntimeError, match="without a lease"):
        view.operator.copy_kv_to_mem_manager(layer_index=1, mem_index=lease, kv=value)


def test_action_kv_mapping_never_mutates_target_logical_row():
    target_mapping = torch.full((4, 10), -1, dtype=torch.int32)
    target_mapping[1, :5] = torch.tensor([11, 12, 13, 91, 92])
    target_before = target_mapping.clone()
    shared = SimpleNamespace(
        kv_buffer=torch.zeros(2, 32, 2, 4),
        req_to_token_indexs=target_mapping,
        head_num=1,
        head_dim=4,
        layer_num=2,
        dtype=torch.float32,
        get_att_input_params=lambda _: None,
        operator=SimpleNamespace(copy_kv_to_mem_manager=lambda **_: None),
    )
    view = ScopedKVMemoryView(shared)
    identity = ActionTaskIdentity(slot_index=0, generation=3, task_id=99)

    action_rows = view.begin_task_mapping(
        identity=identity,
        target_req_indexes=torch.tensor([1], dtype=torch.int32),
        action_req_indexes=torch.tensor([2], dtype=torch.int32),
        prefix_seq_lens=torch.tensor([3], dtype=torch.int32),
        # This is the committed target-prefix snapshot supplied by the future
        # ActionBranchRuntime, independent of later target-row changes.
        prefix_mem_indexes=torch.tensor([11, 12, 13], dtype=torch.int32),
        scratch_mem_indexes=torch.tensor([21, 22], dtype=torch.int32),
    )

    torch.testing.assert_close(action_rows.cpu(), torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(
        view.req_to_token_indexs[2, :5],
        torch.tensor([11, 12, 13, 21, 22], dtype=torch.int32),
    )
    torch.testing.assert_close(target_mapping, target_before)
    view.end_task_mapping(identity)
    assert torch.all(view.req_to_token_indexs[2] == -1)
    torch.testing.assert_close(target_mapping, target_before)


def test_task_selects_allocator_mappings_for_each_action_tp_rank():
    task = ActionExpertTask(
        request_id=7,
        slot_index=0,
        prefix_req_indexes=torch.tensor([1], dtype=torch.int32),
        prefix_seq_lens=torch.tensor([3], dtype=torch.int32),
        prefix_mem_indexes=torch.tensor([[1, 2, 3], [11, 12, 13]]),
        scratch_mem_indexes=torch.tensor([[4, 5], [14, 15]]),
        action_req_indexes=torch.tensor([1], dtype=torch.int32),
        state=None,
        noisy_actions=torch.zeros(1, 2, 3),
        action_horizon=2,
        action_dim=3,
        num_denoise_steps=1,
        generation=2,
        task_id=8,
    )
    prefix, scratch = task.mappings_for_rank(1)
    torch.testing.assert_close(prefix, torch.tensor([11, 12, 13]))
    torch.testing.assert_close(scratch, torch.tensor([14, 15]))


def test_prefix_context_mapping_is_registered_once_and_hot_ticks_are_constant_size():
    identity = PrefixContextIdentity("robot-7", 2, "epoch-1")
    cache = ActionPrefixContextCache()
    prefix = torch.tensor([11, 12, 13], dtype=torch.int32)
    scratch = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
    lengths = torch.tensor([3], dtype=torch.int32)

    first = cache.register(
        identity,
        prefix_mem_indexes=prefix,
        scratch_mem_indexes=scratch,
        prefix_seq_lens=lengths,
    )
    repeated = cache.register(
        identity,
        prefix_mem_indexes=prefix.clone(),
        scratch_mem_indexes=scratch.clone(),
        prefix_seq_lens=lengths.clone(),
    )
    assert repeated is first

    hot_prefix, hot_scratch = cache.resolve(
        identity,
        prefix_seq_lens=lengths,
        suffix_length=2,
    )
    torch.testing.assert_close(hot_prefix, prefix)
    torch.testing.assert_close(hot_scratch, scratch[:2])

    with pytest.raises(RuntimeError, match="different KV mappings"):
        cache.register(
            identity,
            prefix_mem_indexes=prefix + 1,
            scratch_mem_indexes=scratch,
            prefix_seq_lens=lengths,
        )

    assert cache.release(identity)
    assert not cache.release(identity)
    with pytest.raises(RuntimeError, match="not registered"):
        cache.resolve(
            identity,
            prefix_seq_lens=lengths,
            suffix_length=2,
        )


def test_hot_context_task_omits_physical_mappings():
    task = ActionExpertTask(
        request_id=9,
        slot_index=1,
        prefix_req_indexes=torch.tensor([2], dtype=torch.int32),
        prefix_seq_lens=torch.tensor([3], dtype=torch.int32),
        prefix_mem_indexes=None,
        scratch_mem_indexes=None,
        action_req_indexes=torch.tensor([2], dtype=torch.int32),
        state=torch.zeros(1, 4),
        noisy_actions=torch.zeros(1, 2, 3),
        action_horizon=2,
        action_dim=3,
        num_denoise_steps=1,
        generation=3,
        task_id=10,
        prefix_context_identity=PrefixContextIdentity("robot-7", 2, "epoch-1"),
    )

    with pytest.raises(RuntimeError, match="previously registered"):
        task.mappings_for_rank(0)


def test_action_manager_rejects_hot_task_after_context_retirement():
    context = PrefixContextIdentity("robot-7", 2, "epoch-1")
    task = ActionExpertTask(
        request_id=9,
        slot_index=1,
        prefix_req_indexes=torch.tensor([2], dtype=torch.int32),
        prefix_seq_lens=torch.tensor([3], dtype=torch.int32),
        prefix_mem_indexes=None,
        scratch_mem_indexes=None,
        action_req_indexes=torch.tensor([2], dtype=torch.int32),
        state=torch.zeros(1, 4),
        noisy_actions=torch.zeros(1, 2, 3),
        action_horizon=2,
        action_dim=3,
        num_denoise_steps=1,
        generation=3,
        task_id=10,
        prefix_context_identity=context,
    )

    class Store:
        error = None

        @staticmethod
        def matches(_identity):
            return True

        @staticmethod
        def get_status(_slot):
            return ActionStatus.RUNNING

        def write_dispatch_error(self, identity, error):
            self.error = (identity, error)
            return True

    manager = ActionManager.__new__(ActionManager)
    manager.task_registry = ActionTaskRegistry()
    manager.output_store = Store()
    manager._retired_context_versions = {("epoch-1", "robot-7"): 2}

    assert manager._accept_task(task) is None
    assert manager.output_store.error[0] == task.identity
    assert "already retired" in manager.output_store.error[1]

    manager.task_registry = ActionTaskRegistry()
    manager.output_store = Store()
    manager._retired_context_versions = {}
    manager._registered_contexts = set()
    assert manager._accept_task(task) is None
    assert "not registered" in manager.output_store.error[1]


def test_action_manager_activates_context_only_after_every_worker_succeeds():
    context = PrefixContextIdentity("robot-7", 2, "epoch-1")
    task = ActionExpertTask(
        request_id=9,
        slot_index=1,
        prefix_req_indexes=torch.tensor([2], dtype=torch.int32),
        prefix_seq_lens=torch.tensor([3], dtype=torch.int32),
        prefix_mem_indexes=torch.tensor([[11, 12, 13], [21, 22, 23]]),
        scratch_mem_indexes=torch.tensor([[31, 32], [41, 42]]),
        action_req_indexes=torch.tensor([2], dtype=torch.int32),
        state=torch.zeros(1, 4),
        noisy_actions=torch.zeros(1, 2, 3),
        action_horizon=2,
        action_dim=3,
        num_denoise_steps=1,
        generation=3,
        task_id=10,
        prefix_context_identity=context,
    )

    class Store:
        def __init__(self):
            self.acked = set()
            self.output = None

        def mark_worker_ack(self, identity, rank):
            assert identity == task.identity
            self.acked.add(rank)

        def all_workers_acked(self, _slot):
            return self.acked == {0, 1}

        def write_output(self, slot, actions, *, timing_ms, identity):
            self.output = (slot, actions, timing_ms, identity)
            return True

    manager = ActionManager.__new__(ActionManager)
    manager.action_tp = 2
    manager.output_store = Store()
    manager._registered_contexts = set()
    manager._restart_required = False
    results = [
        ActionWorkerAck(
            slot_index=1,
            generation=3,
            task_id=10,
            rank=0,
            outcome=ActionOutcome.SUCCESS,
            safe_to_release=True,
            actions=torch.ones(2, 3),
        ),
        ActionWorkerAck(
            slot_index=1,
            generation=3,
            task_id=10,
            rank=1,
            outcome=ActionOutcome.SUCCESS,
            safe_to_release=True,
        ),
    ]

    manager._publish_worker_results(
        task,
        results,
        queue_ms=0.0,
        publish_output=True,
    )

    assert context in manager._registered_contexts
    assert manager.output_store.output is not None


def test_action_manager_rechecks_fail_stop_after_waiting_for_model_lock():
    task = ActionExpertTask(
        request_id=9,
        slot_index=1,
        prefix_req_indexes=torch.tensor([2], dtype=torch.int32),
        prefix_seq_lens=torch.tensor([3], dtype=torch.int32),
        prefix_mem_indexes=torch.tensor([[11, 12, 13]], dtype=torch.int32),
        scratch_mem_indexes=torch.tensor([[31, 32]], dtype=torch.int32),
        action_req_indexes=torch.tensor([2], dtype=torch.int32),
        state=torch.zeros(1, 4),
        noisy_actions=torch.zeros(1, 2, 3),
        action_horizon=2,
        action_dim=3,
        num_denoise_steps=1,
        generation=3,
        task_id=10,
    )

    class Store:
        def __init__(self):
            self.restart_required = []

        def mark_restart_required(self, identity, reason):
            self.restart_required.append((identity, reason))
            return True

    async def scenario():
        manager = ActionManager.__new__(ActionManager)
        manager.output_store = Store()
        manager._model_lock = asyncio.Lock()
        manager._restart_required = False
        entered_lock_wait = asyncio.Event()
        worker_dispatches = []

        await manager._model_lock.acquire()

        async def acquire_after_optimistic_check(_record):
            entered_lock_wait.set()
            await manager._model_lock.acquire()
            return True

        async def dispatch_to_workers(_record):
            worker_dispatches.append(_record.task.identity)

        manager._acquire_model_or_control = acquire_after_optimistic_check
        manager._dispatch_to_workers = dispatch_to_workers
        execution = asyncio.create_task(manager._execute(_ManagedActionTask(task)))
        await entered_lock_wait.wait()

        # Simulate the active predecessor losing an ACK before this queued task
        # obtains the model lock.
        manager._restart_required = True
        manager._model_lock.release()
        await execution

        assert worker_dispatches == []
        assert manager.output_store.restart_required
        assert manager.output_store.restart_required[0][0] == task.identity
        assert "process restart is required" in manager.output_store.restart_required[0][1]

    asyncio.run(scenario())


def test_action_manager_latches_fail_stop_before_releasing_lock_on_exception():
    task = ActionExpertTask(
        request_id=10,
        slot_index=1,
        prefix_req_indexes=torch.tensor([2], dtype=torch.int32),
        prefix_seq_lens=torch.tensor([1], dtype=torch.int32),
        prefix_mem_indexes=torch.tensor([[11]], dtype=torch.int32),
        scratch_mem_indexes=torch.tensor([[31]], dtype=torch.int32),
        action_req_indexes=torch.tensor([2], dtype=torch.int32),
        state=torch.zeros(1, 1),
        noisy_actions=torch.zeros(1, 1, 1),
        action_horizon=1,
        action_dim=1,
        num_denoise_steps=1,
        generation=4,
        task_id=11,
    )

    async def scenario():
        manager = ActionManager.__new__(ActionManager)
        manager._model_lock = asyncio.Lock()
        manager._restart_required = False

        async def dispatch_failure(_record):
            raise RuntimeError("worker transport failed")

        manager._dispatch_to_workers = dispatch_failure
        with pytest.raises(RuntimeError, match="transport failed"):
            await manager._execute(_ManagedActionTask(task))

        assert manager._restart_required
        assert not manager._model_lock.locked()

    asyncio.run(scenario())


def test_action_manager_skips_queued_context_release_after_fail_stop():
    context = PrefixContextIdentity("robot-7", 2, "epoch-1")

    class Rpc:
        def __init__(self):
            self.releases = []

        async def release_prefix_context(self, identity):
            self.releases.append(identity)
            return True

    async def scenario():
        manager = ActionManager.__new__(ActionManager)
        manager._model_lock = asyncio.Lock()
        manager._restart_required = False
        rpc = Rpc()
        manager.model_rpcs = [rpc]

        await manager._model_lock.acquire()
        cleanup = asyncio.create_task(manager._release_prefix_context(context))
        await asyncio.sleep(0)
        manager._restart_required = True
        manager._model_lock.release()
        await cleanup

        assert rpc.releases == []

    asyncio.run(scenario())


def test_output_store_rejects_late_result_after_slot_reuse(action_store):
    old = ActionTaskIdentity(slot_index=0, generation=7, task_id=70)
    new = ActionTaskIdentity(slot_index=0, generation=8, task_id=80)
    assert action_store.begin_task(old, horizon=2, action_dim=2)
    action_store.write_error(0, "old", identity=old, workers_safe=True)
    action_store.mark_rank_consumed(0, 0, identity=old)
    action_store.mark_rank_consumed(0, 1, identity=old)
    action_store.acknowledge(0, identity=old)
    action_store.mark_rank_done_observed(0, 0, identity=old)
    action_store.mark_rank_done_observed(0, 1, identity=old)
    action_store.mark_rank_retired(0, 0, identity=old)
    action_store.mark_rank_retired(0, 1, identity=old)
    assert action_store.release_slot(old)
    assert action_store.begin_task(new, horizon=2, action_dim=2)

    assert not action_store.write_output(
        0,
        torch.ones(2, 2),
        timing_ms=(0.0, 0.0, 0.0),
        identity=old,
    )
    assert action_store.get_status(0) is ActionStatus.RUNNING
    assert action_store.get_identity(0) == new


@pytest.mark.parametrize(
    ("outcome", "publish"),
    [
        (
            ActionOutcome.ERROR,
            lambda store, identity: store.write_error(0, "failed", identity=identity, workers_safe=False),
        ),
        (ActionOutcome.TIMEOUT, lambda store, identity: store.write_timeout(identity)),
        (ActionOutcome.ABORTED, lambda store, identity: store.write_abort(identity)),
    ],
)
def test_error_timeout_and_abort_wait_for_worker_ack(action_store, outcome, publish):
    identity = ActionTaskIdentity(slot_index=0, generation=11, task_id=12)
    assert action_store.begin_task(identity, horizon=2, action_dim=2)
    assert publish(action_store, identity)
    assert action_store.get_outcome(0) is outcome
    assert action_store.get_release_decision(0, identity=identity) is ActionReleaseDecision.WAIT
    action_store.mark_worker_ack(identity, 0)
    assert action_store.get_release_decision(0, identity=identity) is ActionReleaseDecision.WAIT
    action_store.mark_worker_ack(identity, 1)
    assert action_store.get_release_decision(0, identity=identity) is ActionReleaseDecision.RELEASE


def test_missing_worker_ack_surfaces_restart_required(action_store):
    identity = ActionTaskIdentity(slot_index=0, generation=13, task_id=14)
    assert action_store.begin_task(identity, horizon=2, action_dim=2)
    action_store.write_timeout(identity)
    action_store.mark_worker_ack(identity, 0)
    action_store.mark_restart_required(identity, "timeout grace expired; restart required")
    assert action_store.get_outcome(0) is ActionOutcome.RESTART_REQUIRED
    assert action_store.get_release_decision(0, identity=identity) is ActionReleaseDecision.RESTART_REQUIRED
    response = action_store.read_terminal_response(identity, "request-14")
    assert response.restart_required
    assert response.outcome is ActionOutcome.RESTART_REQUIRED
    assert "restart required" in response.error_info
    with pytest.raises(RuntimeError, match="restart is required"):
        action_store.acknowledge(0, identity=identity)


def test_task_registry_is_exactly_once_and_rejects_aba():
    registry = ActionTaskRegistry()
    first = ActionTaskIdentity(slot_index=3, generation=5, task_id=50)
    same_generation_other_id = ActionTaskIdentity(slot_index=3, generation=5, task_id=51)
    next_generation = ActionTaskIdentity(slot_index=3, generation=6, task_id=60)

    assert registry.submit(first) is TaskSubmissionDecision.ACCEPT
    assert registry.submit(first) is TaskSubmissionDecision.DUPLICATE
    assert registry.submit(same_generation_other_id) is TaskSubmissionDecision.STALE
    assert registry.mark_terminal(first)
    assert registry.submit(next_generation) is TaskSubmissionDecision.ACCEPT
    assert registry.submit(first) is TaskSubmissionDecision.DUPLICATE


@pytest.mark.skipif(not Path(PI0_DIR).is_dir(), reason="pi0 config is not mounted")
def test_action_request_uses_checkpoint_limits():
    config = Pi0VLAConfig.from_model_dir(PI0_DIR)
    request = ActionRequest(
        state=[0.0] * 32,
        action_horizon=4,
        action_dim=7,
        num_denoise_steps=2,
    )
    request_config = request.validate(config)
    assert (
        request_config.action_horizon,
        request_config.action_dim,
        request_config.num_denoise_steps,
    ) == (4, 7, 2)

    with pytest.raises(ValueError, match="num_denoise_steps=.*server maximum"):
        ActionRequest(
            state=[0.0] * 32,
            num_denoise_steps=config.num_denoise_steps + 1,
        ).validate(config)


def test_camera_mapping_uses_checkpoint_order():
    images = {"right": 3, "base": 1, "left": 2}
    keys = (
        "observation.images.base",
        "observation.images.left",
        "observation.images.right",
    )
    assert _ordered_images(images, keys) == [1, 2, 3]
