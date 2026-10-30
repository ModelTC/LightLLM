import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from lightllm.utils import log_utils


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("progress", [False, True])
def test_weight_loading_events_and_progress(monkeypatch, caplog, rank, progress):
    from lightllm.common.basemodel.layer_weights import hf_load_utils as module
    import multiprocessing.pool

    pool = Mock()
    pool.__enter__ = Mock(return_value=pool)
    pool.__exit__ = Mock(return_value=False)
    pool.imap_unordered.return_value = iter([None, None])
    monkeypatch.setattr(multiprocessing.pool, "ThreadPool", Mock(return_value=pool))
    monkeypatch.setattr(module.utils.PetrelHelper, "list", lambda *a, **k: ["a.bin", "b.bin"])
    monkeypatch.setattr(module, "progress_enabled", lambda: progress)
    bar = Mock(side_effect=lambda iterator, **kw: iterator)
    monkeypatch.setattr(module, "tqdm", bar)
    logger = logging.getLogger("weight-regression")
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(log_utils, "_context", {"tp_rank": rank})
    monkeypatch.setattr(log_utils, "_context_pid", log_utils.os.getpid())
    ticks = iter([10, 12.125])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        module.load_hf_weights("fp16", "/unused")
    assert bar.call_args.kwargs["disable"] is (not progress)
    records = [r for r in caplog.records if r.name == logger.name]
    assert [r.event for r in records] == ["model.weights_loading", "model.weights_loaded"]
    assert [r.levelno for r in records] == [logging.INFO if rank == 0 else logging.DEBUG] * 2
    assert records[1].duration_s == 2.125
    assert "duration_s=2.125" in records[1].getMessage()


@pytest.mark.parametrize("tp,dp,node,rank", [(4, 1, 0, 3), (8, 2, 1, 4), (8, 4, 1, 7)])
def test_model_infer_rank_context(monkeypatch, tp, dp, node, rank):
    from lightllm.server.router.model_infer import model_rpc as module

    context = Mock()
    monkeypatch.setattr(module, "set_log_context", context)
    monkeypatch.setattr(module, "graceful_registry", Mock())
    monkeypatch.setattr(module.setproctitle, "setproctitle", Mock())
    monkeypatch.setattr(module, "get_unique_server_name", lambda: "test")
    monkeypatch.setattr(module, "start_parent_check_thread", Mock())
    monkeypatch.setattr(module, "ModelRpcServer", Mock())
    server = Mock()
    monkeypatch.setattr(module, "ThreadedServer", Mock(return_value=server))
    args = SimpleNamespace(tp=tp, dp=dp, node_rank=node)
    module._init_env(args, rank, 0, 4, None, "/unused", Mock())
    context.assert_called_once_with(
        component="model_infer", node_rank=node, global_rank=rank, tp_rank=rank % (tp // dp), dp_rank=rank // (tp // dp)
    )
    server.start.assert_called_once()


def test_detoken_slow_batches_flush_on_idle_and_completion(monkeypatch):
    from lightllm.server.detokenization import manager as module

    manager = module.DeTokenizationManager.__new__(module.DeTokenizationManager)
    manager.zmq_recv_socket = Mock()
    manager.zmq_recv_socket.recv_pyobj.side_effect = module.zmq.ZMQError()
    manager.gen_token_out = Mock(side_effect=[True, False, RuntimeError("stop")])
    stats = Mock()
    monkeypatch.setattr(module, "DebugStatsLogger", Mock(return_value=stats))
    ticks = iter([0, 0.06, 1, 1.07, 2])
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: next(ticks), sleep=Mock()))
    manager.handle_loop()
    assert stats.add.call_count == 2
    assert stats.add.call_args_list[0].kwargs["duration_ms_total"] == pytest.approx(60)
    assert stats.add.call_args_list[1].kwargs["duration_ms_total"] == pytest.approx(70)
    stats.flush.assert_called_once()
    req = SimpleNamespace(request_id=7, can_released_mark=False)
    decode = SimpleNamespace(req=req, request_id=7, output_ids=[1, 2], can_set_release_mark=lambda: True)
    manager.req_id_to_out = {7: decode}
    manager.shm_req_manager = Mock()
    event = Mock()
    monkeypatch.setattr(module, "log_debug_event", event)
    manager.remove_finished_reqs()
    assert req.can_released_mark and not manager.req_id_to_out
    manager.shm_req_manager.put_back_req_obj.assert_called_once_with(req)
    event.assert_called_once_with(module.logger, "detoken.completed", request_id=7, output_tokens=2)


def test_recycle_pending_is_bounded_and_releases_each_choice(monkeypatch):
    from lightllm.server.httpserver import manager as module

    manager = module.HttpServerManager.__new__(module.HttpServerManager)
    manager.req_id_to_out_inf = {}
    manager.shm_req_manager = SimpleNamespace(async_put_back_req_obj=AsyncMock(), async_release_req_index=AsyncMock())
    manager._release_multimodal_resources = AsyncMock()
    events = Mock()
    monkeypatch.setattr(module, "log_debug_event", events)
    clock = [0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    reqs = [SimpleNamespace(index_in_shm_mem=i, can_released_mark=False, ref_count=2) for i in range(10)]
    statuses = {
        i: SimpleNamespace(
            can_release=lambda: False,
            group_req_objs=SimpleNamespace(group_req_id=i, shm_req_objs=[req], multimodal_params=None),
        )
        for i, req in enumerate(reqs)
    }

    class StopLoop(BaseException):
        pass

    step = [0]

    async def wait():
        step[0] += 1
        if step[0] == 1:
            clock[0] = 121
        elif step[0] == 2:
            manager.req_id_to_out_inf.update(statuses)
            clock[0] = 122
        elif step[0] == 3:
            clock[0] = 123
        elif step[0] == 4:
            manager.req_id_to_out_inf[0].can_release = lambda: True
        else:
            raise StopLoop()

    manager.recycle_event = SimpleNamespace(wait=wait, clear=Mock())
    with pytest.raises(StopLoop):
        asyncio.run(manager.recycle_resource_loop())
    pending = [c.kwargs for c in events.call_args_list if c.args[1] == "request.pending"]
    assert len(pending) == 1
    assert pending[0]["count"] == 10
    assert pending[0]["request_id_sample"] == list(range(8))
    assert pending[0]["can_release_sample"] == [False] * 8
    assert pending[0]["ref_count_sample"] == [2] * 8
    released = [c for c in events.call_args_list if c.args[1] == "request.released"]
    assert len(released) == 1 and released[0].kwargs == {"request_id": 0, "choices": 1}
    manager.shm_req_manager.async_put_back_req_obj.assert_awaited_once_with(reqs[0])
    manager.shm_req_manager.async_release_req_index.assert_awaited_once_with(0)


@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_dp_shared_kv_transfer_stats(monkeypatch, caplog, master, level):
    from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans as module

    logger = logging.getLogger("dp-transfer-regression")
    worker = module.DPKVSharedMoudle.__new__(module.DPKVSharedMoudle)
    copy = Mock()
    worker.backend = SimpleNamespace(
        is_master_in_dp=master,
        model=SimpleNamespace(mem_manager=SimpleNamespace(operator=SimpleNamespace(copy_kv_from_other_dp_ranks=copy))),
        mem_managers=[],
        dp_size_in_node=2,
        rank_in_dp=0,
    )
    clock = [0]
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: clock[0])
    worker._debug_transfers = log_utils.DebugStatsLogger(logger, "kv.shared_transfer", interval=10)
    # Preserve tensor math while replacing only the unavailable CUDA placement.
    original_to, original_tensor = torch.Tensor.to, torch.tensor
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, **kw: original_to(tensor, **dict(kw, device="cpu")))
    monkeypatch.setattr(module.torch, "tensor", lambda *a, **kw: original_tensor(*a, **dict(kw, device="cpu")))
    req = SimpleNamespace(cur_kv_len=0, hold_kv_len=6, shm_req=SimpleNamespace())
    task = SimpleNamespace(
        max_kv_len_mem_indexes=original_tensor([1, 2]),
        max_kv_len_dp_rank=1,
        mem_indexes=original_tensor([3, 4]),
        req=req,
    )
    with caplog.at_level(level, logger=logger.name):
        worker.kv_trans([task])
        worker.kv_trans([task])
        worker.kv_trans([task])
        clock[0] = 11
        worker.kv_trans([])
    assert copy.call_count == 3 and req.cur_kv_len == 6
    records = [r for r in caplog.records if r.name == logger.name]
    assert len(records) == 2 * int(master and level == logging.DEBUG)
    if records:
        assert records[0].event == "kv.shared_transfer"
        assert records[1].tokens_total == 4 and records[1].tasks_total == 2
        assert records[0].tokens_total == 2


@pytest.mark.parametrize("dp,nnodes,local_dp", [(1, 1, 1), (4, 2, 2), (1, 2, 1)])
def test_router_opens_launcher_readiness_key(monkeypatch, dp, nnodes, local_dp):
    from lightllm.server.router import manager as module
    from lightllm.server.core.objs import StartArgs

    flags = {}

    def shared(name):
        return flags.setdefault(name, Mock())

    class StopInit(Exception):
        pass

    monkeypatch.setattr(module, "SharedInt", shared)
    monkeypatch.setattr(module, "ShmReqManager", Mock())
    counters = Mock()
    monkeypatch.setattr(module, "EngineCounters", counters)
    monkeypatch.setattr(module, "ReadOnlyStaticsMemoryManager", Mock(side_effect=StopInit))
    args = StartArgs()
    args.tp = 8
    args.dp, args.nnodes = dp, nnodes
    args.node_rank = 0
    with pytest.raises(StopInit):
        module.RouterManager(args)
    counters.assert_called_once_with(local_dp, owner=True)
    assert "service_ready" in flags
    flags["service_ready"].set_value.assert_called_once_with(0)


@pytest.mark.parametrize("progress", [False, True])
def test_model_warmup_progress_and_rank_events(monkeypatch, caplog, progress):
    from lightllm.common.basemodel import basemodel as module

    model = module.TpPartBaseModel.__new__(module.TpPartBaseModel)
    model.batch_max_tokens = 4
    model.layers_num = 8
    model.config = {}
    logger = logging.getLogger("warmup-regression")
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module, "progress_enabled", lambda: progress)
    bar = Mock(return_value=[])
    monkeypatch.setattr(module, "tqdm", bar)
    barrier = Mock()
    monkeypatch.setattr(module.torch.distributed, "barrier", barrier)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(module.Autotuner, "start_autotune_warmup", Mock())
    monkeypatch.setattr(module.Autotuner, "end_autotune_warmup", Mock())
    monkeypatch.setattr(log_utils, "_context", {"tp_rank": 0})
    monkeypatch.setattr(log_utils, "_context_pid", log_utils.os.getpid())
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        model._autotune_warmup()
    assert bar.call_args.args[0] == [4, 1]
    assert bar.call_args.kwargs["disable"] is (not progress)
    assert [c.kwargs for c in barrier.call_args_list] == [{"device_ids": [3]}] * 2
    records = [r for r in caplog.records if r.name == logger.name]
    assert [r.event for r in records] == ["model.warmup_started", "model.warmup_completed"]
    assert records[-1].duration_s >= 0
    assert model.layers_num == 8


def test_graceful_registry_sets_component_without_replacing_other_context(monkeypatch):
    from lightllm.utils import graceful_utils
    import signal

    monkeypatch.setattr(signal, "signal", Mock())
    context = Mock()
    monkeypatch.setattr(log_utils, "set_log_context", context)
    graceful_utils.graceful_registry("detokenization")
    context.assert_called_once_with(component="detokenization")


def test_http_disconnect_remains_499_and_debug(monkeypatch):
    from lightllm.server import api_http
    from lightllm.utils.error_utils import ClientDisconnected

    monkeypatch.setattr(api_http, "get_env_start_args", lambda: SimpleNamespace(run_mode="normal"))
    monkeypatch.setattr(api_http.g_objs, "g_generate_func", AsyncMock(side_effect=ClientDisconnected(8, "closed")))
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", Mock())
    logger = Mock()
    monkeypatch.setattr(api_http, "logger", logger)
    response = asyncio.run(api_http.generate(Mock()))
    assert response.status_code == 499
    logger.debug.assert_called_once()
    logger.error.assert_not_called()
    logger.warning.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_access_middleware_uses_real_output_switch(monkeypatch, caplog, enabled):
    from lightllm.server import api_http

    logger = logging.getLogger("lightllm.access")
    monkeypatch.setattr(logger, "parent", logging.getLogger())
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setenv("LIGHTLLM_LOG_HTTP_ACCESS", str(int(enabled)))

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    messages = []

    async def send(message):
        messages.append(message)

    with caplog.at_level(logging.INFO, logger=logger.name):
        asyncio.run(
            api_http._AccessLogMiddleware(app)({"type": "http", "method": "POST", "path": "/generate"}, None, send)
        )
    assert messages[-1]["body"] == b"ok"
    records = [r for r in caplog.records if r.name == logger.name]
    assert len(records) == int(enabled)
    if records:
        assert records[0].status == 201


@pytest.mark.parametrize("header,expected", [({"X-Request-Id": "client-id"}, "client-id"), ({}, "context-id")])
def test_pd_header_logging_prefers_header_then_context(monkeypatch, header, expected):
    from lightllm.server.httpserver_for_pd_master import manager as module

    manager = module.HttpServerManagerForPDMaster.__new__(module.HttpServerManagerForPDMaster)
    event = Mock()
    monkeypatch.setattr(module, "log_debug_event", event)
    monkeypatch.setattr(module, "get_http_request_id", lambda: "context-id")
    asyncio.run(manager._log_req_header(SimpleNamespace(headers=header), 80))
    event.assert_called_once_with(module.logger, "request.received", request_id=80, http_request_id=expected)


def test_allreduce_probe_failure_has_warning_and_debug_traceback(monkeypatch):
    from lightllm.utils import dist_check_utils as module
    import torch.multiprocessing

    logger = Mock()
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(torch.multiprocessing, "spawn", Mock(side_effect=RuntimeError("probe failed")))
    assert not module._check_ok_two_gpu_all_reduce(lambda rank, port: None, 1234)
    logger.warning.assert_called_once()
    assert "event=allreduce.probe_failed" in logger.warning.call_args.args[0]
    assert "probe failed" in logger.warning.call_args.args[2]
    logger.debug.assert_called_once_with("All-reduce capability probe diagnostics", exc_info=True)


def test_invalid_sse_payload_is_skipped_with_bounded_metadata(monkeypatch):
    from lightllm.server import api_responses as module

    event = Mock()
    monkeypatch.setattr(module, "log_debug_event", event)

    async def chunks():
        yield b"data: not-json\n\n"
        yield b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'

    async def collect():
        return [item async for item in module._openai_sse_to_responses_events(chunks(), {"input": "hi"})]

    output = asyncio.run(collect())
    assert any(b"hello" in item for item in output)
    event.assert_called_once_with(module.logger, "sse.payload_skipped", payload_chars=8, reason="invalid_json")


def test_decode_status_connection_errors_are_deduplicated_and_cancel_is_debug(monkeypatch, caplog):
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import up_status as module
    from collections import OrderedDict

    manager = module.UpStatusManager.__new__(module.UpStatusManager)
    manager.id_to_handle_queue = {}
    from unittest.mock import MagicMock

    connection = MagicMock()
    connection.__aenter__ = AsyncMock(
        side_effect=[RuntimeError("offline"), RuntimeError("offline"), asyncio.CancelledError()]
    )
    connection.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(module.websockets, "connect", Mock(return_value=connection))
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(log_utils, "_exception_windows", OrderedDict())
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: 0)
    logger = logging.getLogger("status-connection-regression")
    monkeypatch.setattr(module, "logger", logger)
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        asyncio.run(manager.up_kv_status_task(SimpleNamespace(host_ip_port="master:8000", node_id=1)))
    records = [r for r in caplog.records if r.name == logger.name]
    failures = [r for r in records if r.levelno == logging.ERROR]
    assert len(failures) == 1 and failures[0].event == "pd.status_connection_failed"
    assert failures[0].exc_info is not None
    assert records[-1].levelno == logging.DEBUG and "cancelled" in records[-1].getMessage()


@pytest.mark.parametrize("overlap", [False, True])
def test_cuda_graph_replay_cost_barrier_uses_current_device(monkeypatch, overlap):
    from lightllm.common.basemodel import cuda_graph as module

    graph = module.CudaGraph.__new__(module.CudaGraph)
    graph.capture_infer_cost = True
    graph.enable_decode_microbatch_overlap = overlap
    graph.infer_cost_ms_by_batch_size = {}
    barrier = Mock()
    monkeypatch.setattr(module.dist, "barrier", barrier)
    monkeypatch.setattr(module.dist, "all_reduce", Mock())
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 3)
    start, end = Mock(), Mock()
    start.elapsed_time.return_value = 2.5
    monkeypatch.setattr(module.torch.cuda, "Event", Mock(side_effect=[start, end]))
    original_tensor = torch.tensor
    monkeypatch.setattr(module.torch, "tensor", lambda *a, **kw: original_tensor(*a, **dict(kw, device="cpu")))
    replay = Mock()
    graph._measure_replay_cost(replay, 4)
    barrier.assert_called_once_with(group=module.dist.group.WORLD, device_ids=[3])
    assert graph.infer_cost_ms_by_batch_size == {8 if overlap else 4: 2.5}
    assert replay.replay.call_count == 2
    end.synchronize.assert_called_once()


@pytest.mark.parametrize("rank", [0, 1])
def test_memory_manager_selection_logs_only_on_leader(monkeypatch, caplog, rank):
    from lightllm.common.kv_cache_mem_manager import mem_utils as module
    from lightllm import models

    for name in ("DeepseekV4TpPartModel", "Deepseek3_2TpPartModel", "Deepseek2TpPartModel"):
        monkeypatch.setitem(vars(models), name, type(name, (), {}))
    logger = logging.getLogger("memory-selection-regression")
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module, "get_llm_model_class", lambda: object)
    monkeypatch.setattr(module, "get_env_start_args", lambda: SimpleNamespace(llm_kv_type="None"))
    monkeypatch.setattr(log_utils, "_context", {"tp_rank": rank})
    monkeypatch.setattr(log_utils, "_context_pid", log_utils.os.getpid())
    module.select_mem_manager_class.cache_clear()
    try:
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            assert module.select_mem_manager_class() is module.MemoryManager
        records = [r for r in caplog.records if r.name == logger.name]
        assert len(records) == 2
        assert all(r.levelno == (logging.INFO if rank == 0 else logging.DEBUG) for r in records)
    finally:
        module.select_mem_manager_class.cache_clear()


def test_idle_prefill_selection_records_decision_metadata(monkeypatch):
    from lightllm.server.httpserver_for_pd_master.pd_selector import cache_aware as module

    policy = module.CacheAwarePolicy.__new__(module.CacheAwarePolicy)
    idle = SimpleNamespace(dispatched_req_num=0, dispatched_prompt_chars=0, client_ip_port="idle:8000")
    busy = SimpleNamespace(dispatched_req_num=1, dispatched_prompt_chars=100, client_ip_port="busy:8000")
    event = Mock()
    monkeypatch.setattr(module, "log_debug_event", event)
    assert policy._select_idle_worker([busy, idle]) is idle
    event.assert_called_once_with(
        module.logger, "pd.prefill_selected", reason="idle_random", candidates=1, selected="idle:8000", load=0
    )
    event.reset_mock()
    assert policy._select_idle_worker([busy]) is None
    event.assert_not_called()


def test_openai_stream_disconnect_stops_without_error_chunks(monkeypatch):
    from lightllm.server import api_openai as module
    from lightllm.utils.error_utils import ClientDisconnected

    logger = Mock()
    monkeypatch.setattr(module, "logger", logger)

    async def source():
        yield b"first"
        raise ClientDisconnected(8, "closed")

    async def collect():
        return [item async for item in module._safe_stream_wrapper(source())]

    assert asyncio.run(collect()) == [b"first"]
    logger.debug.assert_called_once()
    logger.error.assert_not_called()


def test_registration_failure_keeps_one_traceback_before_retry(monkeypatch):
    from lightllm.server.httpserver_for_pd_master import register_loop as module
    from unittest.mock import MagicMock

    connection = MagicMock()
    connection.__aenter__ = AsyncMock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr(module.websockets, "connect", Mock(return_value=connection))
    monkeypatch.setattr(module, "get_shm_port_args", lambda: SimpleNamespace(config_server_port=8000))
    logger = Mock()
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError()))
    manager = SimpleNamespace(args=SimpleNamespace(host="10.0.0.1", config_server_host="config"))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(module.register_loop(manager))
    logger.exception.assert_called_once_with("connetion to config_server has error")
    logger.error.assert_not_called()


@pytest.mark.parametrize("debug", [False, True])
def test_router_batch_diagnostics_bound_request_sample(monkeypatch, debug):
    from lightllm.server.router import manager as module

    router = module.RouterManager.__new__(module.RouterManager)
    router.shm_reqs_io_buffer = Mock()
    router.shm_reqs_io_buffer.is_empty.return_value = True
    reqs = [SimpleNamespace(request_id=i, to_router_rpc_obj=lambda: "req") for i in range(12)]
    logger = Mock()
    logger.isEnabledFor.return_value = debug
    event = Mock()
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module, "log_debug_event", event)
    asyncio.run(router._add_batch(SimpleNamespace(reqs=reqs)))
    router.shm_reqs_io_buffer.write_obj.assert_called_once_with(["req"] * 12)
    router.shm_reqs_io_buffer.set_ready.assert_called_once()
    assert all(req.infer_start_time > 0 for req in reqs)
    if debug:
        event.assert_called_once_with(
            logger, "batch.submitted", batch_id=1, request_count=12, request_id_sample=list(range(8))
        )
    else:
        event.assert_not_called()


def test_router_ready_event_uses_agreed_model_capacity(monkeypatch):
    from lightllm.server.router import manager as module
    from lightllm.server.core.objs import StartArgs

    router = module.RouterManager.__new__(module.RouterManager)
    router.args = StartArgs()
    router.args.disable_dynamic_prompt_cache = True
    router.args.run_mode = "normal"
    router.args.max_req_total_len = 1024
    router.args.mtp_step = 0
    router.world_size = router.dp_size = router.dp_size_in_node = 2
    router.nnodes = 1
    router.node_rank = 0
    router.model_weightdir = "/unused"
    router.load_way = "HF"
    router.max_total_token_num = None
    router.shm_max_total_token_num = Mock()
    client = SimpleNamespace(init_model=AsyncMock(), get_max_total_token_num=AsyncMock(return_value=8192))
    monkeypatch.setattr(module.mp, "Queue", Mock())
    monkeypatch.setattr(module, "start_model_process", AsyncMock(return_value=client))
    monkeypatch.setattr(module, "get_running_max_req_size_per_dp", lambda args: 8)
    monkeypatch.setattr(module, "get_shm_port_args", lambda: SimpleNamespace(nccl_port=1234))
    monkeypatch.setattr(module, "build_req_queue", Mock())
    logger = Mock()
    monkeypatch.setattr(module, "logger", logger)
    asyncio.run(router.wait_to_model_ready())
    records = [c.kwargs["extra"] for c in logger.info.call_args_list if "extra" in c.kwargs]
    assert records == [{"event": "kv_cache.ready", "capacity_tokens_per_dp": 8192, "local_dp_count": 2}]
    router.shm_max_total_token_num.set_value.assert_called_once_with(8192)
    assert router.args.max_total_token_num == 8192


@pytest.mark.parametrize(
    "cache_load,reason,selected_load",
    [(120, "cache", 120), (300, "cache_overloaded", 100), (None, "least_load", 100)],
)
def test_busy_prefill_selection_records_decision_metadata(monkeypatch, cache_load, reason, selected_load):
    from lightllm.server.httpserver_for_pd_master.pd_selector import cache_aware as module

    policy = module.CacheAwarePolicy.__new__(module.CacheAwarePolicy)
    policy.config = SimpleNamespace(balance_rel_threshold=1.5)
    least = SimpleNamespace(dispatched_req_num=1, dispatched_prompt_chars=100, client_ip_port="least:8000")
    cache = SimpleNamespace(
        dispatched_req_num=1, dispatched_prompt_chars=cache_load or 300, client_ip_port="cache:8000"
    )
    event = Mock()
    monkeypatch.setattr(module, "log_debug_event", event)
    selected = policy._select_worker_by_cache_and_load([cache, least], cache if cache_load is not None else None, 20)
    assert selected is (cache if reason == "cache" else least)
    event.assert_called_once_with(
        module.logger,
        "pd.prefill_selected",
        reason=reason,
        selected=selected.client_ip_port,
        load=selected_load,
        least_load=100,
        projected_load=selected_load + 20,
        balance_threshold=1.5,
    )
