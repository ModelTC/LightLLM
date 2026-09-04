import asyncio
import json
from types import SimpleNamespace

import pytest
from easydict import EasyDict

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, PDManager


def test_pd_node_self_request_limit_cli_defaults_to_disabled_and_can_be_enabled():
    parser = make_argument_parser()

    assert parser.parse_args([]).enable_pd_node_self_request_limit is False
    assert parser.parse_args(["--enable_pd_node_self_request_limit"]).enable_pd_node_self_request_limit is True
    assert StartArgs().enable_pd_node_self_request_limit is False


def test_pd_cache_high_priority_cli_defaults_to_enabled_and_can_be_disabled():
    parser = make_argument_parser()

    assert parser.parse_args([]).disable_pd_cache_high_priority is False
    assert parser.parse_args(["--disable_pd_cache_high_priority"]).disable_pd_cache_high_priority is True
    assert StartArgs().disable_pd_cache_high_priority is False


def test_pd_master_request_slot_is_reserved_noop():
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)

    asyncio.run(manager._wait_for_pd_master_request_slot())


def test_auto_set_response_parsers_from_qwen35_model_config(tmp_path):
    from lightllm.utils.config_utils import auto_set_response_parsers

    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
    args = StartArgs(model_dir=str(tmp_path))

    auto_set_response_parsers(args)

    assert args.tool_call_parser == "qwen3_coder"
    assert args.reasoning_parser == "qwen3"


def test_auto_set_response_parsers_preserves_explicit_values(tmp_path):
    from lightllm.utils.config_utils import auto_set_response_parsers

    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
    args = StartArgs(
        model_dir=str(tmp_path),
        tool_call_parser="llama3",
        reasoning_parser="deepseek-r1",
    )

    auto_set_response_parsers(args)

    assert args.tool_call_parser == "llama3"
    assert args.reasoning_parser == "deepseek-r1"


def test_get_server_info_serializes_runtime_easydict(monkeypatch):
    from lightllm.server import api_http

    runtime_args = EasyDict(
        model_name="qwen35_0.8b",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
    )
    monkeypatch.setattr(api_http.g_objs, "args", runtime_args)

    assert api_http.get_server_info() == dict(runtime_args)


def test_get_server_info_serializes_start_args_dataclass(monkeypatch):
    from lightllm.server import api_http

    start_args = StartArgs(model_name="qwen35_0.8b")
    monkeypatch.setattr(api_http.g_objs, "args", start_args)

    server_info = api_http.get_server_info()
    assert server_info["model_name"] == "qwen35_0.8b"
    assert server_info["run_mode"] == "normal"


def test_pd_master_models_endpoint_has_created_timestamp(monkeypatch):
    from lightllm.server import api_http

    args = StartArgs(run_mode="pd_master", model_dir="/tmp/test-model", model_name="test-model")
    manager = SimpleNamespace(get_real_supported_max_req_total_len=lambda: 1024)
    monkeypatch.setattr(api_http, "init_tokenizer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_http.SamplingParams, "load_generation_cfg", lambda _model_dir: None)
    monkeypatch.setattr(api_http.CompletionRequest, "load_generation_cfg", lambda _model_dir: None)
    monkeypatch.setattr(api_http.ChatCompletionRequest, "load_generation_cfg", lambda _model_dir: None)
    monkeypatch.setattr(api_http, "MetricClient", lambda _port: object())
    monkeypatch.setattr(api_http, "get_shm_port_args", lambda: SimpleNamespace(metric_port=1234))
    monkeypatch.setattr(api_http, "HttpServerManagerForPDMaster", lambda args: manager)
    monkeypatch.setattr(api_http, "get_unique_server_name", lambda: "test-server")
    monkeypatch.setattr(api_http.setproctitle, "setproctitle", lambda _title: None)
    monkeypatch.setattr(api_http.time, "time", lambda: 1234.9)

    global_objs = api_http.G_Objs()
    global_objs.set_args(args)
    monkeypatch.setattr(api_http, "g_objs", global_objs)

    response = asyncio.run(api_http.get_models(None))
    assert response.data[0].created == 1234


def test_elastic_pd_nodes_are_ready_with_at_least_one_node_of_each_role():
    manager = PDManager(StartArgs(pd_master_mode="elastic"))
    assert manager.is_pd_nodes_ready() is False

    manager.prefill_nodes = [object()]
    assert manager.is_pd_nodes_ready() is False

    manager.decode_nodes = [object()]
    assert manager.is_pd_nodes_ready() is True

    manager.prefill_nodes.append(object())
    manager.decode_nodes.append(object())
    assert manager.is_pd_nodes_ready() is True


def test_fixed_pd_nodes_are_ready_only_with_exact_node_counts():
    manager = PDManager(StartArgs(pd_master_mode="2p1d"))
    assert manager.is_pd_nodes_ready() is False

    manager.prefill_nodes = [object(), object()]
    manager.decode_nodes = [object()]
    assert manager.is_pd_nodes_ready() is True

    manager.decode_nodes.append(object())
    assert manager.is_pd_nodes_ready() is False


@pytest.mark.parametrize("mode", ["flex", "xpxd", "1p1"])
def test_malformed_pd_master_mode_is_not_ready(mode):
    manager = PDManager(StartArgs(pd_master_mode=mode))
    assert manager.is_pd_nodes_ready() is False


def test_pd_master_readiness_endpoint_uses_pd_node_readiness(monkeypatch):
    from lightllm.server import api_http

    args = StartArgs(run_mode="pd_master", pd_master_mode="elastic")
    pd_manager = PDManager(args)
    monkeypatch.setattr(api_http.g_objs, "args", args)
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", SimpleNamespace(pd_manager=pd_manager))

    response = api_http.readiness()
    assert response.status_code == 503
    assert json.loads(response.body) == {"status": "not ready"}

    pd_manager.prefill_nodes = [object()]
    pd_manager.decode_nodes = [object()]
    response = api_http.readiness()
    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "ok"}


def test_pd_manager_checks_all_connected_node_health(monkeypatch):
    from lightllm.server.httpserver_for_pd_master import manager as manager_module

    requested_urls = []
    responses = {
        "http://10.0.0.1:8000/health": SimpleNamespace(status_code=200),
        "http://10.0.0.2:8000/health": SimpleNamespace(status_code=200),
    }

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            assert kwargs == {"timeout": 8, "trust_env": False}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        async def get(self, url):
            requested_urls.append(url)
            result = responses[url]
            if isinstance(result, BaseException):
                raise result
            return result

    monkeypatch.setattr(manager_module.httpx, "AsyncClient", FakeAsyncClient)

    manager = PDManager(StartArgs())
    manager.prefill_nodes = [SimpleNamespace(mode="prefill", client_ip_port="10.0.0.1:8000")]
    manager.decode_nodes = [SimpleNamespace(mode="decode", client_ip_port="10.0.0.2:8000")]

    assert asyncio.run(manager.check_pd_nodes_health()) is True
    assert set(requested_urls) == set(responses)

    responses["http://10.0.0.2:8000/health"] = SimpleNamespace(status_code=503)
    assert asyncio.run(manager.check_pd_nodes_health()) is False

    responses["http://10.0.0.2:8000/health"] = RuntimeError("connection failed")
    assert asyncio.run(manager.check_pd_nodes_health()) is False


def test_pd_manager_without_connected_nodes_is_healthy():
    manager = PDManager(StartArgs())
    assert asyncio.run(manager.check_pd_nodes_health()) is True


def test_prefill_registration_preserves_existing_inflight_prompt_chars():
    args = StartArgs()
    manager = PDManager(args)

    def register_prefill(node_id, client_ip_port):
        manager.register_pd(
            {
                "node_id": node_id,
                "client_ip_port": client_ip_port,
                "mode": "prefill",
                "start_args": {
                    "max_req_total_len": args.max_req_total_len,
                    "max_image_pixels": args.max_image_pixels,
                    "disable_image_resize": args.disable_image_resize,
                },
            },
            websocket=object(),
        )

    register_prefill(1, "10.0.0.1:8000")
    manager.prefill_nodes[0].dispatched_prompt_chars = 1234
    manager.prefill_nodes[0].dispatched_req_num = 12

    register_prefill(2, "10.0.0.2:8000")

    assert [node.dispatched_prompt_chars for node in manager.prefill_nodes] == [1234, 0]
    assert [node.dispatched_req_num for node in manager.prefill_nodes] == [12, 0]


def test_prefill_reconnection_preserves_other_nodes_inflight_prompt_chars():
    args = StartArgs()
    manager = PDManager(args)

    def pd_info(node_id, client_ip_port):
        return {
            "node_id": node_id,
            "client_ip_port": client_ip_port,
            "mode": "prefill",
            "start_args": {
                "max_req_total_len": args.max_req_total_len,
                "max_image_pixels": args.max_image_pixels,
                "disable_image_resize": args.disable_image_resize,
            },
        }

    manager.register_pd(pd_info(1, "10.0.0.1:8000"), websocket=object())
    manager.register_pd(pd_info(2, "10.0.0.2:8000"), websocket=object())
    manager.prefill_nodes[0].dispatched_prompt_chars = 100
    manager.prefill_nodes[1].dispatched_prompt_chars = 200
    manager.prefill_nodes[0].dispatched_req_num = 1
    manager.prefill_nodes[1].dispatched_req_num = 2

    manager.register_pd(pd_info(3, "10.0.0.2:8000"), websocket=object())

    assert [node.client_ip_port for node in manager.prefill_nodes] == ["10.0.0.1:8000", "10.0.0.2:8000"]
    assert [node.dispatched_prompt_chars for node in manager.prefill_nodes] == [100, 0]
    assert [node.dispatched_req_num for node in manager.prefill_nodes] == [1, 0]


def test_pd_master_inference_health_matches_normal_node_semantics(monkeypatch):
    monkeypatch.setattr("lightllm.server.httpserver_for_pd_master.manager.time.time", lambda: 1000)
    manager = SimpleNamespace(
        health_timeout=200,
        latest_success_infer_time=900,
        running_request_count=1,
        req_id_to_out_inf={1: object()},
    )

    assert HttpServerManagerForPDMaster.is_healthy(manager) is True

    manager.latest_success_infer_time = 700
    assert HttpServerManagerForPDMaster.is_healthy(manager) is False

    manager.running_request_count = 0
    manager.req_id_to_out_inf.clear()
    assert HttpServerManagerForPDMaster.is_healthy(manager) is True


def test_pd_master_restores_request_count_when_preload_fails():
    class FailingMultimodalParams:
        async def verify_and_preload(self, request):
            raise RuntimeError("preload failed")

    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.running_request_count = 0

    async def consume_generate():
        async for _ in manager.generate("prompt", None, FailingMultimodalParams(), None):
            pass

    with pytest.raises(RuntimeError, match="preload failed"):
        asyncio.run(consume_generate())

    assert manager.running_request_count == 0


def test_pd_master_request_count_covers_async_generator_lifecycle():
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.running_request_count = 0
    inner_generator_closed = False

    async def fake_generate(prompt, sampling_params, multimodal_params, request):
        nonlocal inner_generator_closed
        try:
            assert manager.running_request_count == 1
            yield "result"
        finally:
            inner_generator_closed = True

    manager._generate = fake_generate

    async def consume_one_result_and_close():
        results_generator = manager.generate(None, None, None, None)
        assert await results_generator.__anext__() == "result"
        assert manager.running_request_count == 1
        await results_generator.aclose()

    asyncio.run(consume_one_result_and_close())
    assert manager.running_request_count == 0
    assert inner_generator_closed is True


def test_fixed_pd_master_health_endpoint_combines_ready_and_health_status(monkeypatch):
    from lightllm.server import api_http

    args = StartArgs(run_mode="pd_master", pd_master_mode="2p1d")
    pd_manager = PDManager(args)
    pd_node_health_results = iter([True, False])
    pd_node_health_check_count = 0

    async def check_pd_nodes_health():
        nonlocal pd_node_health_check_count
        pd_node_health_check_count += 1
        return next(pd_node_health_results)

    pd_manager.check_pd_nodes_health = check_pd_nodes_health
    httpserver_manager = SimpleNamespace(pd_manager=pd_manager, is_healthy=lambda: True)
    monkeypatch.setattr(api_http.g_objs, "args", args)
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", httpserver_manager)

    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "message": "Error",
        "pd_master_mode": "2p1d",
        "inference_healthy": True,
        "pd_nodes_ready": False,
        "pd_nodes_healthy": False,
        "registered_prefill_nodes": 0,
        "registered_decode_nodes": 0,
    }
    assert pd_node_health_check_count == 0

    pd_manager.prefill_nodes = [object(), object()]
    pd_manager.decode_nodes = [object()]
    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == 200
    assert json.loads(response.body)["message"] == "Ok"
    assert json.loads(response.body)["pd_nodes_ready"] is True
    assert pd_node_health_check_count == 1

    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == 503
    assert json.loads(response.body)["pd_nodes_healthy"] is False
    assert pd_node_health_check_count == 2

    httpserver_manager.is_healthy = lambda: False
    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == 503
    assert json.loads(response.body)["inference_healthy"] is False
    assert pd_node_health_check_count == 2


def test_elastic_pd_master_health_does_not_check_individual_pd_nodes(monkeypatch):
    from lightllm.server import api_http

    args = StartArgs(run_mode="pd_master", pd_master_mode="elastic")
    pd_manager = PDManager(args)
    pd_manager.prefill_nodes = [object()]
    pd_manager.decode_nodes = [object()]

    async def unexpected_pd_node_health_check():
        raise AssertionError("elastic mode must not check individual PD node health")

    pd_manager.check_pd_nodes_health = unexpected_pd_node_health_check
    httpserver_manager = SimpleNamespace(pd_manager=pd_manager, is_healthy=lambda: True)
    monkeypatch.setattr(api_http.g_objs, "args", args)
    monkeypatch.setattr(api_http.g_objs, "httpserver_manager", httpserver_manager)

    response = asyncio.run(api_http.healthcheck(None))
    response_body = json.loads(response.body)
    assert response.status_code == 200
    assert response_body["message"] == "Ok"
    assert "pd_nodes_healthy" not in response_body
