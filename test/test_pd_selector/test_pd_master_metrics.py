import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from prometheus_client import generate_latest

from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, PDManager
from lightllm.server.metrics.manager import MetricServer
from lightllm.server.metrics.metrics import Monitor
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.error_utils import ClientDisconnected, ServerBusyError


class RPyCDictProxyLike:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        return self.values[key]


def test_pd_master_exports_node_load_with_role_and_endpoint_labels():
    args = SimpleNamespace(
        metric_gateway=None,
        job_name="lightllm",
        grouping_key=None,
        enable_monitor_auth=False,
        model_name="test-model",
        max_req_total_len=1024,
        mtp_step=0,
        select_p_d_node_strategy="random",
    )
    monitor = Monitor(args)
    metric_client = MagicMock()
    metric_client.gauge_set.side_effect = monitor.gauge_set
    manager = PDManager(args, metric_client)
    manager.url_to_pd_nodes["10.0.0.1:28761"] = PD_Client_Obj(
        node_id=1,
        client_ip_port="10.0.0.1:28761",
        mode="prefill",
        start_args={},
    )
    manager.url_to_pd_nodes["10.0.0.2:28764"] = PD_Client_Obj(
        node_id=2,
        client_ip_port="10.0.0.2:28764",
        mode="decode",
        start_args={},
    )

    manager.update_node_load_info({"client_ip_port": "10.0.0.1:28761", "total_token_usage_rate": 0.25})
    manager.update_node_load_info({"client_ip_port": "10.0.0.2:28764", "total_token_usage_rate": 0.75})
    monitor.gauge_set("lightllm_pd_master_stage_waiting_requests", 2, labels={"stage": "decode"})

    metrics = generate_latest(monitor.registry).decode()
    assert (
        'lightllm_pd_node_token_usage_ratio{endpoint="10.0.0.1:28761",model_name="test-model",role="prefill"} 0.25'
        in metrics
    )
    assert (
        'lightllm_pd_node_token_usage_ratio{endpoint="10.0.0.2:28764",model_name="test-model",role="decode"} 0.75'
        in metrics
    )
    assert 'lightllm_pd_master_stage_waiting_requests{model_name="test-model",stage="decode"} 2.0' in metrics


def test_metric_server_copies_rpyc_label_proxy_before_updating_gauge():
    args = SimpleNamespace(
        metric_gateway=None,
        job_name="lightllm",
        grouping_key=None,
        enable_monitor_auth=False,
        model_name="test-model",
        max_req_total_len=1024,
        mtp_step=0,
        push_interval=10,
    )
    server = MetricServer(args)

    server.exposed_gauge_set(
        "lightllm_pd_master_stage_waiting_requests",
        3,
        RPyCDictProxyLike({"stage": "decode"}),
    )

    metrics = generate_latest(server.monitor.registry).decode()
    assert 'lightllm_pd_master_stage_waiting_requests{model_name="test-model",stage="decode"} 3.0' in metrics


@pytest.mark.parametrize("stage", ["prefill", "decode"])
@pytest.mark.parametrize(
    "outcome",
    [
        None,
        ServerBusyError(),
        asyncio.CancelledError(),
        ClientDisconnected(group_request_id=123),
        RuntimeError("worker failed"),
    ],
    ids=["success", "timeout", "cancellation", "disconnect", "exception"],
)
def test_pd_master_stage_waiting_gauge_is_balanced_for_every_exit(stage, outcome):
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.metric_client = MagicMock()
        manager.pd_stage_waiting_request_counts = {"prefill": 0, "decode": 0}
        manager._wait_for_event_or_disconnect = AsyncMock(side_effect=outcome)

        if outcome is None:
            await manager._wait_for_pd_stage(AsyncMock(), AsyncMock(), 1, 123, stage)
        else:
            with pytest.raises(type(outcome)):
                await manager._wait_for_pd_stage(AsyncMock(), AsyncMock(), 1, 123, stage)

        assert manager.pd_stage_waiting_request_counts[stage] == 0
        assert manager.metric_client.gauge_set.call_args_list == [
            call("lightllm_pd_master_stage_waiting_requests", 1, labels={"stage": stage}),
            call("lightllm_pd_master_stage_waiting_requests", 0, labels={"stage": stage}),
        ]

    asyncio.run(run())
