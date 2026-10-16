"""Remote registration is driven by global migration plans, not cluster size."""

import builtins
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server.router.model_infer.mode_backend import eplb_transfer as module


@pytest.fixture
def make_transfer(monkeypatch):
    def staging(self, weights, group, rank, world):
        self.device = "mock-device"
        self.global_rank, self.world_size = rank, world
        self.live = [[("weight", "live")]]
        self.staging = [[("weight", "staging")]]

    def ipc(self):
        self._rank_hostnames = ["a", "a", "b", "b", "c", "c"]
        self._same_node_ranks = {
            rank for rank, host in enumerate(self._rank_hostnames) if host == self._rank_hostnames[self.global_rank]
        }
        self._cross_node_ranks = set(range(self.world_size)) - self._same_node_ranks

    monkeypatch.setattr(module.NixlEPLBTransfer, "_init_staging", staging)
    monkeypatch.setattr(module.NixlEPLBTransfer, "_init_ipc_metadata", ipc)
    monkeypatch.setattr(module.NixlEPLBTransfer, "_init_push_layouts", lambda self: None)
    monkeypatch.setattr(module.NixlEPLBTransfer, "_prepare_part", lambda self, layer, plan, part: part)
    monkeypatch.setattr(module.torch.cuda, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(module, "CudaBatchMemcpy", lambda: object())
    monkeypatch.setenv("UCX_TLS", "test-existing-value")
    return lambda rank: module.NixlEPLBTransfer([], object(), rank, 6)


@pytest.mark.parametrize("rank", [0, 2, 4])
def test_multinode_initialization_and_node_local_plans_do_not_import_nixl(monkeypatch, make_transfer, rank):
    original_import = builtins.__import__

    def no_nixl(name, *args, **kwargs):
        if name == "nixl":
            pytest.fail("node-local migration must not require NIXL")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_nixl)
    transfer = make_transfer(rank)
    assert transfer._cross_node_ranks
    assert transfer._nixl_agent is None
    assert transfer.prepare_transfer([]) == []
    plans = [(0, [module.TransferStep(1, 0, 0, 0), module.TransferStep(3, 0, 2, 0)])]
    assert len(transfer.prepare_transfer(plans)) == 1
    assert transfer._nixl_agent is None
    assert transfer._registered_descs is None
    transfer.shutdown()


@pytest.mark.parametrize("rank", [0, 2, 4])
def test_first_remote_plan_initializes_all_ranks_once_including_nonparticipants(monkeypatch, make_transfer, rank):
    agent = Mock()
    agent.get_reg_descs.return_value = "descriptors"
    nixl = SimpleNamespace(nixl_agent=Mock(return_value=agent), nixl_agent_config=Mock(return_value="config"))
    monkeypatch.setitem(sys.modules, "nixl", nixl)
    exchange = Mock()
    monkeypatch.setattr(module.NixlEPLBTransfer, "_init_remote_metadata", exchange)
    transfer = make_transfer(rank)
    nixl.nixl_agent.assert_not_called()
    plans = [(0, [module.TransferStep(2, 0, 0, 0)])]
    transfer.prepare_transfer(plans)
    transfer.prepare_transfer(plans)
    nixl.nixl_agent.assert_called_once()
    exchange.assert_called_once()
    agent.get_reg_descs.assert_called_once_with(["live", "staging"])
    agent.register_memory.assert_called_once_with("descriptors", backends=["UCX"])
    transfer.shutdown()
    transfer.shutdown()
    agent.deregister_memory.assert_called_once_with("descriptors", backends=["UCX"])


@pytest.mark.parametrize("failure", ["import", "register", "metadata"])
def test_remote_initialization_failure_cleans_partial_resources(monkeypatch, make_transfer, failure):
    agent = Mock()
    agent.get_reg_descs.return_value = "descriptors"
    nixl = SimpleNamespace(nixl_agent=Mock(return_value=agent), nixl_agent_config=Mock(return_value="config"))
    monkeypatch.setitem(sys.modules, "nixl", None if failure == "import" else nixl)
    if failure == "register":
        agent.register_memory.side_effect = RuntimeError("registration failed")
    exchange = Mock(side_effect=ValueError("metadata failed") if failure == "metadata" else None)
    monkeypatch.setattr(module.NixlEPLBTransfer, "_init_remote_metadata", exchange)
    transfer = make_transfer(4)
    plans = [(0, [module.TransferStep(2, 0, 0, 0)])]
    with pytest.raises(RuntimeError):
        transfer.prepare_transfer(plans)
    assert transfer._nixl_agent is None
    assert transfer._registered_descs is None
    assert transfer._remote_agents == {}
    if failure != "import":
        agent.deregister_memory.assert_called_once_with("descriptors", backends=["UCX"])
    transfer.shutdown()
