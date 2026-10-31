"""Exercise production PD transporters between independent CUDA processes."""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from typing import TYPE_CHECKING
from unit_tests.common.att_resources.model_helpers import main_linear_state

import importlib.util
import os
import pickle
import time
import traceback

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.pd_io_struct import PDAgentMetadata, PDChunckedTransTask
from lightllm.server.router.model_infer.mode_backend.pd.resource_io import allocate_transfer_buffer, copy_page
from lightllm.utils.envs_utils import get_env_start_args, set_env_start_args
from unit_tests.common.att_resources.test_model_resources import make_resources

if TYPE_CHECKING:
    from collections.abc import Sequence
    from lightllm.common.attention_state.manager import AttentionStateManager
    from lightllm.common.attention_state.full_att.packed_mla_state import PackedMLAFullAttentionState
    from lightllm.server.router.model_infer.mode_backend.pd.nccl_kv_transporter import NcclKVTransporter
    from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import NixlKVTransporter
    from multiprocessing.connection import Connection


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("backend", ["nccl", "nixl"])
def test_pd_network_restores_packed_kv_accepted_linear_state_and_full_tail(backend: str):
    if backend == "nixl" and importlib.util.find_spec("nixl") is None:
        pytest.skip("NIXL unavailable")
    context = mp.get_context("spawn")
    pipes = [context.Pipe() for _ in range(2)]
    processes = [
        context.Process(target=_worker, args=(child, backend, sender))
        for (_, child), sender in zip(pipes, (True, False))
    ]
    try:
        for process in processes:
            process.start()
        for _, child in pipes:
            child.close()
        metadata = [_receive(parent, "metadata") for parent, _ in pipes]
        for (parent, _), peer in zip(pipes, reversed(metadata)):
            parent.send(peer)
        source, target = [_receive(parent, "result") for parent, _ in pipes]
        assert source.keys() == target.keys()
        for field in source:
            np.testing.assert_array_equal(source[field], target[field], err_msg=field)
        for parent, _ in pipes:
            parent.send(None)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
        for parent, child in pipes:
            parent.close()
            child.close()


def _notification(
    transporter: NcclKVTransporter | NixlKVTransporter,
    stage: str,
    pending: list[PDChunckedTransTask],
    key: str | None = None,
):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        for messages in transporter.get_new_notifs().values():
            pending.extend(pickle.loads(message) for message in messages)
        # Notifications are grouped by peer, so a later page's request can be
        # returned before the current page's done. Match tasks as production does.
        for index, task in enumerate(pending):
            if task.error_info is not None:
                raise AssertionError(task.error_info)
            if task.write_stage == stage and (key is None or task.get_key() == key):
                return pending.pop(index)
        time.sleep(0.002)
    raise TimeoutError(f"PD {stage} notification did not arrive")


def _task(peer: PDAgentMetadata, kind: str):
    return PDChunckedTransTask(
        request_id=123,
        start_kv_index=0 if kind == "kv" else 5,
        end_kv_index=5,
        time_out_secs=60,
        pd_master_node_id=0,
        prefill_dp_index=0,
        decode_dp_index=0,
        src_device_id=0,
        dst_device_id=1,
        mem_indexes=[3, 0, 7, 1, 9] if kind == "kv" else [],
        prefill_agent_name=None,
        prefill_agent_metadata=None,
        prefill_num_pages=None,
        prefill_page_reg_desc=None,
        decode_agent_name=peer.agent_name,
        decode_agent_metadata=peer.agent_metadata,
        decode_num_pages=peer.num_pages,
        decode_page_reg_desc=peer.page_reg_desc,
        first_gen_token_id=42,
        first_gen_token_logprob=-0.5,
        src_page_index=1 if kind == "kv" else 2,
        page_kind=kind,
        req_idx=1,
    )


def _snapshot(
    resources: AttentionStateManager,
    full: PackedMLAFullAttentionState,
    indices: Sequence[int] | torch.Tensor,
    req_idx: int,
    cursor: int,
):
    result = {}
    for name, field in full_state_test_fields(full, full_state_test_buffers(full)).items():
        if "token" in field.axes:
            ids = torch.tensor(indices, device=field.tensor.device)
            result[name] = field.tensor.index_select(field.axes.index("token"), ids).float().cpu().numpy()
    state = main_linear_state(resources)
    result["conv"] = state.conv_state[:, req_idx, ..., cursor : cursor + state.config.conv_width].float().cpu().numpy()
    result["ssm"] = state.ssm_state[:, req_idx * 3 + cursor].cpu().numpy()
    tail = resources.main_compressor_indexer_w4_s4_bf16_state.buffer[:, req_idx]
    result["tail"] = torch.stack([tail[:, position % tail.shape[1]] for position in range(1, 5)], dim=1).float().cpu().numpy()
    return result


def _worker(connection: Connection, backend: str, sender: bool):
    transporter = None
    pending = []
    try:
        device = 0 if sender else 1
        torch.cuda.set_device(device)
        torch.set_num_threads(2)
        set_env_start_args(StartArgs(run_mode="prefill" if sender else "decode", pd_kv_page_size=32))
        get_env_start_args.cache_clear()
        resources, full = make_resources(
            glm=True,
            tp=1,
            rank=0,
            mtp=2,
        )
        buffer = allocate_transfer_buffer(resources, 3, device=f"cuda:{device}")
        buffer.fill_(93 if sender else 11)
        copy_stream = torch.cuda.Stream()
        if backend == "nccl":
            from lightllm.server.router.model_infer.mode_backend.pd.nccl_kv_transporter import NcclKVTransporter

            transporter = NcclKVTransporter(
                node_id=os.getpid(),
                tp_idx=device,
                kv_move_buffer=buffer,
                host_ip="127.0.0.1",
            )
        else:
            from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import NixlKVTransporter

            transporter = NixlKVTransporter(os.getpid(), device, buffer)
        connection.send(
            (
                "metadata",
                PDAgentMetadata(
                    agent_name=transporter.agent_name,
                    agent_metadata=transporter.agent_metadata,
                    num_pages=transporter.num_pages,
                    page_reg_desc=transporter.local_page_mem_desc,
                ),
            )
        )
        assert connection.poll(60), "peer metadata did not arrive"
        peer = connection.recv()
        if sender:
            for i, field in enumerate(full_state_test_fields(full, full_state_test_buffers(full)).values()):
                if "token" in field.axes:
                    values = torch.arange(field.tensor.numel(), device=device).reshape(field.tensor.shape) % 17 + i + 1
                    field.tensor.copy_(values)
            state = main_linear_state(resources)
            for i, tensor in enumerate(
                (state.conv_state, state.ssm_state, resources.main_compressor_indexer_w4_s4_bf16_state.buffer)
            ):
                tensor.copy_(torch.arange(tensor.numel(), device=device).reshape(tensor.shape) % (23 + i * 7))
            expected = _snapshot(
                resources=resources,
                full=full,
                indices=[3, 0, 7, 1, 9],
                req_idx=1,
                cursor=0,
            )
            for kind in ("kv", "att_state"):
                task = _task(peer, kind)
                with torch.cuda.stream(copy_stream):
                    copy_page(
                        node_att_managers=[resources],
                        pd_page_buffers=buffer,
                        pd_page_index=task.src_page_index,
                        dp_rank_in_node=0,
                        tp_world_size=1,
                        token_indices=torch.tensor(task.mem_indexes, dtype=torch.long, device=device),
                        task_kind=kind,
                        req_idx=1,
                        state_token_len=task.end_kv_index,
                        write_to_pd_page=True,
                    )
                # The production sender publishes a page only after its copy event completes.
                copy_stream.synchronize()
                transporter.send_write_request_task_to_decode_node(task)
                ready = _notification(
                    transporter=transporter,
                    stage="ready",
                    pending=pending,
                    key=task.get_key(),
                )
                assert ready.get_key() == task.get_key()
                task.dst_page_index = ready.dst_page_index
                task.xfer_handle = transporter.write_blocks_paged(task)
                deadline = time.monotonic() + 60
                while transporter.check_task_status(task) == "PROC":
                    assert time.monotonic() < deadline, "PD transfer timed out"
                    time.sleep(0.002)
                assert transporter.check_task_status(task) == "DONE"
                transporter.send_write_done_task_to_decode_node(task)
                transporter.release_xfer_handle(task.xfer_handle)
        else:
            for storage_buffer in full_state_test_buffers(full).values():
                storage_buffer.zero_()
            pool = main_linear_state(resources).big_pages_pool
            pool.conv_state.fill_(41)
            pool.ssm_state.fill_(43)
            for kind in ("kv", "att_state"):
                request = _notification(transporter, "request", pending)
                assert request.page_kind == kind
                request.dst_page_index = 2 if kind == "kv" else 1
                transporter.send_write_ready_task_to_prefill_node(request)
                done = _notification(
                    transporter=transporter,
                    stage="done",
                    pending=pending,
                    key=request.get_key(),
                )
                assert done.get_key() == request.get_key()
                # Match production: unpack on a separate stream after the done notification.
                with torch.cuda.stream(copy_stream):
                    copy_page(
                        node_att_managers=[resources],
                        pd_page_buffers=buffer,
                        pd_page_index=request.dst_page_index,
                        dp_rank_in_node=0,
                        tp_world_size=1,
                        token_indices=torch.tensor([8, 12, 10, 15, 11], dtype=torch.long, device=device) if kind == "kv" else torch.tensor([], dtype=torch.long, device=device),
                        task_kind=kind,
                        req_idx=2,
                        state_token_len=request.end_kv_index,
                        write_to_pd_page=False,
                    )
                copy_stream.synchronize()
            expected = _snapshot(
                resources=resources,
                full=full,
                indices=[8, 12, 10, 15, 11],
                req_idx=2,
                cursor=0,
            )
            assert main_linear_state(resources).mtp_state_index[2] == 0
            assert not main_linear_state(resources).conv_state[:, :2].any()
            assert not main_linear_state(resources).ssm_state[:, :6].any()
            assert not resources.main_compressor_indexer_w4_s4_bf16_state.buffer[:, :2].any()
            assert not read_kv_storage(full)[:, [0, 1, 2, 3]].any()
            assert (pool.conv_state == 41).all()
            assert (pool.ssm_state == 43).all()
        assert (buffer[0] == (93 if sender else 11)).all()
        connection.send(("result", expected))
        assert connection.poll(60), "parent did not acknowledge completion"
        connection.recv()
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise
    finally:
        if transporter is not None:
            transporter.shutdown()
        connection.close()


def _receive(connection: Connection, stage: str):
    assert connection.poll(120), f"PD worker did not report {stage}"
    kind, result = connection.recv()
    assert kind == stage, result
    return result
