"""PD 索引由普通 CPU Tensor 上传，使用实际拷贝流并跳过运行态任务。"""

from types import SimpleNamespace
from unittest.mock import Mock
from queue import Queue

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl import prefill_trans_process


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prefill", [True, False])
def test_pd_uploads_cpu_indices_on_copy_stream_and_skips_runtime_pages(
    monkeypatch: pytest.MonkeyPatch,
    prefill: bool,
) -> None:
    implementation = prefill_trans_process if prefill else decode_trans_process
    module_class = implementation._PrefillTransModule if prefill else implementation._DecodeTransModule
    device_id = torch.cuda.current_device()
    args = SimpleNamespace(tp=2, dp=2, nnodes=1, pd_kv_page_num=2, pd_kv_page_size=4, pd_node_id=1)
    resource = SimpleNamespace(get_pd_operator=lambda: SimpleNamespace(get_page_bytes=lambda: 64))
    monkeypatch.setattr("lightllm.common.attention_state.manifest.build_state_manifest", lambda _: {"version": 1})
    uploads = []
    copies = []
    original_cuda = torch.Tensor.cuda

    def track_upload(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        uploads.append(
            (tensor.device.type, tensor.is_pinned(), kwargs.get("non_blocking"), torch.cuda.current_stream())
        )
        return original_cuda(tensor, *args, **kwargs)

    def capture_copy(**kwargs: object) -> None:
        indices = kwargs["token_indices"]
        copies.append((kwargs, None if indices is None else indices.clone()))

    monkeypatch.setattr(torch.Tensor, "cuda", track_upload)
    planned = []

    def bind_plan(managers, buffer, write):
        dp_index = len(planned)
        planned.append(tuple(managers))
        return SimpleNamespace(
            run=lambda **kwargs: capture_copy(dp_rank_in_node=dp_index, write_to_pd_page=write, **kwargs)
        )

    monkeypatch.setattr(implementation.CopyPlan, "for_pd", bind_plan)
    monkeypatch.setattr(implementation, "create_kv_transporter", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(implementation.threading.Thread, "start", lambda self: None)
    if prefill:
        module = module_class(
            args=args,
            device_id=device_id,
            task_in_queue=None,
            task_out_queue=None,
            resource_managers=[resource, resource],
        )
    else:
        module = module_class(
            args=args,
            device_id=device_id,
            task_in_queue=None,
            task_out_queue=None,
            resource_managers=[resource, resource],
            up_status_in_queue=None,
        )

    # 构造时两个 DP 的预热均上传单个零索引。
    assert [indices.cpu().tolist() for _, indices in copies] == [[0], [0]]
    assert [kwargs["dp_rank_in_node"] for kwargs, _ in copies] == [0, 1]

    tasks = [
        SimpleNamespace(
            mem_indexes=indices,
            page_kind=kind,
            src_page_index=page,
            dst_page_index=page,
            prefill_dp_index=0,
            decode_dp_index=0,
            req_idx=1,
            end_kv_index=7,
            xfer_handle=None,
        )
        for page, kind, indices in [(0, "kv", [5, 1, 7]), (1, "kv", [2]), (0, "att_state", None)]
    ]
    task_queue = module.local_copy_kv_queue if prefill else module.ready_page_task_queue
    copy_loop = module.local_copy_kv_loop if prefill else module.read_page_to_mems_loop
    monkeypatch.setattr(task_queue, "get", Mock(side_effect=[*tasks, _StopLoop()]))
    with pytest.raises(_StopLoop):
        copy_loop()
    module.copy_cuda_stream.synchronize()

    copied_indices = [None if indices is None else indices.cpu().tolist() for _, indices in copies[2:]]
    assert copied_indices == [[5, 1, 7], [2], None]
    assert len(uploads) == 4  # 两次预热、两次 KV 上传；运行态任务不访问 mem_indexes。

    # 下一批使用较短的索引，验证上传的数据与本次任务一致。
    tasks[0].mem_indexes = [9, 8]
    monkeypatch.setattr(task_queue, "get", Mock(side_effect=[tasks[0], _StopLoop()]))
    with pytest.raises(_StopLoop):
        copy_loop()
    module.copy_cuda_stream.synchronize()
    assert copies[-1][1].cpu().tolist() == [9, 8]
    assert all(
        device == "cpu" and not pinned and non_blocking and stream == module.copy_cuda_stream
        for device, pinned, non_blocking, stream in uploads
    )
    assert all(kwargs["write_to_pd_page"] is prefill for kwargs, _ in copies)
    assert all(indices is None or indices.is_cuda and indices.dtype == torch.long for _, indices in copies)

    written = torch.zeros(8, device=module.kv_move_buffer.device)

    def fail_after_first_field(**kwargs):
        written.fill_(17)
        raise RuntimeError("invalid second field")

    module.copy_plans[0].run = fail_after_first_field
    monkeypatch.setattr(task_queue, "get", Mock(side_effect=[tasks[0], _StopLoop()]))
    with pytest.raises(_StopLoop):
        copy_loop()
    failed = module.failed_queue.get_nowait()
    assert failed is tasks[0]
    assert "invalid second field" in failed.error_info
    completion = failed._local_completion
    fence = completion.source_reusable if prefill else completion.destination_ready
    fence.synchronize()
    assert torch.equal(written, torch.full_like(written, 17))


class _StopLoop(Exception):
    pass


def test_failed_receive_is_not_read_or_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    module = decode_trans_process._DecodeTransModule.__new__(decode_trans_process._DecodeTransModule)
    module.device_id = 0
    module.failed_queue = Queue()
    handle = SimpleNamespace(check_status=Mock(return_value="ERR"), error_info="recv failed")
    task = SimpleNamespace(xfer_handle=handle)
    module.transporter = SimpleNamespace(check_task_status=Mock(return_value="ERR"))
    module.ready_page_task_queue = SimpleNamespace(get=Mock(side_effect=[task, _StopLoop()]))
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    copy = Mock()
    module.copy_plans = [SimpleNamespace(run=copy)]

    with pytest.raises(_StopLoop):
        module.read_page_to_mems_loop()

    assert module.failed_queue.get_nowait() is task
    assert task.error_info == "recv failed"
    copy.assert_not_called()
