"""Optional runtime enqueue hints preserve CUDA and avoid CPU/GPU barriers."""
from unittest.mock import Mock, patch
import threading
import os
import subprocess
import sys

import pytest

from lightllm.backends.cuda.runtime import CudaRuntime
from lightllm.backends.metal.runtime import MetalRuntime


def test_cuda_runtime_has_no_layer_submission_hook():
    with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA probe")):
        assert CudaRuntime().layer_submission_hook is None


@pytest.mark.parametrize("batch,expected", [(1, 24), (4, 24), (8, 6), (16, 6)])
def test_metal_layer_submission_reuses_non_timing_event_without_wait(batch, expected):
    event = Mock()
    event.synchronize.side_effect = AssertionError("CPU wait")
    event.wait.side_effect = AssertionError("stream wait")
    with patch("torch.mps.Event", return_value=event) as factory, patch(
        "torch.mps.synchronize", side_effect=AssertionError("device barrier")
    ):
        runtime = MetalRuntime()
        factory.assert_not_called()
        for layer in range(24):
            runtime.layer_submission_hook(layer, batch)
        factory.assert_called_once_with(enable_timing=False)
        assert event.record.call_count == expected
        event.query.assert_not_called()
        event.wait.assert_not_called()
        event.synchronize.assert_not_called()


def test_metal_layer_submission_events_are_owned_by_the_calling_thread():
    events = []

    def create(**kwargs):
        assert kwargs == {"enable_timing": False}
        event = Mock()
        events.append(event)
        return event

    with patch("torch.mps.Event", side_effect=create):
        runtime = MetalRuntime()
        for _ in range(2):
            worker = threading.Thread(target=lambda: runtime.layer_submission_hook(0, 1))
            worker.start()
            worker.join()
        assert len(events) == 2 and events[0] is not events[1]
        assert all(event.record.call_count == 1 for event in events)


def test_common_decode_calls_optional_hook_after_each_layer():
    code = r"""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from lightllm.common.basemodel import basemodel as module
def passthrough(input,**kwargs):return input
model=object.__new__(module.TpPartBaseModel)
model.pre_post_weight=None
model.pre_infer=SimpleNamespace(token_forward=lambda *args:torch.zeros(1,1),_tpsp_sp_split=passthrough)
model.post_infer=SimpleNamespace(token_forward=passthrough,_tpsp_allgather=passthrough)
model.layers_num=3
model.trans_layers_weight=[None]*3
model._create_model_output=lambda x,*args:x
events=[]
class Layer:
    def __init__(self,i):self.i=i
    def token_forward(self,input,*args):
        events.append(('layer',self.i));return input+1
model.layers_infer=[Layer(i) for i in range(3)]
def add(layer_index,hidden):events.append(('collect',layer_index))
state=SimpleNamespace(input_ids=torch.tensor([0]),batch_size=4,is_cuda_graph=False,
    hidden_collector=SimpleNamespace(add=add,add_final_hidden=lambda *args:None))
def submit(i,batch):
    assert batch==4
    events.append(('submit',i))
with patch.object(module.dist_group_manager,'clear_deepep_buffer'):
    for hook in (None,submit):
        events.clear()
        with patch.object(module,'_layer_submission_hook',hook):actual=model._token_forward(state)
        assert actual.item()==3
        expected=[]
        for i in range(3):
            expected.append(('layer',i))
            if hook:expected.append(('submit',i))
            expected.append(('collect',i))
        assert events==expected,events
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LIGHTLLM_BACKEND": "metal"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
