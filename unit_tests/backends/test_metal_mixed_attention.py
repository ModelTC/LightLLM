"""Mixed forwards must preserve LightLLM's paged mapping and query ordering."""
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")


@pytest.mark.parametrize("mode", ["native", "fallback", "window", "alibi"])
def test_mixed_attention_preserves_mapping_and_special_controls(mode):
    code = r"""
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from lightllm.backends.metal.attention import MetalPrefillAttState
from lightllm.backends.metal.paged_attention import decode
from lightllm.common.basemodel.attention.base_att import AttControl

torch.set_num_threads(4); torch.manual_seed(91205)
mode=sys.argv[1]
table=torch.randperm(6*129).view(6,129).to(torch.int32)
hq=torch.randn(11,8,256,dtype=torch.float16)*.7
hk=torch.randn(6*129,2,256,dtype=torch.float16)*.7
hv=torch.randn_like(hk)
q,k,v=(t.to('mps') for t in (hq,hk,hv)); gpu_table=table.to('mps')
meta=dict(b_req_idx=[4,1,3],b_seq_len=[129,129,21],b_ready_cache_len=[128,120,20],
          b_q_seq_len=[1,9,1],b_q_start_loc=[9,0,10])
state=SimpleNamespace(**{n:torch.tensor(x,device='mps',dtype=torch.int32) for n,x in meta.items()},
                      req_manager=SimpleNamespace(req_to_token_indexs=gpu_table))
att=MetalPrefillAttState(backend=None,infer_state=state)
control=AttControl()
if mode=='window':control=AttControl(use_sliding_window=True,sliding_window=(7,0))
if mode=='alibi':control=AttControl(use_alibi=True,tp_alibi=torch.arange(8,device='mps',dtype=torch.float32)/16)
storage=torch.full((hq.numel()+7,),-17.,device='mps',dtype=q.dtype)
output=storage[3:-4].view_as(q)
def alloc(shape,**kwargs):
    assert tuple(shape)==tuple(q.shape);output.fill_(float('nan'));return output
native_patch={'side_effect':AssertionError('special control dispatched')}
if mode=='native':native_patch={'wraps':decode}
if mode=='fallback':native_patch={'return_value':None}
with torch.inference_mode(),patch('lightllm.backends.metal.paged_attention.decode',**native_patch) as native:
    got=att.prefill_att(q,k,v,control,alloc_func=alloc)
    assert native.call_count==(2 if mode in ('native','fallback') else 0)
    assert got.data_ptr()==output.data_ptr()
    # Reuse state, including its groups, across layers/attention calls.
    again=att.prefill_att(q,k,v,control,alloc_func=alloc)
    assert native.call_count==(4 if mode in ('native','fallback') else 0)
expected=torch.empty_like(hq,dtype=torch.float32)
for req,seq,ready,n,start in zip(*meta.values()):
    idx=table[req,:seq].long()
    query=hq[start:start+n].float().transpose(0,1).unsqueeze(0)
    key=hk[idx].float().repeat_interleave(4,1).transpose(0,1).unsqueeze(0)
    value=hv[idx].float().repeat_interleave(4,1).transpose(0,1).unsqueeze(0)
    pos=torch.arange(ready,ready+n)[:,None];cols=torch.arange(seq)[None,:]
    allowed=cols<=pos
    if mode=='window':allowed &= cols>=pos-7
    mask=allowed
    if mode=='alibi':
        bias=torch.arange(8,dtype=torch.float32)[None,:,None,None]/16*(cols-pos)
        mask=bias.masked_fill(~allowed,-float('inf'))
    expected[start:start+n]=F.scaled_dot_product_attention(query,key,value,attn_mask=mask)[0].transpose(0,1)
torch.testing.assert_close(got.cpu().float(),expected,atol=.002,rtol=.003)
assert torch.all(storage[:3].cpu()==-17) and torch.all(storage[-4:].cpu()==-17)
for device,host in [(q,hq),(k,hk),(v,hv),(gpu_table,table)]:torch.testing.assert_close(device.cpu(),host,atol=0,rtol=0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, mode],
        env={**os.environ, "LIGHTLLM_BACKEND": "metal"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
