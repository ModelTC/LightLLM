from lightllm.common.attention.api.environment import BackendEnv
from lightllm.common.attention.session import AttentionSession

"""启动子进程中的小规模数值检查，不接触模型的真实 KV 或请求状态。"""

import torch

from .api import AttentionBatch, AttentionPhase, ContextCapacity, LayerAttention
from .api.inputs import DenseInputs, DenseResources
from .api.workspace import AttentionWorkspace


def probe_dense(backend_cls, spec, runtime):
    page = spec.cache.page_size
    stride = ((16 + page - 1) // page) * page
    q_heads, kv_heads, dim = spec.num_q_heads, spec.cache.key_heads, spec.head_size
    env = BackendEnv(
        runtime=runtime,
        network_config={},
        hold_req_id=2,
        draft_step=0,
        max_seq_len=stride,
        tp_size=1,
        alloc_tensor=lambda shape, dtype, device=runtime.device: torch.empty(shape, dtype=dtype, device=device),
    )
    backend = backend_cls(env, spec)
    device = runtime.device
    lengths = (7, 13)
    q_lengths = (2, 3) if spec.phase is AttentionPhase.PREFILL else (1, 1)
    query = torch.randn((sum(q_lengths), q_heads, dim), dtype=spec.dtype, device=device)
    key, value = (torch.randn((3 * stride, kv_heads, dim), dtype=spec.dtype, device=device) for _ in range(2))
    batch = AttentionBatch(
        phase=spec.phase,
        batch_size=2,
        token_num=sum(q_lengths),
        total_token_num=sum(lengths),
        max_q_seq_len=max(q_lengths),
        max_kv_seq_len=max(lengths),
        max_cache_len=max(n - q for n, q in zip(lengths, q_lengths)),
        b_req_idx=torch.tensor([0, 1], dtype=torch.int32, device=device),
        b_seq_len=torch.tensor(lengths, dtype=torch.int32, device=device),
        b_q_seq_len=torch.tensor(q_lengths, dtype=torch.int32, device=device),
        b1_cu_q_seq_len=torch.tensor([0, q_lengths[0], sum(q_lengths)], dtype=torch.int32, device=device),
        b1_cu_kv_seq_len=torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32, device=device),
        req_to_token_indexs=torch.arange(3 * stride, dtype=torch.int32, device=device).view(3, stride),
        hold_req_id=2,
        b_ready_cache_len=torch.tensor([n - q for n, q in zip(lengths, q_lengths)], dtype=torch.int32, device=device),
    )
    capacity = ContextCapacity(
        phase=spec.phase, batch_size=2, token_num=sum(q_lengths), max_kv_seq_len=stride, max_q_seq_len=max(q_lengths)
    )
    workspace = AttentionWorkspace(runtime=runtime)
    workspace.reserve(backend.resource_requirements(capacity, capture=False))
    session = AttentionSession()
    context = session.allocate(backend, capacity, capture=False, workspace=workspace)
    session.plan(backend, context, batch)
    session.build(backend, context)
    session.contexts[0] = context
    attention = LayerAttention(
        group_id=0,
        logical_layer_id=0,
        resources=DenseResources(key, value),
        backend=backend,
    )
    actual = attention.forward(session, DenseInputs(query), out=torch.empty_like(query))
    expected, start = [], 0
    for request, (length, q_length) in enumerate(zip(lengths, q_lengths)):
        q = query[start : start + q_length].float()
        k = key[request * stride : request * stride + length].float().repeat_interleave(q_heads // kv_heads, dim=1)
        v = value[request * stride : request * stride + length].float().repeat_interleave(q_heads // kv_heads, dim=1)
        logits = torch.einsum("qhd,khd->hqk", q, k) * dim ** -0.5
        if spec.causal:
            mask = torch.arange(length, device=device)[None, :] <= (
                length - q_length + torch.arange(q_length, device=device)[:, None]
            )
            logits.masked_fill_(~mask, -torch.inf)
        expected.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), v))
        start += q_length
    expected = torch.cat(expected).to(spec.dtype)
    runtime.synchronize()
    if not torch.allclose(actual, expected, atol=0.02, rtol=0.02):
        return False, f"dense {spec.phase.value} output mismatch: max diff {(actual - expected).abs().max().item()}"
    return True, None
