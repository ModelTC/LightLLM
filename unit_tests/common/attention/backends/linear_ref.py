"""Sequential fp32 torch references for GDN and KDA, and a driver that runs a backend through
prefill, speculative decode steps and acceptance commits the way the serving loop does."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from lightllm.common.attention.backend import AttentionMetadata


def conv_token(history: torch.Tensor, x: torch.Tensor, weight: torch.Tensor):
    """One causal depthwise conv step with SiLU; history [C, W-1] holds the previous raw inputs."""
    window = torch.cat([history, x[:, None]], dim=1)
    y = F.silu((window * weight.float()).sum(-1))
    # Kernels store the conv output in the activation dtype before attention.
    return y.bfloat16().float(), window[:, 1:]


def l2norm(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def delta_rule(state, q, k, v, decay, beta):
    """state [H, K, V]; decay broadcastable to [H, K, 1]; beta [H]. Returns (out [H, V], new state)."""
    state = state * decay
    delta = (v - torch.einsum("hkv,hk->hv", state, k)) * beta[:, None]
    state = state + k[:, :, None] * delta[:, None, :]
    return torch.einsum("hkv,hk->hv", state, q), state


@dataclass
class GDNShape:
    hk: int = 2
    hv: int = 4
    dk: int = 64
    dv: int = 64
    width: int = 4

    @property
    def conv_channels(self):
        return 2 * self.hk * self.dk + self.hv * self.dv


@dataclass
class KDAShape:
    h: int = 2
    d: int = 128
    width: int = 4
    lower_bound: float = -5.0

    @property
    def conv_channels(self):
        return 3 * self.h * self.d


def gdn_token(shape: GDNShape, params, history, state, x, a, b):
    y, history = conv_token(history, x, params["conv_weight"])
    q, k, v = y.split([shape.hk * shape.dk, shape.hk * shape.dk, shape.hv * shape.dv])
    group = shape.hv // shape.hk
    q = (l2norm(q.view(shape.hk, shape.dk)) * shape.dk ** -0.5).repeat_interleave(group, 0)
    k = l2norm(k.view(shape.hk, shape.dk)).repeat_interleave(group, 0)
    v = v.view(shape.hv, shape.dv)
    g = -params["A_log"].float().exp() * F.softplus(a.float() + params["dt_bias"].float())
    out, state = delta_rule(state, q, k, v, g.exp()[:, None, None], b.float().sigmoid())
    return out, history, state


def kda_token(shape: KDAShape, params, history, state, x, gate, beta):
    y, history = conv_token(history, x, params["conv_weight"])
    q, k, v = (part.view(shape.h, shape.d) for part in y.split(shape.h * shape.d))
    q, k = l2norm(q) * shape.d ** -0.5, l2norm(k)
    amplitude = params["A_log"].float().exp()[:, None]
    gate = gate.float().view(shape.h, shape.d) + params["dt_bias"].float().view(shape.h, shape.d)
    decay = torch.exp(shape.lower_bound * torch.sigmoid(amplitude * gate))
    out, state = delta_rule(state, q, k, v, decay[:, :, None], beta.float().sigmoid())
    return out, history, state


@dataclass
class RequestRef:
    """A request's accepted conv history and SSM state, and the snapshots after each token of the last step."""

    history: torch.Tensor
    state: torch.Tensor
    snapshots: list = field(default_factory=list)

    def run(self, token_fn, xs, gates, betas):
        outs, self.snapshots = [], []
        history, state = self.history, self.state
        for x, gate, beta in zip(xs, gates, betas):
            out, history, state = token_fn(history, state, x.float(), gate, beta)
            outs.append(out)
            self.snapshots.append((history, state))
        # Without speculation every token is accepted.
        self.history, self.state = history, state
        return torch.stack(outs) if outs else None

    def accept(self, accepted: int):
        self.history, self.state = self.snapshots[accepted - 1]


def decode_batch(reqs, lengths, width, hold, req_to_mtp_state_index, *, dynamic, microbatch_index=0, graph=False):
    """Rows of one speculative decode step: request i contributes lengths[i] rows (width without dynamic)."""
    rows = [r for r, n in zip(reqs, lengths) for _ in range(n)]
    mtp_index = [j for n in lengths for j in range(n)]
    if dynamic:
        pad = width * len(reqs) - len(rows)
        rows, mtp_index = rows + [hold] * pad, mtp_index + [0] * pad
    b_req_idx = torch.tensor(rows, dtype=torch.int32, device="cuda")
    return AttentionMetadata(
        req_to_token_indexs=None,
        b_req_idx=b_req_idx,
        b_seq_len=torch.full_like(b_req_idx, 64),
        batch_size=b_req_idx.numel(),
        max_kv_seq_len=64,
        total_token_num=64 * b_req_idx.numel(),
        decode_rows_per_request=width,
        dynamic_decode_rows=dynamic and width > 1,
        pad_req_idx=hold,
        b_mtp_index=torch.tensor(mtp_index, dtype=torch.int32, device="cuda"),
        req_to_mtp_state_index=req_to_mtp_state_index if width > 1 else None,
        cuda_graph_max_kv_seq_len=64 if graph else None,
        microbatch_index=microbatch_index,
    )


def prefill_batch(reqs, lengths, ready):
    b_req_idx = torch.tensor(reqs, dtype=torch.int32, device="cuda")
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    ready = torch.tensor(ready, dtype=torch.int32, device="cuda")
    return AttentionMetadata(
        req_to_token_indexs=None,
        b_req_idx=b_req_idx,
        b_seq_len=ready + lens,
        batch_size=len(reqs),
        max_kv_seq_len=int((ready + lens).max()),
        total_token_num=int((ready + lens).sum()),
        cu_q_seq_len=F.pad(lens.cumsum(0, dtype=torch.int32), (1, 0)),
        b_ready_cache_len=ready,
        max_q_seq_len=max(lengths),
        max_ready_cache_len=int(ready.max()),
    )


def assert_rows_close(actual, expected, what):
    """bf16 outputs against the fp32 reference: bounded by a fraction of the reference scale."""
    actual, expected = actual.float(), expected.float()
    scale = expected.abs().max().clamp_min(1e-3)
    error = (actual - expected).abs().max() / scale
    assert error < 2e-2, f"{what}: max error {error:.3e} of reference scale {scale:.3e}"


# Each case: (lengths per request, accepted per request) for one decode step.
SPEC_STEPS = {
    False: [([3, 3, 3], [3, 1, 2]), ([3, 3, 3], [2, 3, 1]), ([3, 3, 3], [1, 2, 3])],
    True: [([3, 1, 2], [2, 1, 2]), ([2, 3, 1], [2, 3, 1]), ([1, 2, 3], [1, 1, 3])],
}


CAPACITY = 5  # request slots; the last one is the replay state's HOLD slot
HOLD = CAPACITY - 1
REQS = [2, 0, 3]


def accepted_ssm(state, replay, reqs):
    """Each request's accepted SSM state [len(reqs), H, K, V]: native picks its accepted speculative slot;
    replay folds the accepted history into a copy of the canonical state."""
    ssm = state.ssm_state[0]
    req_t = torch.tensor(reqs, dtype=torch.int32, device="cuda")
    if not replay:
        width = ssm.shape[0] // CAPACITY
        return ssm[req_t.long() * width + state.mtp_state_index[req_t.long()].long()]
    from lightllm.common.attention.kernels.gdn.replayssm.replayssm_fold import fold_prefix

    saved, folded = state.ssm_state, state.ssm_state.clone()
    state.ssm_state = folded
    try:
        fold_prefix(state, req_t)
    finally:
        state.ssm_state = saved
    return folded[0, req_t.long()]


def run_prefill_then_decode(family, *, replay, mtp_step, dynamic, prefill_impl, decode_impl, prefill_builder, seed=7):
    """Prefill fresh and cached requests, then run speculative decode steps with acceptance commits, comparing
    outputs and accepted conv/SSM state against the sequential reference after every step."""
    from lightllm.common.attention.backends.gdn.triton import GDNDecodeBuildContext

    torch.manual_seed(seed)
    width = mtp_step + 1
    state = family.make_state(replay=replay, mtp_step=mtp_step)
    conv, ssm = state.get_layer_view(logical_layer_id=0)
    conv.copy_(torch.randn_like(conv, dtype=torch.float32).mul_(0.5).to(conv.dtype))
    ssm.copy_(torch.randn_like(ssm, dtype=torch.float32).mul_(0.1).to(ssm.dtype))
    slots_per_request = 1 if replay else width
    lengths, ready = [6, 4, 9], [0, 5, 7]
    refs = []
    for req, cached in zip(REQS, ready):
        first = req * slots_per_request
        if not cached:
            # Allocation zeroes a new request's SSM; the conv history is ignored without a cached prefix.
            ssm[first : first + slots_per_request].zero_()
        history = conv[req, :, : family.shape.width - 1].float() if cached else conv.new_zeros(conv.shape[1], 3).float()
        refs.append(RequestRef(history=history.clone(), state=ssm[first].float().clone()))
    state.mtp_state_index.zero_()

    # Prefill.
    rows = sum(lengths)
    tensors = family.make_inputs(rows)
    step = prefill_impl.build(prefill_batch(REQS, lengths, ready), prefill_builder)
    out = family.alloc_out(rows)
    prefill_impl.forward(step, family.layer_inputs(_clones(tensors), conv, ssm, state if replay else None), out=out)
    start = 0
    for i, (ref, n) in enumerate(zip(refs, lengths)):
        expected = ref.run(family.token_fn, *family.token_args(tensors, start, start + n))
        assert_rows_close(out[start : start + n], expected, f"prefill output of request {i}")
        start += n
    _check_state(state, conv, replay, refs, "prefill")

    # Decode.
    builder = GDNDecodeBuildContext(replay=replay)
    steps = SPEC_STEPS[dynamic] if width > 1 else [([1, 1, 1], [1, 1, 1])] * 3
    for step_index, (step_lengths, accepted) in enumerate(steps):
        step_lengths = step_lengths if dynamic else [width] * len(REQS)
        batch = decode_batch(REQS, step_lengths, width, HOLD, state.mtp_state_index, dynamic=dynamic)
        tensors = family.make_inputs(batch.batch_size)
        step = decode_impl.build(batch, builder)
        out = family.alloc_out(batch.batch_size)
        # A replay decode impl advances the replay cursors itself after the last layer.
        decode_impl.forward(step, family.layer_inputs(_clones(tensors), conv, ssm, state if replay else None), out=out)
        start = 0
        for i, (ref, n) in enumerate(zip(refs, step_lengths)):
            expected = ref.run(family.token_fn, *family.token_args(tensors, start, start + n))
            assert_rows_close(out[start : start + n], expected, f"decode step {step_index} request {i}")
            start += n
        if width > 1:
            state.commit_bounded_att_state(
                torch.tensor(REQS, dtype=torch.int32, device="cuda"),
                torch.tensor(accepted, dtype=torch.int32, device="cuda"),
                torch.tensor(step_lengths, dtype=torch.int32, device="cuda"),
            )
            for ref, n in zip(refs, accepted):
                ref.accept(n)
        _check_state(state, conv, replay, refs, f"decode step {step_index}")


def _clones(tensors):
    # Decode conv kernels write their output over the input rows; keep the reference's copy intact.
    return {name: t.clone() for name, t in tensors.items()}


def _check_state(state, conv, replay, refs, what):
    for i, (req, ref) in enumerate(zip(REQS, refs)):
        offset = int(state.mtp_state_index[req])
        window = conv[req, :, offset : offset + ref.history.shape[1]].float()
        assert torch.equal(window, ref.history), f"{what}: conv history of request {i}"
    actual = accepted_ssm(state, replay, REQS).float()
    for i, ref in enumerate(refs):
        assert_rows_close(actual[i], ref.state, f"{what}: accepted SSM state of request {i}")
