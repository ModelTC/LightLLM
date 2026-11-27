"""The CUDA graph contract every decode backend must keep (see AttentionBackend.build).

A captured graph reruns its kernels with the capture step's attention state, so a replay step can change
only tensor contents. For each decode backend this checks, with a capture step followed by a
replay step of the same (microbatch, batch size) but different requests and lengths:

1. every tensor the backend owns (any tensor that is not a batch's own) keeps its address and layout, and every
   other object, such as a FlashInfer wrapper, is the captured one;
2. those tensors hold the replay step's values, as an eager build of the same batch computes them;
3. host scalars stay as captured, except fields documented as hints that never size a launch;
4. the other microbatch's capture never shares storage with this one's.
"""

import dataclasses

import pytest
import torch

from lightllm.common.attention.backends.gqa import GQALayerConfig
from lightllm.common.attention.backends.mla import MLADecodeLayerConfig
from unit_tests.common.attention.backends.groups_runner import set_up
from unit_tests.common.attention.backends.step_fixture import make_infer_state, make_resources

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

REQUESTS, MAX_LEN, PAD, CAPACITY = 8, 512, 7, 512

# (b_req_idx, b_seq_len, b_mtp_index) of a capture step and a later replay step with the same batch size.
LAYOUTS = {
    "one_row": (([3, 0, 2, 1], [33, 9, 259, 100], None), ([1, 2, 0, 3], [400, 7, 64, 1], None)),
    "fixed_rows": (([3, 3, 0, 0], [40, 41, 299, 300], None), ([2, 2, 1, 1], [10, 11, 500, 501], None)),
    # Requests with one to three rows; PAD rows are padding.
    "dynamic_rows": (
        ([3, 3, 0, PAD], [40, 41, 299, 1], [0, 1, 0, 0]),
        ([1, 2, 2, 2], [77, 5, 6, 7], [0, 0, 1, 2]),
    ),
}


def make_batch(token_map, step, layout, *, capacity, microbatch, extra):
    req_ids, lengths, mtp_index = step
    device = token_map.device
    as_tensor = lambda values: torch.tensor(values, device=device, dtype=torch.int32)  # noqa: E731
    cu_q = torch.arange(len(lengths) + 1, device=device, dtype=torch.int32)
    cu_kv = torch.tensor([0, *lengths], device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    fields = dict(
        req_to_token_indexs=token_map,
        b_req_idx=as_tensor(req_ids),
        b_seq_len=as_tensor(lengths),
        batch_size=len(lengths),
        max_kv_seq_len=max(lengths),
        total_token_num=sum(lengths),
        b1_cu_q_seq_len=cu_q,
        b1_cu_kv_seq_len=cu_kv,
        pad_req_idx=PAD,
        cuda_graph_max_kv_seq_len=capacity,
        microbatch_index=microbatch,
        # Each request's last accepted draft position, read by speculative recurrent-state decode.
        req_to_mtp_state_index=torch.arange(REQUESTS, device=device, dtype=torch.int32) % 2,
    )
    if layout == "fixed_rows":
        fields.update(decode_rows_per_request=2)
    elif layout == "dynamic_rows":
        fields.update(decode_rows_per_request=3, dynamic_decode_rows=True, b_mtp_index=as_tensor(mtp_index))
    return make_infer_state(**{**fields, **extra(step)})


def diverse_groups(step):
    # Rows of a request share a radix-tree prefix node; the shared length stays below the row's length.
    req_ids, lengths, _ = step
    shared = [min(length - 1, 16) if req_ids.count(req) > 1 else 0 for req, length in zip(req_ids, lengths)]
    return dict(
        b_shared_seq_len=torch.tensor(shared, device="cuda", dtype=torch.int32),
        b_shared_radix_node_id=torch.tensor(req_ids, device="cuda", dtype=torch.int32),
    )


def no_extra(step):
    return {}


def ready(backend_cls, layer_config, **resources):
    """A backend set up as its state group's first create_att_state sets it up."""
    fields = dict(max_seq_len=MAX_LEN, max_cuda_graph_batch_size=8, max_decode_rows=8)
    return set_up(backend_cls, layer_config, make_resources(**{**fields, **resources}))


def gqa_config(**fields):
    return GQALayerConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, **fields)


MLA_DECODE = dict(num_heads=16, kv_lora_rank=512, qk_rope_head_dim=64, softmax_scale=0.1, dtype=torch.bfloat16)


def flashinfer_gqa():
    pytest.importorskip("flashinfer")
    from lightllm.common.attention.backends.gqa.flashinfer import FlashInferGQADecodeBackend

    return ready(FlashInferGQADecodeBackend, gqa_config(), page_size=1)


def flashinfer_mla():
    pytest.importorskip("flashinfer")
    from lightllm.common.attention.backends.mla.flashinfer import FlashInferMLADecodeBackend

    return ready(FlashInferMLADecodeBackend, MLADecodeLayerConfig(**MLA_DECODE), page_size=1)


def fa3_gqa(page_size):
    def factory():
        from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeBackend

        return ready(FA3GQADecodeBackend, gqa_config(), page_size=page_size)

    return factory


def triton_gqa():
    from lightllm.common.attention.backends.gqa.triton import TritonGQADecodeBackend

    return ready(TritonGQADecodeBackend, gqa_config())


def triton_quant_gqa_diverse():
    from lightllm.common.attention.backends.gqa.triton_quant import TritonQuantGQADecodeBackend

    config = gqa_config(kv_quant="int8_group", kv_quant_group_size=8)
    return ready(TritonQuantGQADecodeBackend, config, diverse_decode=True)


def triton_mla():
    from lightllm.common.attention.backends.mla.triton import TritonMLADecodeBackend

    return ready(TritonMLADecodeBackend, MLADecodeLayerConfig(**MLA_DECODE))


def fa3_sparse_mla():
    from lightllm.common.attention.backends.sparse_mla import SparseMLALayerConfig
    from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLADecodeBackend

    return ready(FA3SparseMLADecodeBackend, SparseMLALayerConfig(**MLA_DECODE, topk=64, kpool=4))


def recurrent(replay):
    def factory():
        from lightllm.common.attention.backends.gdn import GDNLayerConfig
        from lightllm.common.attention.backends.gdn.triton import ReplayGDNDecodeBackend, TritonGDNDecodeBackend

        config = GDNLayerConfig(2, 2, 16, 16, 4, "silu", torch.bfloat16, state_mtp_step=2, replay=replay)
        return ready(ReplayGDNDecodeBackend if replay else TritonGDNDecodeBackend, config)

    return factory


# (backend factory, layouts it serves, batch fields it needs, attention state scalars that are only hints).
# Hints select kernel configs or bound nothing a replay reads: kernels read actual lengths on the GPU.
# FlashInfer page-index buffers are valid only up to their indptr's last entry.
CASES = {
    "triton_gqa": (triton_gqa, ["one_row", "fixed_rows", "dynamic_rows"], no_extra, {"max_kv_seq_len"}),
    "triton_quant_gqa_diverse": (triton_quant_gqa_diverse, ["fixed_rows"], diverse_groups, {"max_kv_seq_len"}),
    "flashinfer_gqa": (flashinfer_gqa, ["one_row", "fixed_rows"], no_extra, set()),
    "fa3_gqa_page1": (fa3_gqa(1), ["one_row", "fixed_rows", "dynamic_rows"], no_extra, {"max_seqlen_k"}),
    "fa3_gqa_page16": (fa3_gqa(16), ["one_row", "fixed_rows", "dynamic_rows"], no_extra, {"max_seqlen_k"}),
    "triton_mla": (triton_mla, ["one_row"], no_extra, set()),
    "flashinfer_mla": (flashinfer_mla, ["one_row"], no_extra, set()),
    "fa3_sparse_mla": (fa3_sparse_mla, ["one_row"], no_extra, set()),
    "gdn_native": (recurrent(False), ["one_row", "fixed_rows", "dynamic_rows"], no_extra, set()),
    "gdn_replay": (recurrent(True), ["one_row", "fixed_rows", "dynamic_rows"], no_extra, set()),
}

SCALARS = (int, float, bool, str, torch.dtype, type(None))


def leaves(value, path=()):
    """Yield (path, value) for every tensor, scalar and other object reachable through attention state."""
    if isinstance(value, torch.Tensor) or isinstance(value, SCALARS):
        yield path, value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from leaves(getattr(value, field.name), (*path, field.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from leaves(item, (*path, index))
    else:
        yield path, value
        # A wrapper object, such as FlashInfer's, keeps the index buffers its captured kernels read.
        for name, item in sorted(vars(value).items()):
            if name.endswith("_buf") and isinstance(item, torch.Tensor):
                yield (*path, name), item


def batch_storage(batch):
    return {
        tensor.untyped_storage().data_ptr()
        for tensor in (*vars(batch).values(), batch.req_to_token_indexs)
        if isinstance(tensor, torch.Tensor)
    }


def owned_tensors(att_state, *batches):
    """The tensors of attention state the backend owns, as opposed to the batches' own tensors.

    Attention state may keep the capture batch's tensors; the caller rewrites those before each replay.
    """
    caller = set().union(*map(batch_storage, batches))
    return {
        path: t
        for path, t in leaves(att_state)
        if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in caller
    }


VALID_PREFIX = {"_paged_kv_indices_buf": "_paged_kv_indptr_buf", "_kv_indices_buf": "_kv_indptr_buf"}


def leading_block(tensor, shape):
    return tensor[tuple(slice(0, n) for n in shape)]


def valid_part(att_state_leaves, path, tensor):
    indptr = VALID_PREFIX.get(path[-1])
    if indptr is None:
        return tensor
    return tensor[: int(att_state_leaves[(*path[:-1], indptr)][-1])]


PARAMS = [(name, layout) for name, (_, layouts, _, _) in CASES.items() for layout in layouts]


@pytest.mark.parametrize("name,layout", PARAMS, ids=[f"{name}-{layout}" for name, layout in PARAMS])
def test_decode_backend_keeps_the_cuda_graph_contract(name, layout):
    factory, _, extra, hints = CASES[name]
    backend = factory()
    token_map = torch.randperm(REQUESTS * MAX_LEN, device="cuda", dtype=torch.int32).view(REQUESTS, MAX_LEN)
    capture_step, replay_step = LAYOUTS[layout]

    def build(step, *, capacity=CAPACITY, microbatch=0, builder=backend):
        batch = make_batch(token_map, step, layout, capacity=capacity, microbatch=microbatch, extra=extra)
        return batch, builder.build(batch)

    capture_batch, captured = build(capture_step)
    captured_leaves = dict(leaves(captured))
    captured_owned = owned_tensors(captured, capture_batch)
    captured_ptrs = {path: (t.data_ptr(), t.shape, t.stride()) for path, t in captured_owned.items()}

    replay_batch, replayed = build(replay_step)
    replayed_leaves = dict(leaves(replayed))
    assert replayed_leaves.keys() == captured_leaves.keys()
    for path, value in captured_leaves.items():
        new = replayed_leaves[path]
        if path in captured_owned:
            # (1) The graph reads the captured address, so the replay's values must land there.
            assert (new.data_ptr(), new.shape, new.stride()) == captured_ptrs[path], f"{name}: {path} moved on replay"
        elif isinstance(value, torch.Tensor):
            continue
        elif isinstance(value, SCALARS):
            # (3) Host values are baked into the graph.
            if path[-1] not in hints:
                assert new == value, f"{name}: host field {path} changed on replay: {value} -> {new}"
        else:
            assert new is value, f"{name}: {path} is a new object on replay"

    # (2) The kept tensors hold what an eager build of the replay step computes.
    _, eager = build(replay_step, capacity=None, builder=factory())
    eager_leaves = dict(leaves(eager))
    for path, kept in owned_tensors(replayed, capture_batch, replay_batch).items():
        expected = eager_leaves[path]
        if isinstance(expected, torch.Tensor):
            kept, expected = valid_part(replayed_leaves, path, kept), valid_part(eager_leaves, path, expected)
            assert kept.dim() == expected.dim(), f"{name}: {path} rank differs from eager"
            assert all(k >= e for k, e in zip(kept.shape, expected.shape)), f"{name}: {path} is smaller than eager"
            torch.testing.assert_close(
                leading_block(kept, expected.shape), expected, rtol=0, atol=0, msg=f"{name}: {path} != eager"
            )

    # (4) The other microbatch's graph keeps its own storage, and capturing it leaves this one's values.
    snapshot = {path: t.clone() for path, t in owned_tensors(replayed, capture_batch, replay_batch).items()}
    other_batch, other = build(capture_step, microbatch=1)
    other_storage = {t.untyped_storage().data_ptr() for t in owned_tensors(other, other_batch).values()}
    for path, t in owned_tensors(replayed, capture_batch, replay_batch).items():
        assert t.untyped_storage().data_ptr() not in other_storage, f"{name}: {path} shared across microbatches"
        torch.testing.assert_close(t, snapshot[path], rtol=0, atol=0)
