from lightllm.common.attention_state.full_att import FullAttentionState
from lightllm.common.attention_state.token_copy import TokenPageCopyPlan, TransferBatch
from lightllm.common.attention_state.bounded_att.copy_plan import BoundedPageCopy
from lightllm.common.attention_state.bounded_att.peer_copy import BoundedPeerCopy
from lightllm.common.attention_state.bounded_att.providers import bind_runtime_page_copy


def copy_token_pages(state, indices, pages, page_ids, layout, write, ready=None, pad_tail=False):
    if ready is not None and len(ready) == len(page_ids) and all(ready):
        return
    batch = TransferBatch.prepare(indices, page_ids, ready, page_count=pages.shape[0])
    plan = TokenPageCopyPlan.bind(state.get_transfer_fields(), layout, pages, write, pad_tail=pad_tail)
    plan.run(batch)


def copy_tokens(source, destination, src, dst):
    plan = TokenPageCopyPlan.for_tokens(source.get_token_copy_fields(), destination.get_token_copy_fields())
    plan.run(TransferBatch(src, dst))


def copy_snapshot_pages(state, kind, slots, pages, page_ids, layout, write, ready=None):
    plan = BoundedPageCopy.bind(state.get_snapshot_transfer_view(kind), pages, layout, write)
    plan.run(slots, page_ids, ready)


def copy_pd_page(state, pages, page_id, layout, write, token_indices=None, req_idx=None, state_token_len=None):
    if isinstance(state, FullAttentionState):
        copy_token_pages(state, token_indices, pages, [page_id], layout, write, pad_tail=True)
    else:
        bind_runtime_page_copy(state, pages, layout, write)(
            req_idx=req_idx, pd_page_index=page_id, state_token_len=state_token_len
        )


def copy_bounded_peer(source, destination, src, dst, runtime):
    plan = BoundedPeerCopy.bind(source, destination)
    if runtime:
        plan.copy_runtime(src, dst)
    else:
        plan.copy_pages(src, dst)
