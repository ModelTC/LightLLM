import pytest
import torch

from unit_tests.common.att_resources.test_pd_resource_binding import bounded_peers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA bounded snapshot kernels")
@pytest.mark.parametrize("family", ["gdn", "kda", "window", "compressor"])
@pytest.mark.parametrize("page_kind", ["big", "small"])
def test_bound_cpu_snapshots_reuse_native_views_and_preserve_ready_pages(pd_args, family, page_kind, monkeypatch):
    pd_args.cpu_cache_token_page_size = 128
    pd_args.linear_att_hash_page_size = 128
    source = bounded_peers(family, 1, False, 3)[0]
    target = bounded_peers(family, 1, False, 1)[0]
    for manager in (source, target):
        for state in manager.get_bounded_attention_states():
            state.initialize_page_pools(3, 2, pin_memory=True)
    operator = source.get_cpu_cache_operator()
    layouts = operator.get_page_layouts()
    pages = torch.full((3, operator.get_page_bytes()), 173, dtype=torch.uint8, pin_memory=True)
    writers = operator._bound_state_plans(pages, layouts, page_kind, True)
    readers = target.get_cpu_cache_operator()._bound_state_plans(pages, layouts, "big", False)
    for state in source.get_bounded_attention_states():
        for index in range(2):
            for field in state.get_snapshot_fields(page_kind, index).values():
                field.tensor.fill_(31 + index)
    with monkeypatch.context() as patch:
        for manager in (source, target):
            for state in manager.get_bounded_attention_states():
                for name in ("get_snapshot_transfer_view", "get_snapshot_fields", "bind_cpu_copy"):
                    patch.setattr(state, name, lambda *args: pytest.fail("native CPU fields must be prebound"))
        assert operator._bound_state_plans(pages, layouts, page_kind, True) is writers
        for writer in writers:
            writer.run([0, 1], [2, 1], [False, True])
            writer.run([1], [0])
        for reader in readers:
            reader.run([1, 2], [2, 0])
    torch.cuda.synchronize()
    assert (pages[1] == 173).all()
    for state in target.get_bounded_attention_states():
        for index, expected in ((1, 31), (2, 32)):
            for field in state.get_snapshot_fields("big", index).values():
                assert (field.tensor == expected).all()
