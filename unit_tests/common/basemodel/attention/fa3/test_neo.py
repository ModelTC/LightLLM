import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.fa3 import fp, neo


def _load_neo(monkeypatch, attention_func):
    interface = None
    if attention_func is not None:
        interface = ModuleType("flash_attn_interface")
        interface.flash_attn_with_kvcache = attention_func
    monkeypatch.setitem(sys.modules, "flash_attn_interface", interface)
    name = "lightllm.common.basemodel.attention.fa3._neo_import_test"
    spec = importlib.util.spec_from_file_location(name, neo.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_recognizes_neo_interface_without_replacing_regular_fa3(monkeypatch):
    def neo_attention(q, image_token_end=None):
        return q

    regular_attention = fp.flash_attn_with_kvcache
    module = _load_neo(monkeypatch, neo_attention)

    assert module.HAS_FLASH_ATTN_INTERFACE
    assert module.flash_attn_with_kvcache_neo is neo_attention
    assert fp.flash_attn_with_kvcache is regular_attention


@pytest.mark.parametrize("interface_kind", ["missing", "regular", "legacy_tag"])
def test_unavailable_neo_interface_provides_source_install_instructions(monkeypatch, interface_kind):
    def regular_attention(q):
        return q

    def legacy_attention(q, image_token_tag=None):
        return q

    attention_func = {"missing": None, "regular": regular_attention, "legacy_tag": legacy_attention}[interface_kind]
    with pytest.warns(UserWarning, match="Neo FA3 is unavailable") as warnings:
        module = _load_neo(monkeypatch, attention_func)

    assert not module.HAS_FLASH_ATTN_INTERFACE
    assert module.flash_attn_with_kvcache_neo is None
    message = str(warnings[0].message)
    assert "https://github.com/WANDY666/flash-attention.git" in message
    assert "cd flash-attention/hopper" in message
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in message
    assert "--no-deps --force-reinstall" in message
    assert "--llm_prefill_att_backend triton" in message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("page_size", [1, 16, 64])
def test_prefill_initializes_paged_kv_metadata_and_forwards_image_ends(monkeypatch, page_size):
    def indices(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    image_ends = indices([2 * page_size, 2 * page_size, 0])
    # Each request uses two nonadjacent physical pages, in a different order.
    req_to_token_indexs = indices(
        [
            list(range(2 * page_size, 3 * page_size)) + list(range(page_size)),
            list(range(3 * page_size, 4 * page_size)) + list(range(page_size, 2 * page_size)),
        ]
    )
    infer_state = SimpleNamespace(
        batch_size=2,
        max_kv_seq_len=2 * page_size,
        max_q_seq_len=2,
        input_ids=torch.zeros(3, dtype=torch.int64, device="cuda"),
        b_req_idx=indices([1, 0]),
        b_seq_len=indices([2 * page_size, 2 * page_size - 1]),
        b1_cu_q_seq_len=indices([0, 2, 3]).long(),
        b1_cu_kv_seq_len=indices([0, 2 * page_size, 4 * page_size - 1]).long(),
        b_image_token_end=image_ends,
        req_manager=SimpleNamespace(req_to_token_indexs=req_to_token_indexs),
    )
    backend = SimpleNamespace(uses_causal_attention=lambda: True, infer_page_size=page_size)
    state = neo.NeoFa3AttBackend.create_att_prefill_state(backend, infer_state)
    state.init_state()

    q = torch.randn(3, 2, 16, device="cuda")
    k = torch.randn(4 * page_size, 1, 16, device="cuda")
    v = torch.randn_like(k)
    output = torch.empty_like(q)

    def neo_attention(*, image_token_end, **kwargs):
        assert image_token_end is image_ends
        assert kwargs["q"] is q
        assert kwargs["k_cache"].shape == kwargs["v_cache"].shape == (4, page_size, 1, 16)
        torch.testing.assert_close(kwargs["page_table"], indices([[3, 1], [2, 0]]))
        token_indices = req_to_token_indexs[infer_state.b_req_idx]
        torch.testing.assert_close(kwargs["k_cache"][kwargs["page_table"]].flatten(1, 2), k[token_indices])
        torch.testing.assert_close(kwargs["v_cache"][kwargs["page_table"]].flatten(1, 2), v[token_indices])
        torch.testing.assert_close(kwargs["cu_seqlens_q"], indices([0, 2, 3]))
        torch.testing.assert_close(kwargs["cu_seqlens_k_new"], indices([0, 2 * page_size, 4 * page_size - 1]))
        assert kwargs["cache_seqlens"] is infer_state.b_seq_len
        assert kwargs["max_seqlen_q"] == 2
        assert kwargs["causal"] is True
        assert "image_token_tag" not in kwargs
        return output

    monkeypatch.setattr(neo, "HAS_FLASH_ATTN_INTERFACE", True)
    monkeypatch.setattr(neo, "flash_attn_with_kvcache_neo", neo_attention)
    assert state.prefill_att(q, k, v) is output


def test_prefill_missing_extension_reports_install_instructions(monkeypatch):
    monkeypatch.setattr(neo, "HAS_FLASH_ATTN_INTERFACE", False)
    state = neo.NeoFa3PrefillAttState()
    with pytest.raises(ImportError, match="WANDY666/flash-attention"):
        state.prefill_att(None, None, None)
