"""Native quantized state must agree with the separate CPU service's byte layout."""

from dataclasses import asdict
import json

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.core.formats.kv.group_quantized import GroupQuantizedKVWriter
from lightllm.core.contracts.state import StateSpec
from lightllm.core.storage.attention import DenseStateStore
from lightllm.common.transfer import Transfer
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.cpu_cache_layout import build_cpu_cache_layouts
from lightllm.utils.envs_utils import get_env_start_args
from unit_tests.common.execution.test_group_quantized import make_format


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group", [8, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_quantized_cpu_service_and_pd_round_trip(bits, group, dtype, monkeypatch, tmp_path):
    config = dict(
        model_type="llama", hidden_size=256, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = StartArgs(
        model_dir=str(tmp_path),
        data_type="float16" if dtype == torch.float16 else "bfloat16",
        llm_kv_type=f"int{bits}kv",
        llm_kv_quant_group_size=group,
        page_size=4,
        cpu_cache_token_page_size=4,
        pd_kv_page_size=4,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    monkeypatch.setattr("lightllm.utils.llm_utils.get_llm_model_class", lambda: LlamaTpPartModel)
    get_env_start_args.cache_clear()
    try:
        stores, managers = [], []
        declarations = tuple(StateSpec("main", f"kv.{i}", i, (i,), (2, 64)) for i in range(2))
        for _ in range(2):
            manager = AttentionStateManager(2)
            store = DenseStateStore(declarations, make_format(bits, dtype, group), manager)
            manager.finish_state_registration()
            for state in manager.get_full_attention_states():
                state.device = torch.device("cuda")
                state.allocate_storage(32)
            stores.append(store)
            managers.append(manager)
        src, dst = (store.views() for store in stores)
        src_ids = torch.tensor([3, 13, 7, 9, 17], device="cuda")
        dst_ids = torch.tensor([2, 8, 15, 4, 21], device="cuda")
        for view in src.values():
            GroupQuantizedKVWriter(view)(token_indices=src_ids, kv=torch.randn(5, 4, 64, dtype=dtype, device="cuda"))

        def check():
            for key, view in src.items():
                for original, restored in zip(view.fields, dst[key].fields):
                    torch.testing.assert_close(restored[dst_ids], original[src_ids], rtol=0, atol=0)

        cpu_src, cpu_dst = (manager.get_cpu_cache_operator() for manager in managers)
        layouts = build_cpu_cache_layouts()
        assert layouts == cpu_src.get_page_layouts() == cpu_dst.get_page_layouts()
        pages = torch.zeros(2, cpu_src.get_page_bytes(), dtype=torch.uint8, pin_memory=True)
        write = cpu_src.plan_write_req(src_ids, pages, [1, 0], layouts)
        read = cpu_dst.plan_read_req(pages, [1, 0], dst_ids, layouts)
        transfer = Transfer()
        transfer.submit(write).destination_ready.synchronize()
        transfer.submit(read).destination_ready.synchronize()
        check()

        pd_src, pd_dst = (manager.get_pd_operator() for manager in managers)
        pd_layouts = pd_src.get_pd_full_att_page_layouts()
        assert pd_layouts == pd_dst.get_pd_full_att_page_layouts()
        pages = torch.empty(2, pd_src.get_page_bytes(), dtype=torch.uint8, device="cuda")
        for view in dst.values():
            for field in view.fields:
                field.zero_()
        plans = []
        for i in range(2):
            tokens = slice(i * 4, (i + 1) * 4)
            plans.extend(
                (
                    pd_src.plan_write_page(pages, i, "kv", pd_layouts, token_indices=src_ids[tokens]),
                    pd_dst.plan_read_page(pages, i, "kv", pd_layouts, token_indices=dst_ids[tokens]),
                )
            )
        for manager in managers:
            monkeypatch.setattr(manager, "get_full_attention_states", lambda *a: pytest.fail("unbound copy"))
        for plan in plans:
            transfer.submit(plan).destination_ready.synchronize()
        check()
    finally:
        get_env_start_args.cache_clear()
