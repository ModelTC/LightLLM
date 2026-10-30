from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import build_hybrid_resources
from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from typing import TYPE_CHECKING
from unit_tests.common.att_resources.model_helpers import main_linear_state
import dataclasses
import json

import pytest
import torch

from lightllm.common.att_resources.allocators import FullAttKVAllocator
from unit_tests.common.att_resources.model_helpers import create_full_attention_roles
from lightllm.common.att_resources.full_att import (
    GQAFullAttentionState,
    FullAttentionConfig,
)
from lightllm.common.att_resources.full_att import PackedMLAFullAttentionState, PackedMLAAttentionConfig
from lightllm.models.qwen3next.config import Qwen3NextResourceConfig
from lightllm.models.glm5_next.config import Glm5NextResourceConfig
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args

if TYPE_CHECKING:
    from easydict import EasyDict


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_resources(
    glm: bool,
    tp: int,
    rank: int,
    mtp: int,
    draft_layers: int = 0,
):
    gk, gv = (4, 4) if glm else (2, 4)
    cls = Glm5NextResourceConfig if glm else Qwen3NextResourceConfig
    config = cls(
        tp_world_size=tp,
        full_att_all_num_kv_heads=1 if glm else 2,
        full_att_head_dim=512 if glm else 16,
        global_linear_k_heads=gk,
        global_linear_v_heads=gv,
        head_linear_k_dim=8,
        head_linear_v_dim=8,
        conv_kernel_size=3,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=4 if glm else 2,
        all_layer_num=4,
    )
    if glm:
        full = PackedMLAFullAttentionState(
            PackedMLAAttentionConfig(
                logical_layer_groups=tuple((layer,) for layer in (3, *range(4, 4 + draft_layers))),
                global_head_num=1,
                head_dim=584,
                dtype=torch.bfloat16,
                tp_rank=rank,
                tp_world_size=tp,
                kv_dim=512,
            ),
             page_size=4,
        )
        full.initialize_storage(16)
    else:
        full = GQAFullAttentionState(
            FullAttentionConfig(
                logical_layer_groups=tuple((layer,) for layer in (1, 3, *range(4, 4 + draft_layers))),
                global_head_num=2,
                head_dim=16,
                dtype=torch.bfloat16,
                tp_rank=rank,
                tp_world_size=tp,
            ),
             page_size=4,
        )
        full.initialize_storage(16)
    main, draft = create_full_attention_roles(full, 4)
    resources = build_hybrid_resources(
        config=config,
        request_capacity=3,
        mtp_step=mtp,
        tp_rank=rank,
        full_state=main,
        full_allocator=FullAttKVAllocator(16, 4),
        draft_full_state=draft,
        big_page_capacity=4,
        small_page_capacity=2,
    )
    return resources, resources.get_full_attention_states("main")[0]


@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("mtp", [0, 2])
def test_model_cpu_layout_roundtrip_tp_shards_replicas_and_tail(
    glm: bool,
    tp: int,
    mtp: int,
    cpu_args: EasyDict,
):
    sources = [make_resources(
        glm=glm,
        tp=tp,
        rank=rank,
        mtp=mtp,
    ) for rank in range(tp)]
    targets = [make_resources(
        glm=glm,
        tp=tp,
        rank=rank,
        mtp=mtp,
    ) for rank in range(tp)]
    expected = []
    for rank, (resources, full) in enumerate(sources):
        cfg = main_linear_state(resources).config
        for storage_buffer in full_state_test_buffers(full).values():
            storage_buffer.fill_(1 if glm else rank // max(1, tp // 2) + 1)
        state = main_linear_state(resources)
        local_start = 0
        for local_heads, global_heads in (
            (cfg.num_k_heads, cfg.global_k_heads),
            (cfg.num_k_heads, cfg.global_k_heads),
            (cfg.num_v_heads, cfg.global_v_heads),
        ):
            start = rank * local_heads // max(1, tp * local_heads // global_heads)
            for h in range(local_heads):
                state.conv_state[:, 0, local_start + h * 8 : local_start + (h + 1) * 8].fill_(start + h + 3)
            local_start += local_heads * 8
        for h in range(cfg.num_v_heads):
            state.ssm_state[:, : mtp + 1, h].fill_(rank * cfg.num_v_heads + h + 7)
        slot = resources.bounded_att_big_page_allocator.alloc(1)[0]
        resources.save_bounded_att_pages([0], page_kind="big", slot_ids=[slot])
        torch.cuda.synchronize()
        expected.append((state.big_pages_pool.conv_state[slot].clone(), state.big_pages_pool.ssm_state[slot].clone()))
    size = sources[0][0].get_cpu_page_bytes()
    buffer = torch.zeros((2, size), dtype=torch.uint8, pin_memory=True)
    for resources, full in sources:
        resources.write_req_to_cpu(
            token_indices=torch.arange(8, dtype=torch.long, device="cuda"),
            cpu_page_buffers=buffer,
            cpu_page_indices=[1],
            layouts=resources.get_cpu_transfer_layouts(),
            big_page_slot_ids=[0],
        )
    torch.cuda.synchronize()
    for rank, (resources, full) in enumerate(targets):
        if glm:
            resources.main_kpool_c4_bf16_state.tail_buffer.fill_(13)
        slot = resources.bounded_att_big_page_allocator.alloc(1)[0]
        resources.read_req_from_cpu(
            cpu_page_buffers=buffer,
            cpu_page_indices=[1],
            token_indices=torch.arange(
                start=8,
                end=16,
                dtype=torch.long,
                device="cuda",
            ),
            layouts=resources.get_cpu_transfer_layouts(),
            big_page_slot_ids=[slot],
            req_idx=1,
        )
        torch.testing.assert_close(main_linear_state(resources).conv_state[:, 1, :, :2].cpu(), expected[rank][0])
        torch.testing.assert_close(main_linear_state(resources).ssm_state[:, mtp + 1].cpu(), expected[rank][1])
        src_fields = full_state_test_fields(sources[rank][1], full_state_test_buffers(sources[rank][1]))
        dst_fields = full_state_test_fields(full, full_state_test_buffers(full))
        for name in src_fields:
            assert torch.equal(
                src_fields[name].tensor[:, :8].contiguous().view(torch.uint8),
                dst_fields[name].tensor[:, 8:16].contiguous().view(torch.uint8),
            )
        if glm:
            assert not resources.main_kpool_c4_bf16_state.tail_buffer[:, 1].any()
            assert (resources.main_kpool_c4_bf16_state.tail_buffer[:, [0, 2]] == 13).all()
    assert not buffer[0].any()


def test_cpu_layout_query_does_not_require_cpu_cache_enabled(monkeypatch: pytest.MonkeyPatch):
    args = StartArgs(enable_cpu_cache=False, cpu_cache_token_page_size=8)  # 未启用缓存时也可查询全局页配置。
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(args)))
    get_env_start_args.cache_clear()
    resources, _ = make_resources(
        glm=True,
        tp=1,
        rank=0,
        mtp=0,
    )
    assert all(layout.tokens_per_page == 8 for layout in resources.get_cpu_transfer_layouts().values())
    get_env_start_args.cache_clear()
