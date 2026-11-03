from lightllm.common.attention_state.storage import allocate_state_storage
import json
import os
import statistics

import torch
import triton.testing

from benchmarks.attention_plan import eager_time
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.attention_state.full_att.gqa.gqa_transfer import transfer_gqa_pages
from lightllm.common.attention_state.token_copy import TokenPageCopyPlan, TransferBatch
from lightllm.utils.envs_utils import get_env_start_args


def main():
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(dict(cpu_cache_token_page_size=128, pd_kv_page_size=128))
    get_env_start_args.cache_clear()
    results = []
    for count in (1, 4):
        manager = AttentionStateManager(request_capacity=1)
        for i in range(count):
            state = GQAFullAttentionState(
                GQAAttentionConfig(((i,),), global_head_num=4, head_dim=128, dtype=torch.bfloat16), device="cuda"
            )
            allocate_state_storage(state, 4096)
            state.k_storage.normal_()
            state.v_storage.normal_()
            manager.register_state(f"draft{i}", state)
        manager.finish_state_registration()
        states = manager.get_full_attention_states()
        layouts = manager.get_cpu_cache_operator().get_page_layouts()
        page_bytes = max(layout.byte_end for layout in layouts.values())
        for device in ("cuda", "cpu"):
            pages = torch.zeros((36, page_bytes), dtype=torch.uint8, device=device, pin_memory=device == "cpu")
            plans = [
                TokenPageCopyPlan.bind(s.get_transfer_fields(), layouts[s.state_name], pages, True) for s in states
            ]
            for tokens in (128, 1023, 4096):
                ids = torch.randperm(4096, device="cuda", dtype=torch.int32)[:tokens]
                page_ids = list(range((tokens + 127) // 128))[::-1]

                def legacy():
                    for state in states:
                        transfer_gqa_pages(
                            state.k_storage,
                            state.v_storage,
                            ids,
                            pages,
                            page_ids,
                            layouts[state.state_name],
                            write=True,
                        )

                def bound():
                    batch = TransferBatch.prepare(ids, page_ids)
                    for plan in plans:
                        plan.run(batch)

                pages.zero_()
                legacy()
                torch.cuda.synchronize()
                expected = pages.clone()
                pages.zero_()
                bound()
                torch.cuda.synchronize()
                torch.testing.assert_close(pages, expected, rtol=0, atol=0)
                timings = {name: [] for name in ("legacy", "bound")}
                for trial in range(5):
                    for name, run in [("legacy", legacy), ("bound", bound)][:: 1 if trial % 2 else -1]:
                        timings[name].append(eager_time(run, repeats=100))

                batch = TransferBatch.prepare(ids, page_ids)

                def run_kernels():
                    for plan in plans:
                        plan.run(batch)

                graph_us = 1000 * triton.testing.do_bench_cudagraph(run_kernels, rep=100)
                results.append(
                    dict(
                        states=count,
                        tokens=tokens,
                        page_device=device,
                        graph_us=graph_us,
                        **{f"{name}_eager_us": statistics.median(values) for name, values in timings.items()},
                    )
                )
    print(json.dumps(dict(device=torch.cuda.get_device_name(), torch=torch.__version__, results=results), indent=2))


if __name__ == "__main__":
    main()
