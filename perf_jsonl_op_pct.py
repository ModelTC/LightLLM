# 统计jsonl文件中每种OP的时间占比，默认跳过前两轮推理（warmup），可以通过第二个参数调整跳过的轮数

import sys, json
from typing import DefaultDict

jsonl_file = sys.argv[1]
skip_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print("===", jsonl_file)

OMIT_FIRST_INFER = skip_layers

op_times = DefaultDict(float)
layer_times = []
with open(jsonl_file, "r") as f:
    layer_cnt = 0
    for i, line in enumerate(f):
        if line[0] == '#':
            layer_cnt += 1
        if layer_cnt > OMIT_FIRST_INFER:
            omit_first = i
            break

    # Rewind to start reading again
    f.seek(0)
    for i, line in enumerate(f):
        if i < omit_first:
            continue
        if line[0] == '#':
            continue
        # {"name": "mm", "type": "GEMM_OP", "shapes": {"m": 1024, "k": 8192, "n": 29568}, "depth": 2, "t_start_ms": 14.655648231506348, "t_elapsed_ms": 4.317984104156494, "marker": "_context_forward1"}
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line ({i}):", line)
            continue
        if data["type"].endswith("_OP"):
            op_times[data["type"]] += data["t_elapsed_ms"]
        if data["type"] == "LAYER":
            layer_times.append(data["t_elapsed_ms"])


all_op_time = sum(op_times.values())
print(f"Total time: {all_op_time} ms")
results = [f"{op_type}: {time / all_op_time * 100:.2f}%" for op_type, time in op_times.items()]
results.sort()
for result in results:
    print(result)

print("Average layer time:", sum(layer_times) / len(layer_times) if layer_times else 0)
print("Median layer time:", sorted(layer_times)[len(layer_times) // 2] if layer_times else 0)
