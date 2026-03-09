import math
import torch
import torch.nn as nn
import os

@torch.no_grad()
def pack_4bit_to_int32(q: torch.Tensor) -> torch.Tensor:
    assert q.dtype == torch.int32, "q should be int32 (store 0..15 values)"
    L = q.shape[-1]
    assert L % 8 == 0, "last dim must be multiple of 8 for packing"

    groups = L // 8
    order_idx = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=q.device, dtype=torch.long)
    shifts = (order_idx.to(torch.int32) * 4).view(1, 1, 8)

    q_view = q.view(q.shape[0], groups, 8)
    packed = torch.sum(q_view << shifts, dim=-1).to(torch.int32)
    return packed.contiguous()


def _pack_4bit_to_int32_reference(q: torch.Tensor) -> torch.Tensor:
    assert q.dtype == torch.int32, "q should be int32 (store 0..15 values)"
    L = q.shape[-1]
    assert L % 8 == 0, "last dim must be multiple of 8 for packing"

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    out_a = torch.zeros((q.shape[0], q.shape[1] // 8), device=q.device, dtype=torch.int32)
    for col in range(q.shape[1] // 8):
        for i in range(8):
            out_a[:, col] |= (q[:, col * 8 + i] << (order_map[i] * 4))
    return out_a


def test_pack_4bit_to_int32_equivalence() -> None:
    torch.manual_seed(0)
    shapes = [(1, 8), (2, 16), (3, 24), (4, 32), (768, 2048)]
    for rows, cols in shapes:
        q = torch.randint(0, 16, (rows, cols), dtype=torch.int32)
        q = torch.ones_like(q).to(torch.int32) * 15
        expected = _pack_4bit_to_int32_reference(q)
        actual = pack_4bit_to_int32(q)
        assert torch.equal(actual, expected), f"Mismatch for shape {(rows, cols)}"


# test_pack_4bit_to_int32_equivalence()

@torch.no_grad()
def quantize_awq_4bit_per_group(weight: torch.Tensor, group_size: int = 128, eps: float = 1e-8):
    # assert weight.dim() == 2, "expect [out_features, in_features]"
    out_features, in_features = weight.shape

    # 以 in_features 方向分组
    num_groups = math.ceil(in_features / group_size)
    # 变形为 [out, num_groups, group_size]
    W = weight.view(out_features, num_groups, group_size)

    # 计算每个 (out, group) 的 min/max、scale/zero
    w_min = W.amin(dim=-1)  # [out, num_groups]
    w_max = W.amax(dim=-1)  # [out, num_groups]

    # 4bit: 0..15
    q_levels = 15.0
    scales = (w_max - w_min) / q_levels
    scales = torch.clamp(scales, min=eps)

    # 非对称零点，取整到 [0, 15]
    z = (-torch.round(w_min / scales)).clamp(0, 15)  # [out, num_groups]
    z_int = z.to(torch.int32)

    # 量化
    # q = round(W / scale + z), 按广播对齐到 [out, num_groups, group_size]
    q = (torch.round(W / scales.unsqueeze(-1)) + z.unsqueeze(-1)).clamp(0, 15)

    # 转 int32 方便按 4bit 打包
    q = q.to(torch.int32).view(out_features, in_features)
    q = q.transpose(0, 1).contiguous()

    # 每组按最后一维每 8 个 4bit 打 1 个 int32
    qweight_packed = pack_4bit_to_int32(q)

    # scales 转 bfloat16
    scales_bf16 = scales
    scales_bf16 = scales_bf16.transpose(0, 1).contiguous()

    z_int = z_int.transpose(0, 1).contiguous()
    qzeros_packed = pack_4bit_to_int32(z_int)
    return qweight_packed.contiguous(), scales_bf16.contiguous(), qzeros_packed.contiguous()

# 在这里制定源路径，和目标路径即可
source_path = "/data/Qwen3-Omni-30B-A3B-Instruct"
dst_path = "/data/Qwen3-Omni-30B-A3B-Instruct-AWQ/"
import shutil
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

all_files = os.listdir(source_path)
for file in tqdm(all_files):
    if not file.endswith(".safetensors"):
        shutil.copy(os.path.join(source_path, file), os.path.join(dst_path, file))
        continue
    model = safe_open(os.path.join(source_path, file), "pt", "cpu")
    new_model = {}
    for k in model.keys():
        if ".buf" in k:
            continue
        if "mlp.gate.weight" in k:
            new_model[k] = model.get_tensor(k)
            continue
        if "proj" in k and "thinker.model" in k:
            proj_weight = model.get_tensor(k)
            qweight, scales, qzeros = quantize_awq_4bit_per_group(proj_weight, group_size=128)
            new_model[k.replace(".weight", ".qweight")] = qweight
            new_model[k.replace(".weight", ".scales")] = scales
            new_model[k.replace(".weight", ".qzeros")] = qzeros
        else:
            new_model[k] = model.get_tensor(k)
            continue
    del model
    save_file(new_model, os.path.join(dst_path, file))