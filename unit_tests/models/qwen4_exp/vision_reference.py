import torch
import torch.nn.functional as F


def patch_coordinates(grid, merge, device):
    coordinates = []
    for frames, height, width in grid.cpu().tolist():
        coordinates.extend(
            (y + dy, x + dx)
            for _ in range(frames)
            for y in range(0, height, merge)
            for x in range(0, width, merge)
            for dy in range(merge)
            for dx in range(merge)
        )
    return torch.tensor(coordinates, dtype=torch.int64, device=device)


def interpolate_positions(table, grid, merge):
    side = int(table.shape[0] ** 0.5)
    result = []
    for frames, height, width in grid.cpu().tolist():
        coords = patch_coordinates(torch.tensor([[frames, height, width]]), merge, table.device)
        ys = torch.linspace(0, side - 1, height, device=table.device)[coords[:, 0]]
        xs = torch.linspace(0, side - 1, width, device=table.device)[coords[:, 1]]
        y0, x0 = ys.floor().long(), xs.floor().long()
        y1, x1 = (y0 + 1).clamp_max(side - 1), (x0 + 1).clamp_max(side - 1)
        dy, dx = ys - y0, xs - x0
        indices = torch.stack([y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1], -1)
        coefficients = torch.stack([(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx], -1)
        result.append((table[indices] * coefficients[:, :, None]).sum(1).to(table.dtype))
    return torch.cat(result)


def vision_reference(pixels, grid, weights, config):
    hidden, heads = config["hidden_size"], config["num_heads"]
    patch, temporal, merge = config["patch_size"], config["temporal_patch_size"], config["spatial_merge_size"]
    values = F.conv3d(
        pixels.view(-1, config["in_channels"], temporal, patch, patch),
        weights["patch_embed.proj.weight"],
        weights["patch_embed.proj.bias"],
        stride=(temporal, patch, patch),
    ).view(-1, hidden)
    values = values + interpolate_positions(weights["pos_embed.weight"], grid, merge)
    dim = hidden // heads
    theta = config.get("rope_parameters", {}).get("rope_theta", 10000.0)
    invfreq = theta ** (-torch.arange(0, dim // 2, 2, device=pixels.device).float() / (dim // 2))
    coordinates = patch_coordinates(grid, merge, pixels.device)
    angles = (coordinates[:, :, None] * invfreq).flatten(1).repeat(1, 2)
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]

    def norm(value, prefix):
        return F.layer_norm(value, (value.shape[-1],), weights[prefix + ".weight"], weights[prefix + ".bias"], 1e-6)

    def rotate(value):
        value = value.float()
        a, b = value.chunk(2, -1)
        return (value * cos + torch.cat([-b, a], -1) * sin).to(pixels.dtype)

    lengths = [height * width for frames, height, width in grid.cpu().tolist() for _ in range(frames)]
    for layer in range(config["depth"]):
        prefix = f"blocks.{layer}"
        qkv = F.linear(
            norm(values, prefix + ".norm1"), weights[prefix + ".attn.qkv.weight"], weights[prefix + ".attn.qkv.bias"]
        )
        q, k, v = qkv.view(-1, 3, heads, dim).unbind(1)
        q, k = rotate(q), rotate(k)
        outputs, offset = [], 0
        for count in lengths:
            outputs.append(
                F.scaled_dot_product_attention(
                    q[offset : offset + count].transpose(0, 1)[None],
                    k[offset : offset + count].transpose(0, 1)[None],
                    v[offset : offset + count].transpose(0, 1)[None],
                )[0].transpose(0, 1)
            )
            offset += count
        attention = torch.cat(outputs).reshape(-1, hidden)
        values = values + F.linear(
            attention, weights[prefix + ".attn.proj.weight"], weights[prefix + ".attn.proj.bias"]
        )
        mlp = F.linear(
            norm(values, prefix + ".norm2"),
            weights[prefix + ".mlp.linear_fc1.weight"],
            weights[prefix + ".mlp.linear_fc1.bias"],
        )
        mlp = F.gelu(mlp, approximate="tanh" if config["hidden_act"] == "gelu_pytorch_tanh" else "none")
        values = values + F.linear(
            mlp, weights[prefix + ".mlp.linear_fc2.weight"], weights[prefix + ".mlp.linear_fc2.bias"]
        )
    values = norm(values, "merger.norm").view(-1, hidden * merge * merge)
    values = F.gelu(F.linear(values, weights["merger.linear_fc1.weight"], weights["merger.linear_fc1.bias"]))
    return F.linear(values, weights["merger.linear_fc2.weight"], weights["merger.linear_fc2.bias"])
