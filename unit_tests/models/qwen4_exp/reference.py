import torch
import torch.nn.functional as F

from .helpers import reference_hyper, reference_moe


def grouped_norm(x, weight, config):
    values = x.float().unflatten(-1, (config["hc_count"], config["hidden_size"]))
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + config["rms_norm_eps"])
    return (normalized.flatten(-2) * (1 + weight.cuda().float())).to(x.dtype)


def reference_ple(residual, tokens, weights, config, ngram_history, conv_history):
    prefix = "model.language_model.layers.1.ple"
    history = torch.cat([ngram_history, tokens[:, None]], dim=1)
    eos = config["eos_token_id"]
    eos = eos[0] if isinstance(eos, list) else eos
    context = config["ngram_size"] - 1
    shifted = [tokens]
    for distance in range(1, context + 1):
        preceding = history[:, context - distance : context]
        shifted.append(torch.where((preceding == eos).any(-1), eos, history[:, context - distance]))
    multipliers = weights[prefix + ".ple_embedding.layer_multipliers"].cuda()
    sizes = weights[prefix + ".ple_embedding.ngram_heads_vocab_sizes"].cuda()
    offsets = weights[prefix + ".ple_embedding.ngram_heads_offsets"].cuda()
    indices = []
    for head in range(sizes.numel()):
        order = head // config["heads_per_ngram"] + 2
        mixed = shifted[0] * multipliers[0]
        for position in range(1, order):
            mixed = mixed ^ (shifted[position] * multipliers[position])
        indices.append(mixed.remainder(sizes[head]) + offsets[head])
    embedding = reference_ngram_lookup(torch.stack(indices, -1), weights, prefix).flatten(-2)
    branches, hidden = config["hc_count"], config["hidden_size"]
    key = grouped_norm(
        F.linear(embedding, weights[prefix + ".key_proj.weight"].cuda()), weights[prefix + ".norm_key.weight"], config
    ).view(-1, branches, hidden)
    query = grouped_norm(residual, weights[prefix + ".norm_query.weight"], config).view(-1, branches, hidden)
    gate = (key * query).sum(-1, keepdim=True) / (hidden**0.5)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    value = F.linear(embedding, weights[prefix + ".value_proj.weight"].cuda())
    gated = (gate.sigmoid() * value[:, None]).flatten(-2)
    normalized = grouped_norm(gated, weights[prefix + ".norm_conv.weight"], config)
    full = torch.cat([conv_history, normalized[:, None]], dim=1)
    dilation = config["ngram_size"]
    conv = F.conv1d(
        full.transpose(1, 2), weights[prefix + ".conv1d.weight"].cuda(), groups=branches * hidden, dilation=dilation
    ).squeeze(-1)
    conv_context = (config["ple_conv_kernel_size"] - 1) * dilation
    return (
        gated + F.silu(conv),
        history[:, -context:].contiguous(),
        full[:, -conv_context:].contiguous() if conv_context else full[:, :0].contiguous(),
    )


def reference_ngram_lookup(indices, weights, prefix):
    name = prefix + ".ple_embedding.ngram_embedding"
    first = weights[name + ".shard_0.weight"]
    shard_rows, dim = first.shape
    flat = indices.cpu().flatten()
    shards = flat // shard_rows
    result = torch.empty(flat.numel(), dim, dtype=first.dtype)
    for shard in shards.unique().tolist():
        positions = torch.where(shards == shard)[0]
        source = weights[name + f".shard_{shard}.weight"]
        rows = (flat[positions] % shard_rows).to(source.device)
        result[positions] = source[rows].cpu()
    return result.to(indices.device).view(*indices.shape, dim)


def reference_gdn(mixed, weights, config, conv_history, recurrent):
    prefix = "model.language_model.layers.1.linear_attn"
    key_heads, value_heads = config["linear_num_key_heads"], config["linear_num_value_heads"]
    key_dim, value_dim = config["linear_key_head_dim"], config["linear_value_head_dim"]
    kd, vd = key_heads * key_dim, value_heads * value_dim
    qkv = F.linear(mixed, weights[prefix + ".in_proj_qkv.weight"].cuda())
    full = torch.cat([conv_history, qkv[:, :, None]], dim=-1)
    conv = (full.float() * weights[prefix + ".conv1d.weight"].squeeze(1).cuda().float()[None]).sum(-1).to(mixed.dtype)
    conv = F.silu(conv)
    q, k, v = torch.split(conv, [kd, kd, vd], dim=-1)
    q, k = q.view(-1, key_heads, key_dim).float(), k.view(-1, key_heads, key_dim).float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(value_heads // key_heads, dim=1) / (key_dim**0.5)
    k = k.repeat_interleave(value_heads // key_heads, dim=1)
    v = v.view(-1, value_heads, value_dim).float()
    a = F.linear(mixed, weights[prefix + ".in_proj_a.weight"].cuda())
    b = F.linear(mixed, weights[prefix + ".in_proj_b.weight"].cuda())
    decay = -weights[prefix + ".A_log"].cuda().float().exp() * F.softplus(
        a.float() + weights[prefix + ".dt_bias"].cuda()
    )
    beta = b.sigmoid().float()
    next_state = recurrent * decay.exp()[:, :, None, None]
    memory = (next_state * k[:, :, :, None]).sum(-2)
    delta = (v - memory) * beta[:, :, None]
    next_state = next_state + k[:, :, :, None] * delta[:, :, None, :]
    core = (next_state * q[:, :, :, None]).sum(-2).to(mixed.dtype)
    normed = core.float() * torch.rsqrt(core.float().square().mean(-1, keepdim=True) + 1e-6)
    weighted = weights[prefix + ".norm.weight"].cuda() * normed.to(mixed.dtype)
    z = F.linear(mixed, weights[prefix + ".in_proj_z.weight"].cuda()).view(-1, value_heads, value_dim)
    gated = (weighted * z.float().sigmoid()).to(mixed.dtype)
    output = F.linear(gated.flatten(1), weights[prefix + ".out_proj.weight"].cuda())
    return output, full[:, :, -3:].contiguous(), next_state


def reference_step(residual, tokens, source, config, ngram, ple, conv, recurrent):
    lexical, ngram, ple = reference_ple(residual, tokens, source, config, ngram, ple)
    expected = residual + lexical
    mixed, gates = reference_hyper(expected, source, "model.language_model.layers.1.attn_hyper_connection", config)
    output, conv, recurrent = reference_gdn(mixed, source, config, conv, recurrent)
    expected = expected + (output[:, None] * gates[:, :, None]).flatten(1)
    mixed, gates = reference_hyper(expected, source, "model.language_model.layers.1.mlp_hyper_connection", config)
    output = reference_moe(mixed, source, "model.language_model.layers.1", config)
    expected = expected + (output[:, None] * gates[:, :, None]).flatten(1)
    return expected, ngram, ple, conv, recurrent


def norm(x, weight, eps=1e-6):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * (1 + weight.cuda().float())).to(x.dtype)


def rotate(x, cos, sin):
    width = cos.shape[-1]
    a, b = x[..., :width].chunk(2, -1)
    return torch.cat([x[..., :width] * cos + torch.cat([-b, a], -1) * sin, x[..., width:]], -1)


def reference_layer(residual, weights, config, cos, sin, lengths):
    prefix = "model.language_model.layers.3"
    mixed, gates = reference_hyper(residual, weights, prefix + ".attn_hyper_connection", config)
    dim, heads, kvheads = config["head_dim"], config["num_attention_heads"], config["num_key_value_heads"]
    qg = F.linear(mixed, weights[prefix + ".self_attn.q_proj.weight"].cuda()).view(-1, heads, dim * 2)
    query, gate = qg.chunk(2, -1)
    key = F.linear(mixed, weights[prefix + ".self_attn.k_proj.weight"].cuda()).view(-1, kvheads, dim)
    value = F.linear(mixed, weights[prefix + ".self_attn.v_proj.weight"].cuda()).view(-1, kvheads, dim)
    query = rotate(norm(query, weights[prefix + ".self_attn.q_norm.weight"]), cos[:, None], sin[:, None])
    key = rotate(norm(key, weights[prefix + ".self_attn.k_norm.weight"]), cos[:, None], sin[:, None])
    projected = F.linear(mixed, weights[prefix + ".self_attn.indexer.index_qk_proj.weight"].cuda())
    ih, idim = config["indexer_n_heads"], config["indexer_head_dim"]
    iq = projected[:, : ih * idim].view(-1, ih, idim)
    iq = rotate(norm(iq, weights[prefix + ".self_attn.indexer.q_layernorm.weight"]), cos[:, None], sin[:, None])
    raw = projected[:, ih * idim :]
    ratio, budget = config["indexer_compress_ratio"], config["indexer_budget"] // config["indexer_compress_ratio"]
    result = []
    start = 0
    for count in lengths:
        for position in range(count):
            complete = (position + 1) // ratio
            selected = []
            if complete:
                groups = raw[start : start + complete * ratio].view(complete, ratio, idim).float().mean(1).to(raw.dtype)
                pooled = norm(groups, weights[prefix + ".self_attn.indexer.k_layernorm.weight"])
                pooled = rotate(
                    pooled, cos[start : start + complete * ratio : ratio], sin[start : start + complete * ratio : ratio]
                )
                scores = (iq[start + position].float() @ pooled.float().T).relu().sum(0) / (idim**0.5)
                blocks = scores.topk(min(complete, budget)).indices
                selected = (blocks[:, None] * ratio + torch.arange(ratio, device=residual.device)).flatten().tolist()
            selected += list(range(complete * ratio, position + 1))
            ids = torch.tensor(selected, device=residual.device, dtype=torch.int64) + start
            out = F.scaled_dot_product_attention(
                query[start + position].unsqueeze(0).unsqueeze(2),
                key[ids].transpose(0, 1).unsqueeze(0),
                value[ids].transpose(0, 1).unsqueeze(0),
                enable_gqa=True,
            )
            result.append(out[0, :, 0])
        start += count
    attention = torch.stack(result)
    attention = attention * gate.sigmoid()
    output = F.linear(attention.flatten(1), weights[prefix + ".self_attn.o_proj.weight"].cuda())
    residual = residual + (output[:, None, :] * gates[:, :, None]).flatten(1)
    mixed, gates = reference_hyper(residual, weights, prefix + ".mlp_hyper_connection", config)
    update = reference_moe(mixed, weights, prefix, config)
    return residual + (update[:, None, :] * gates[:, :, None]).flatten(1)


def reference_model(tokens, cos, sin, lengths, weights, config, input_embeddings=None):
    if input_embeddings is None:
        input_embeddings = F.embedding(tokens, weights["model.language_model.embed_tokens.weight"].cuda())
    residual = input_embeddings.repeat(1, config["hc_count"])
    for layer in range(config["n_layer"]):
        prefix = f"model.language_model.layers.{layer}"
        if config["layer_types"][layer] == "full_attention":
            residual = reference_layer(residual, weights, config, cos, sin, lengths)
            continue
        if layer + 1 in config["ple_layer_ids"]:
            updates, start = [], 0
            for length in lengths:
                ngram = torch.full((1, 2), config["eos_token_id"], device="cuda", dtype=torch.int64)
                convolution = torch.zeros(1, 9, 64, device="cuda", dtype=torch.bfloat16)
                for row in range(start, start + length):
                    update, ngram, convolution = reference_ple(
                        residual[row : row + 1], tokens[row : row + 1], weights, config, ngram, convolution
                    )
                    updates.append(update)
                start += length
            residual = residual + torch.cat(updates)
        mixed, gates = reference_hyper(residual, weights, prefix + ".attn_hyper_connection", config)

        local = {
            name.replace(prefix, "model.language_model.layers.1", 1): value
            for name, value in weights.items()
            if name.startswith(prefix + ".")
        }
        updates, start = [], 0
        for length in lengths:
            channels = (
                2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
                + config["linear_num_value_heads"] * config["linear_value_head_dim"]
            )
            convolution = torch.zeros(
                1, channels, config["linear_conv_kernel_dim"] - 1, device="cuda", dtype=torch.bfloat16
            )
            recurrent = torch.zeros(
                1,
                config["linear_num_value_heads"],
                config["linear_key_head_dim"],
                config["linear_value_head_dim"],
                device="cuda",
                dtype=torch.float32,
            )
            for row in range(start, start + length):
                update, convolution, recurrent = reference_gdn(
                    mixed[row : row + 1], local, config, convolution, recurrent
                )
                updates.append(update)
            start += length
        residual = residual + (torch.cat(updates)[:, None] * gates[:, :, None]).flatten(1)
        mixed, gates = reference_hyper(residual, weights, prefix + ".mlp_hyper_connection", config)
        residual = residual + (reference_moe(mixed, weights, prefix, config)[:, None] * gates[:, :, None]).flatten(1)
    hidden = reference_hyper(residual, weights, "model.language_model.hyper_connection_mixer", config, combine=False)
    return F.linear(hidden, weights["lm_head.weight"].cuda()).float()
