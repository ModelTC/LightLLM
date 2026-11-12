import torch


def rand(*shape):
    return torch.randn(*shape, dtype=torch.bfloat16) * 0.1


def add_hyper(weights, prefix, config, combine=True):
    hidden, branches, rank = config["hidden_size"], config["hc_count"], config["hc_lowrank"]
    width = hidden * branches
    weights[prefix + ".hc_norm.weight"] = rand(width)
    weights[prefix + ".input_mix_weight_down.weight"] = rand(rank, width)
    weights[prefix + ".input_mix_weight_up.weight"] = rand(width, rank)
    if combine:
        weights[prefix + ".block_inject_weight.weight"] = rand(branches, width)


def checkpoint(config, layer):
    hidden, inter, experts = config["hidden_size"], config["moe_intermediate_size"], config["num_experts"]
    prefix = f"model.language_model.layers.{layer}"
    weights = {}
    for block in ["attn_hyper_connection", "mlp_hyper_connection"]:
        add_hyper(weights, prefix + "." + block, config)
    weights[prefix + ".mlp.gate.weight"] = rand(experts, hidden)
    weights[prefix + ".mlp.experts.gate_up_proj"] = rand(experts, 2 * inter, hidden)
    weights[prefix + ".mlp.experts.down_proj"] = rand(experts, hidden, inter)
    weights[prefix + ".mlp.shared_expert_gate.weight"] = rand(1, hidden)
    for proj, shape in [("gate", (inter, hidden)), ("up", (inter, hidden)), ("down", (hidden, inter))]:
        weights[prefix + f".mlp.shared_expert.{proj}_proj.weight"] = rand(*shape)
    if config["layer_types"][layer] == "linear_attention":
        keydim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
        vdim = config["linear_num_value_heads"] * config["linear_value_head_dim"]
        channels = 2 * keydim + vdim
        weights[prefix + ".linear_attn.conv1d.weight"] = rand(channels, 1, 4)
        for proj, rows in [
            ("qkv", channels),
            ("z", vdim),
            ("b", config["linear_num_value_heads"]),
            ("a", config["linear_num_value_heads"]),
        ]:
            weights[prefix + f".linear_attn.in_proj_{proj}.weight"] = rand(rows, hidden)
        weights[prefix + ".linear_attn.out_proj.weight"] = rand(hidden, vdim)
        weights[prefix + ".linear_attn.A_log"] = torch.zeros(config["linear_num_value_heads"], dtype=torch.float32)
        weights[prefix + ".linear_attn.dt_bias"] = torch.ones(config["linear_num_value_heads"], dtype=torch.float32)
        weights[prefix + ".linear_attn.norm.weight"] = 1 + rand(config["linear_value_head_dim"])
    else:
        dim, qheads, kvheads = config["head_dim"], config["num_attention_heads"], config["num_key_value_heads"]
        for name, shape in [
            ("q_proj", (qheads * dim * 2, hidden)),
            ("k_proj", (kvheads * dim, hidden)),
            ("v_proj", (kvheads * dim, hidden)),
            ("o_proj", (hidden, qheads * dim)),
        ]:
            weights[prefix + f".self_attn.{name}.weight"] = rand(*shape)
        for name in ["q_norm", "k_norm"]:
            weights[prefix + f".self_attn.{name}.weight"] = rand(dim)
        weights[prefix + ".self_attn.indexer.index_qk_proj.weight"] = rand(
            (config["indexer_n_heads"] + 1) * config["indexer_head_dim"], hidden
        )
        for name in ["q_layernorm", "k_layernorm"]:
            weights[prefix + f".self_attn.indexer.{name}.weight"] = rand(config["indexer_head_dim"])
    if layer + 1 in config["ple_layer_ids"]:
        width, embed = config["hc_count"] * hidden, config["ple_embed_dim"]
        weights[prefix + ".ple.key_proj.weight"] = rand(width, embed)
        weights[prefix + ".ple.value_proj.weight"] = rand(hidden, embed)
        weights[prefix + ".ple.conv1d.weight"] = rand(width, 1, 4)
        for name in ["norm_key", "norm_query", "norm_conv"]:
            weights[prefix + f".ple.{name}.weight"] = rand(width)
        weights[prefix + ".ple.ple_embedding.layer_multipliers"] = torch.tensor(
            [23703573157769, 20109073645365, 8052911324071], dtype=torch.int64
        )
        weights[prefix + ".ple.ple_embedding.ngram_heads_vocab_sizes"] = torch.tensor([5, 7, 11, 13], dtype=torch.int64)
        weights[prefix + ".ple.ple_embedding.ngram_heads_offsets"] = torch.tensor([0, 5, 12, 23], dtype=torch.int64)
        table = rand(36, 8)
        for shard, values in enumerate(table.chunk(4)):
            weights[prefix + f".ple.ple_embedding.ngram_embedding.shard_{shard}.weight"] = values
    return weights


def prepost_checkpoint(config):
    weights = {
        "model.language_model.embed_tokens.weight": rand(config["vocab_size"], config["hidden_size"]),
        "lm_head.weight": rand(config["vocab_size"], config["hidden_size"]),
    }
    add_hyper(weights, "model.language_model.hyper_connection_mixer", config, False)
    return weights


def reference_hyper(x, weights, prefix, config, combine=True):
    grouped = x.float().unflatten(-1, (config["hc_count"], config["hidden_size"]))
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + config["rms_norm_eps"])
    normalized = (normalized.flatten(-2) * (1 + weights[prefix + ".hc_norm.weight"].cuda().float())).to(x.dtype)
    low = torch.nn.functional.silu(
        torch.nn.functional.linear(normalized, weights[prefix + ".input_mix_weight_down.weight"].cuda())
        / config["hc_count"]
    )
    gates = torch.nn.functional.linear(low, weights[prefix + ".input_mix_weight_up.weight"].cuda()).sigmoid()
    mixed = (normalized * gates).unflatten(-1, (config["hc_count"], config["hidden_size"])).mean(-2)
    if not combine:
        return mixed
    injection = 2 * torch.sigmoid(
        torch.nn.functional.linear(normalized, weights[prefix + ".block_inject_weight.weight"].cuda())
        / config["hc_count"]
    )
    return mixed, injection


def reference_moe(x, weights, prefix, config):
    logits = torch.nn.functional.linear(x, weights[prefix + ".mlp.gate.weight"].cuda())
    prob = logits.float().softmax(-1)
    values, indices = prob.topk(config["num_experts_per_tok"], dim=-1)
    if config["norm_topk_prob"]:
        values /= values.sum(-1, keepdim=True)
    values = values.to(x.dtype)
    out = torch.zeros_like(x)
    inter = config["moe_intermediate_size"]

    for expert in indices.unique(sorted=True).tolist():
        top_position, token = torch.where((indices == expert).T)
        gu = torch.nn.functional.linear(x[token], weights[prefix + ".mlp.experts.gate_up_proj"][expert].cuda())
        output = torch.nn.functional.linear(
            torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:],
            weights[prefix + ".mlp.experts.down_proj"][expert].cuda(),
        )
        weighted = output * values[token, top_position, None]
        out.index_add_(0, token, weighted.to(out.dtype))
    gate = torch.nn.functional.linear(x, weights[prefix + ".mlp.shared_expert_gate.weight"].cuda()).sigmoid()
    up = torch.nn.functional.linear(x, weights[prefix + ".mlp.shared_expert.up_proj.weight"].cuda())
    g = torch.nn.functional.linear(x, weights[prefix + ".mlp.shared_expert.gate_proj.weight"].cuda())
    shared = torch.nn.functional.linear(
        torch.nn.functional.silu(g) * up, weights[prefix + ".mlp.shared_expert.down_proj.weight"].cuda()
    )
    return out + gate * shared
