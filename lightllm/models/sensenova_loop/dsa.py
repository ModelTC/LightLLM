"""Qwen-DSA math and packed KV layout, also executable on CPU."""
import math
import torch
import torch.nn.functional as F


def validate_dsa_config(config):
    dsa = config.get("dsa")
    if not dsa:
        return None
    contract = config.get("dsa_export") or {}
    if contract.get("schema") != "sensenova_loop.dsa" or contract.get("version") != 1:
        raise ValueError("DSA requires a complete Swift sensenova_loop.dsa v1 export")
    for name in ("num_heads", "head_dim", "topk"):
        if type(dsa.get(name)) is not int or dsa[name] < 1:
            raise ValueError(f"Invalid DSA {name}")
    if dsa["topk"] > 2048:
        raise ValueError("This sparse inference backend supports DSA topk <= 2048")
    if math.frexp((dsa["num_heads"] * dsa["head_dim"]) ** -0.5)[0] != 0.5:
        raise ValueError("BF16 DSA requires a power-of-two Indexer scale (e.g. 8 x 128)")
    text = config.get("text_config", config)
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    rotary_dim = int(text["head_dim"] * rope.get("partial_rotary_factor", 0.25))
    if dsa["head_dim"] < rotary_dim:
        raise ValueError("DSA head_dim does not cover Qwen rotary_dim")
    if contract.get("key_norm_eps") != 1e-6 or contract.get("indexer_rope") != "qwen_partial_mrope":
        raise ValueError("Unsupported DSA normalization/RoPE contract")
    return dict(dsa, rotary_dim=rotary_dim)


def project_indexer(hidden, wq, wk, weights_proj, norm_weight, norm_bias, num_heads):
    dim = wk.shape[0]
    query = F.linear(hidden, wq).view(-1, num_heads, dim)
    key = F.layer_norm(F.linear(hidden, wk), (dim,), norm_weight, norm_bias, 1e-6)
    scale = (num_heads * dim) ** -0.5
    weights = F.linear(hidden, weights_proj).float() * scale
    if query.dtype == torch.bfloat16 and math.frexp(scale)[0] == 0.5:
        weights = weights.to(query.dtype)
    return query, key, weights


def yarn_cos_sin(text, length, device):
    """Match Titan's truncated YaRN ramp and FP32 MRoPE cache."""
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    dim = int(text["head_dim"] * rope.get("partial_rotary_factor", 0.25))
    base = rope.get("rope_theta", text.get("rope_theta", 10000000.0))
    factor = rope.get("factor", 1.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    if rope.get("rope_type", rope.get("type")) == "yarn" and factor > 1:
        original = rope["original_max_position_embeddings"]

        def cutoff(rotations):
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(cutoff(rope.get("beta_fast", 32.0))), 0)
        high = min(math.ceil(cutoff(rope.get("beta_slow", 1.0))), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, device=device).float() - low) / (high - low)).clamp(0, 1)
        inv_freq = inv_freq / factor * ramp + inv_freq * (1 - ramp)
    phase = torch.outer(torch.arange(length, device=device, dtype=inv_freq.dtype), inv_freq).float()
    phase = torch.cat((phase, phase), dim=-1)
    scale = rope.get("attention_factor", 1 + 0.1 * math.log(factor))
    return (phase.cos() * scale)[:, : dim // 2], (phase.sin() * scale)[:, : dim // 2]


def apply_partial_mrope(query, key, cos, sin, rotary_dim, sections=(11, 11, 10)):
    """Titan's FP32 rotate-half arithmetic followed by one BF16 rounding.

    Positions/cos/sin use Qwen's [time,height,width] axes. This path also
    avoids changing Indexer rotary dimension from 64 to 32 just because
    its head dimension (128) is smaller than the attention head (256).
    """
    half = rotary_dim // 2
    index = torch.arange(half, device=query.device)
    height = (index % 3 == 1) & (index < 3 * sections[1])
    width = (index % 3 == 2) & (index < 3 * sections[2])
    if cos.ndim == 3:
        cos = torch.where(height, cos[1], torch.where(width, cos[2], cos[0]))
        sin = torch.where(height, sin[1], torch.where(width, sin[2], sin[0]))
    cos, sin = cos[:, None, :half].float(), sin[:, None, :half].float()
    for tensor in (query, key):
        left = tensor[..., :half].float()
        right = tensor[..., half:rotary_dim].float()
        rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), -1)
        tensor[..., :rotary_dim].copy_(rotated.to(tensor.dtype))
    return query, key


def pack_kv(kv, indexer_key, attention_dim, index_dim):
    """Keep one replicated Indexer K in every K-head tail; V tails are unused.

    Standard KV, CPU cache, prefix-cache moves and NIXL pages all move this
    single tensor. Unequal prefill/decode TP therefore needs no extra channel.
    Draft (dense) layers pass indexer_key=None and retain their ordinary KV.
    """
    if kv.shape[-1] != attention_dim:
        raise ValueError("Unexpected attention KV head dimension")
    result = kv.new_zeros((*kv.shape[:-1], attention_dim + index_dim))
    result[..., :attention_dim] = kv
    if indexer_key is not None:
        if tuple(indexer_key.shape) != (kv.shape[0], index_dim):
            raise ValueError("Unexpected Indexer K shape")
        result[:, : kv.shape[1] // 2, attention_dim:] = indexer_key[:, None]
    return result


def query_metadata(cu_q, kv_lengths, req_ids, num_queries):
    # Do not use MRoPE position_ids for causality: image tokens have independent
    # text/height/width positions, whereas causal order is the token sequence.
    rows = torch.arange(num_queries, device=cu_q.device, dtype=cu_q.dtype)
    batch = torch.searchsorted(cu_q[1:].contiguous(), rows, right=True)
    batch = batch.clamp_max(kv_lengths.numel() - 1)
    q_lengths = cu_q[1:] - cu_q[:-1]
    positions = kv_lengths[batch] - q_lengths[batch] + rows - cu_q[batch]
    return req_ids[batch].contiguous(), positions.contiguous()


def reference_scores(query, indexer_keys, weights):
    scores = None
    for head in range(query.shape[1]):
        product = query[:, head].float() @ indexer_keys.float().T
        term = product.relu() * weights[:, head, None].float()
        scores = term if scores is None else scores + term
    return scores


def reference_paged_attention(q, iq, iw, kv, req_to_token, req_ids, positions, *, topk, attention_dim):
    """CPU oracle: same score, causal selection, grouped heads and scaling."""
    heads, dim = q.shape[1:]
    kv_heads = kv.shape[1] // 2
    output = torch.zeros_like(q)
    selections = []
    for row in range(q.shape[0]):
        count = int(positions[row]) + 1
        if count <= 0:
            selections.append(torch.empty(0, dtype=torch.long))
            continue
        mem = req_to_token[int(req_ids[row]), :count].long()
        keys = kv[mem, 0, attention_dim:]
        scores = reference_scores(iq[row : row + 1], keys, iw[row : row + 1])[0]
        indices = torch.argsort(scores, descending=True, stable=True)[: min(topk, count)].sort().values
        selections.append(indices)
        selected = kv[mem[indices]]
        for head in range(heads):
            group = head // (heads // kv_heads)
            logits = (q[row, head].float() * selected[:, group, :dim].float()).sum(-1) * dim ** -0.5
            output[row, head] = logits.softmax(-1) @ selected[:, kv_heads + group, :dim].float()
    return output, selections
