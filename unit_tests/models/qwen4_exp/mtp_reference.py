import torch.nn.functional as F

from .reference import norm, reference_layer
from .helpers import reference_hyper


def reference_fusion(embedding, hidden, weights, config):
    embedding = norm(embedding, weights["mtp.pre_fc_norm_embedding.weight"], config["rms_norm_eps"])
    hidden = norm(hidden, weights["mtp.pre_fc_norm_hidden.weight"], config["rms_norm_eps"])
    projected_embedding = F.linear(embedding, weights["mtp.fc_embedding.weight"].cuda())
    projected_hidden = F.linear(
        hidden.view(-1, config["hc_count"], config["hidden_size"]), weights["mtp.fc_hidden.weight"].cuda()
    )
    return (projected_hidden + projected_embedding[:, None]).flatten(1)


def reference_draft(tokens, hidden, weights, config, cos, sin, lengths):
    embeddings = F.embedding(tokens, weights["model.language_model.embed_tokens.weight"].cuda())
    residual = reference_fusion(embeddings, hidden, weights, config)
    mapped = {
        name.replace("mtp.layers.0.", "model.language_model.layers.3.", 1): value
        for name, value in weights.items()
        if name.startswith("mtp.layers.0.")
    }
    residual = reference_layer(residual, mapped, config, cos, sin, lengths)
    mixed = reference_hyper(residual, weights, "mtp.hyper_connection_mixer", config, combine=False)
    return residual, F.linear(mixed, weights["lm_head.weight"].cuda()).float()
