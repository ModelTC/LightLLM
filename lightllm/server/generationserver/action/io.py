"""Model-independent Action HTTP input adaptation and model I/O lookup."""

from __future__ import annotations

from typing import Any

from lightllm.utils.config_utils import get_config_json


def _action_model_class(model_config: dict):
    # HTTP requests use only the mapping functions above. Keep model imports on
    # the startup path so protocol adaptation stays cheap in HTTP workers.
    from lightllm.models.registry import get_generation_model_class

    return get_generation_model_class(model_config, "action")


def adapt_action_domain_body(
    body: dict[str, Any], config: dict | None
) -> dict[str, Any]:
    """Resolve a wire domain and its public action width before dispatch."""

    raw_domain = body.get("domain")
    domain_name = body.get("domain_name")
    if raw_domain is not None and domain_name is not None and raw_domain != domain_name:
        raise ValueError("domain and domain_name must match when both are provided")
    raw_domain = raw_domain if raw_domain is not None else domain_name
    if raw_domain is None:
        return body
    if not isinstance(raw_domain, str) or not raw_domain.strip():
        raise ValueError("domain must be a non-empty string")
    domain = raw_domain.strip().lower()
    aliases = {} if config is None else config.get("domain_aliases", {})
    domain = aliases.get(domain, domain)
    profiles = {} if config is None else config.get("domain_profiles", {})
    profile = profiles.get(domain)
    if profile is None:
        raise ValueError(f"unsupported action domain={raw_domain!r}")

    adapted = dict(body)
    adapted["domain"] = domain
    expected_action_dim = int(profile["action_dim"])
    supplied_action_dim = adapted.get("action_dim")
    if supplied_action_dim is None:
        adapted["action_dim"] = expected_action_dim
    elif supplied_action_dim != expected_action_dim:
        raise ValueError(
            f"domain={domain!r} requires action_dim={expected_action_dim}, "
            f"got {supplied_action_dim}"
        )
    return adapted


def adapt_action_domain_images(
    domain: str | None,
    image_inputs: list[tuple[str | None, Any]],
    config: dict | None,
) -> list[tuple[str | None, Any]]:
    """Map environment camera names to checkpoint canonical image slots."""

    if domain is None:
        return image_inputs
    if config is None:
        raise ValueError("action domain profiles are unavailable")
    profile = config["domain_profiles"][domain]
    image_keys = tuple(config.get("image_keys", ()))
    image_map = profile.get("image_map", {})
    ignored = set(profile.get("ignored_images", []))
    mapped = {}
    for source, image in image_inputs:
        if source is None:
            raise ValueError("domain action requests require named images")
        if source in ignored:
            continue
        if source in image_keys:
            target = source
        else:
            target = image_map.get(source)
            if target is None:
                raise ValueError(f"unknown camera {source!r} for domain={domain!r}")
        if image_keys and target not in image_keys:
            raise ValueError(
                f"domain={domain!r} maps camera {source!r} to unknown image slot "
                f"{target!r}"
            )
        if target in mapped:
            raise ValueError(f"multiple domain cameras map to image slot {target!r}")
        mapped[target] = image

    required = set(profile.get("required_images", []))
    missing = required.difference(mapped)
    if missing:
        raise ValueError(
            f"domain={domain!r} is missing required image slots: "
            + ", ".join(sorted(missing))
        )
    order = image_keys or tuple(mapped)
    return [(key, mapped[key]) for key in order if key in mapped]


def load_action_config(args) -> dict:
    model_class = _action_model_class(get_config_json(args.model_dir))
    return model_class.generation_config_loader(args)


def create_action_processor(config: dict):
    model_class = _action_model_class({"type": config["model_type"]})
    return model_class.generation_processor_class(config)
