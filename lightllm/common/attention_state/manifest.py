from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import torch
import yaml


def _canonical(value):
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def digest(value):
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_content_identity(model):

    weights = hashlib.sha256()
    if model.weight_dict is not None:
        for name, tensor in sorted(model.weight_dict.items()):
            weights.update(name.encode())
            weights.update(str((tensor.shape, tensor.dtype)).encode())
            weights.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    else:
        directory = Path(model.weight_dir_)
        files = sorted(directory.glob("*.safetensors")) or sorted(directory.glob("*.bin"))
        if not files:
            raise ValueError("cannot identify cached state without checkpoint weights")
        for path in files:
            before = path.stat()
            weights.update(path.name.encode())
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    weights.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("checkpoint changed while computing its state identity")
    if hasattr(model, "quant_cfg"):
        quantization = model.quant_cfg.content_identity()
    else:
        quant_config = yaml.safe_load(Path(model.quant_cfg_path).read_text()) if model.quant_cfg_path else None
        quantization = dict(method=model.quant_type, config=quant_config)
    return digest(
        dict(
            config=model.config,
            dtype=str(model.data_type),
            weights=weights.hexdigest(),
            weight_quantization=quantization,
        )
    )


def logical_state_groups(state):
    groups = getattr(state.config, "logical_layer_groups", None)
    return groups if groups is not None else tuple((layer,) for layer in state.config.logical_layer_ids)


def state_semantics(state):
    def global_config(value):
        if isinstance(value, dict):
            return {
                key: global_config(item)
                for key, item in value.items()
                if key not in ("logical_layer_groups", "logical_layer_ids", "tp_rank", "tp_world_size", "replicated")
            }
        return value

    config = global_config(asdict(state.config))
    describe = getattr(state, "get_attention_format", None)
    format = describe() if describe is not None else None
    representation = asdict(format) if format is not None else {"encoding": state.get_state_type()}
    for key in ("tp_rank", "tp_size", "local_heads", "replicated", "page_size", "layout"):
        representation.pop(key, None)
    return dict(config=config, representation=representation)


def _resource_fields(layout, row):
    fields = []
    for region in layout.regions:
        external_row = row
        mapping = next((m for m in region.axis_mappings if m.axis == "layer"), None)
        if mapping is not None:
            if not mapping.local_start <= row < mapping.local_start + mapping.length:
                continue
            external_row = row - mapping.local_start + mapping.external_start
        field = asdict(region)
        field.pop("axis_mappings")
        field.pop("write_enabled")
        field["layer_index"] = external_row
        fields.append(field)
    return fields


def build_state_manifest(manager, layouts=None, *, domain="pd"):
    if domain not in ("pd", "cpu"):
        raise ValueError(f"unknown state manifest domain {domain}")
    if layouts is None:
        if domain == "cpu":
            layouts = manager.get_cpu_cache_operator().get_page_layouts()
        else:
            operator = manager.get_pd_operator()
            layouts = {**operator.get_pd_full_att_page_layouts(), **operator.get_pd_bounded_att_page_layouts()}
    states = []
    for state in (*manager.get_full_attention_states(), *manager.get_bounded_attention_states()):
        layout = layouts.get(state.state_name)
        if layout is None and (domain != "cpu" or state.get_cpu_transfer_regions()):
            raise ValueError(f"missing {domain} transfer layout for {state.state_name}")
        for row, group in enumerate(logical_state_groups(state)):
            item = dict(
                role=state.model_role,
                kind=state.resource_kind,
                producer_layer=group[0],
                consumers=sorted(group),
                **state_semantics(state),
                fields=_resource_fields(layout, row) if layout is not None else [],
                tokens_per_page=layout.tokens_per_page if layout is not None else None,
                byte_start=layout.byte_start if layout is not None else None,
                byte_end=layout.byte_end if layout is not None else None,
            )
            if hasattr(state, "kv_calibration_digests"):
                item["kv_calibration"] = state.kv_calibration_digests[group[0]]
            elif hasattr(state, "scales"):
                item["kv_calibration"] = digest(state.scales[row].detach().cpu().tolist())
            states.append(item)
    states.sort(key=lambda item: (item["kind"] != "kv", item["role"], item["kind"], item["producer_layer"]))
    return _canonical(dict(version=2, models=getattr(manager, "model_identities", {}), states=states))


def state_manifest_fingerprint(manifest):

    if manifest is None:
        return None
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int:
        raise ValueError("invalid state manifest version")
    if set(manifest) == {"version", "models_digest", "states_digest"}:
        for key in ("models_digest", "states_digest"):
            value = manifest[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid state manifest digest")
        return dict(manifest)
    if set(manifest) != {"version", "models", "states"}:
        raise ValueError("incomplete state manifest")
    return dict(
        version=manifest["version"], models_digest=digest(manifest["models"]), states_digest=digest(manifest["states"])
    )


def validate_state_manifest(local, remote):
    if local is None and remote is None:
        return
    if local is None or remote is None:
        raise ValueError("PD peer did not provide a state manifest")
    local, remote = state_manifest_fingerprint(local), state_manifest_fingerprint(remote)
    if local["version"] != remote["version"]:
        raise ValueError("PD state manifest versions differ")
    if local["models_digest"] != remote["models_digest"]:
        raise ValueError("PD peers have different model or weight identities")
    if local["states_digest"] != remote["states_digest"]:
        raise ValueError("PD peers have incompatible state semantics, calibration or wire layouts")


def validate_disk_manifest(directory, manifest):

    import fcntl
    import os
    import tempfile

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state_manifest.json"
    with (directory / "state_manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if digest(json.loads(path.read_text())) != digest(manifest):
                raise ValueError("disk cache content or layout differs; use a directory with a matching state manifest")
            return
        if any(directory.glob("cache_file*")):
            raise ValueError("disk cache has no state manifest; use a fresh directory for identified cache content")
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(_canonical(manifest), stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
