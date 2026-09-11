from dataclasses import dataclass

ATT_TABLE: dict[tuple[str, str, str], list["AttBackendEntry"]] = {}


@dataclass(frozen=True)
class AttBackendEntry:
    # The name of the att backend. e.g., "triton", "flashinfer", "fa3".
    name: str
    # The category of the att backend. e.g., "standard", "mla", "nsa".
    category: str
    # The type of the kv. e.g., "int8kv", "int4kv", "fp8kv_sph".
    kv_type: str
    # The class of the att backend.
    backend_cls: type
    # The platforms of the att backend. e.g., "cuda", "ascend".
    platforms: tuple[str, ...]


def register_att_backend(
    name: str,
    *,
    category: str,
    kv_type: str = "None",
    platforms: tuple[str, ...] = ("cuda",),
):
    def decorator(backend_cls):
        if not platforms:
            raise ValueError("The value of platforms must be specified for att backend registration.")

        key = (name, category, kv_type)
        # Check if the att backend is already registered.
        for existing in ATT_TABLE.get(key, []):
            if set(existing.platforms) & set(platforms):
                raise ValueError(f"The att backend {name} is already registered for platforms {existing.platforms}.")

        ATT_TABLE.setdefault(key, []).append(
            AttBackendEntry(
                name=name,
                category=category,
                kv_type=kv_type,
                backend_cls=backend_cls,
                platforms=platforms,
            )
        )
        return backend_cls

    return decorator


def get_att_backend_class(
    name: str,
    category: str,
    kv_type: str,
    platform: str,
) -> type:
    key = (name, category, kv_type)
    for entry in ATT_TABLE.get(key, []):
        if platform in entry.platforms:
            return entry.backend_cls
    raise ValueError(
        f"The att backend {name} is not registered for category {category}, kv type {kv_type}, and platform {platform}."
    )
