def bind_compute(spec, view, device, factories, *, requested="auto", probe_kwargs=None, bind_kwargs=None):
    if not view.finalized:
        raise ValueError("weight format must be finalized before compute binding")
    failures = []
    for create in factories:
        backend = create()
        if requested not in ("auto", backend.name, getattr(backend, "implementation", backend.name)):
            continue
        result = backend.probe(spec, view.format, device, **(probe_kwargs or {}))
        if result.offers:
            bound = backend.bind(result.offers[0], **view.fields, **(bind_kwargs or {}))
            bound.weight_view = view
            return bound
        failures.append(f"{backend.name}: {result.reason}")
    raise ValueError("; ".join(failures) or f"compute backend is not registered: {requested}")
