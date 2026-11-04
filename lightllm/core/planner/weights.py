def bind_compute(spec, view, device, factories, *, requested="auto", probe_kwargs=None, bind_kwargs=None):
    failures = []
    for create in factories:
        backend = create()
        if requested not in ("auto", backend.name):
            continue
        result = backend.probe(spec, view.format, device, **(probe_kwargs or {}))
        if result.offers:
            return backend.bind(result.offers[0], **view.fields, **(bind_kwargs or {}))
        failures.append(f"{backend.name}: {result.reason}")
    raise ValueError("; ".join(failures) or f"compute backend is not registered: {requested}")
