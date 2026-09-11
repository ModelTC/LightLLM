import pytest

from lightllm.server.generationserver.action.protocol import (
    ActionRequest,
    PrefixContextIdentity,
    PrefixContextOp,
    PrefixContextRef,
)


class _Config(dict):
    def __init__(self, *, is_pi05: bool):
        super().__init__(
            model_type="pi05" if is_pi05 else "pi0",
            max_state_dim=4,
            max_action_dim=4,
            action_horizon=2,
            action_dim=3,
            num_denoise_steps=2,
            state_in_prefix=is_pi05,
            action_expert_uses_state=not is_pi05,
        )


def _identity() -> PrefixContextIdentity:
    return PrefixContextIdentity(
        context_id="context-7",
        version=3,
        server_epoch="server-epoch-2",
    )


def _context(op: PrefixContextOp) -> PrefixContextRef:
    if op is PrefixContextOp.CREATE:
        return PrefixContextRef(op=op, context_id=_identity().context_id)
    identity = _identity() if op.requires_identity else None
    return PrefixContextRef(op=op, identity=identity)


def test_prefix_context_identity_and_ref_round_trip():
    identity = _identity()
    ref = PrefixContextRef(op=PrefixContextOp.REUSE, identity=identity)
    create_ref = PrefixContextRef(
        op=PrefixContextOp.CREATE,
        context_id=identity.context_id,
    )

    assert PrefixContextIdentity.from_dict(identity.to_dict()) == identity
    assert PrefixContextRef.from_dict(ref.to_dict()) == ref
    assert create_ref.to_dict() == {
        "op": "create",
        "context_id": "context-7",
    }
    assert PrefixContextRef.from_dict(create_ref.to_dict()) == create_ref
    assert PrefixContextRef.from_dict(None) == PrefixContextRef()
    assert PrefixContextRef.from_dict("oneshot") == PrefixContextRef()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_id": "", "version": 1, "server_epoch": "epoch"}, "context_id"),
        ({"context_id": "ctx", "version": 0, "server_epoch": "epoch"}, "positive"),
        ({"context_id": "ctx", "version": True, "server_epoch": "epoch"}, "integer"),
        ({"context_id": "ctx", "version": 1, "server_epoch": ""}, "server_epoch"),
    ],
)
def test_prefix_context_identity_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PrefixContextIdentity(**kwargs)


def test_prefix_context_ref_requires_exact_identity_for_existing_context_ops():
    for op in (
        PrefixContextOp.REUSE,
        PrefixContextOp.REPLACE,
        PrefixContextOp.CLOSE,
    ):
        with pytest.raises(ValueError, match="requires an identity"):
            PrefixContextRef(op=op)

    with pytest.raises(ValueError, match="requires a context_id"):
        PrefixContextRef(op=PrefixContextOp.CREATE)
    with pytest.raises(ValueError, match="must not carry an identity"):
        PrefixContextRef(
            op=PrefixContextOp.CREATE,
            identity=_identity(),
            context_id=_identity().context_id,
        )

    for op in (PrefixContextOp.ONESHOT,):
        with pytest.raises(ValueError, match="must not carry an identity"):
            PrefixContextRef(op=op, identity=_identity())


def test_oneshot_action_request_omits_the_context_envelope():
    wire = {
        "state": [0.0, 1.0],
        "request_id": "oneshot-request",
        "raw_state": [0.0, 1.0],
        "noise": None,
        "action_horizon": None,
        "action_dim": None,
        "num_denoise_steps": None,
        "timeout": 120.0,
    }

    request = ActionRequest.from_dict(wire)

    assert request.context_op is PrefixContextOp.ONESHOT
    assert request.context_identity is None
    assert request.requires_prefix_inputs
    assert request.produces_action
    assert not request.persists_prefix
    assert request.to_dict() == wire
    request.validate(_Config(is_pi05=False))


@pytest.mark.parametrize(
    (
        "op",
        "requires_prefix_inputs",
        "produces_action",
        "persists_prefix",
    ),
    [
        (PrefixContextOp.ONESHOT, True, True, False),
        (PrefixContextOp.CREATE, True, True, True),
        (PrefixContextOp.REUSE, False, True, False),
        (PrefixContextOp.REPLACE, True, True, True),
        (PrefixContextOp.CLOSE, False, False, False),
    ],
)
def test_action_request_exposes_context_routing_properties(
    op,
    requires_prefix_inputs,
    produces_action,
    persists_prefix,
):
    request = ActionRequest(state=None, prefix_context=_context(op))

    assert request.context_op is op
    assert request.requires_prefix_inputs is requires_prefix_inputs
    assert request.produces_action is produces_action
    assert request.persists_prefix is persists_prefix
    assert (request.context_identity is not None) is op.requires_identity
    assert request.context_id == (
        None if op is PrefixContextOp.ONESHOT else _identity().context_id
    )
    assert (
        ActionRequest.from_dict(request.to_dict()).prefix_context
        == request.prefix_context
    )


def test_pi0_reuse_requires_continuous_task_state():
    config = _Config(is_pi05=False)
    ref = _context(PrefixContextOp.REUSE)

    request = ActionRequest(state=[0.25, -0.5], prefix_context=ref)
    assert request.accepts_model_state(config)
    request.validate(config)
    with pytest.raises(ValueError, match="op=reuse requires state"):
        ActionRequest(state=None, prefix_context=ref).validate(config)


def test_action_seed_is_an_explicit_integer_field():
    request = ActionRequest(state=[0.0], seed=7)
    assert request.to_dict()["seed"] == 7
    request.validate(_Config(is_pi05=False))
    with pytest.raises(ValueError, match="seed must be an integer"):
        ActionRequest(state=[0.0], seed=True).validate(_Config(is_pi05=False))


def test_noise_uses_the_checkpoint_action_width():
    config = _Config(is_pi05=False)
    full_width_noise = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]

    ActionRequest(state=[0.0], noise=full_width_noise).validate(config)
    with pytest.raises(ValueError, match=r"noise must have shape \(1, 2, 4\)"):
        ActionRequest(
            state=[0.0],
            noise=[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ).validate(config)


def test_domain_contract_sets_action_width_and_validates_state_width():
    config = _Config(is_pi05=False)
    config["domain_profiles"] = {
        "libero": {"state_dim": 2, "action_dim": 1},
    }

    request_config = ActionRequest(state=[0.0, 1.0], domain="libero").validate(config)
    assert request_config["action_dim"] == 1

    with pytest.raises(ValueError, match="requires state_dim=2"):
        ActionRequest(state=[0.0], domain="libero").validate(config)
    with pytest.raises(ValueError, match="requires action_dim=1"):
        ActionRequest(state=[0.0, 1.0], domain="libero", action_dim=2).validate(config)
    with pytest.raises(ValueError, match="unsupported action domain"):
        ActionRequest(state=[0.0, 1.0], domain="unknown").validate(config)


def test_embedded_state_reuse_cannot_silently_update_prefix_state():
    config = _Config(is_pi05=True)
    ref = _context(PrefixContextOp.REUSE)

    request = ActionRequest(
        state=None,
        raw_state=[0.25, -0.5],
        noise=[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        prefix_context=ref,
    )
    assert not request.accepts_model_state(config)
    request.validate(config)

    with pytest.raises(ValueError, match="prefix reuse cannot update embedded state"):
        ActionRequest(state=[0.25, -0.5], prefix_context=ref).validate(config)


@pytest.mark.parametrize("op", [PrefixContextOp.CREATE, PrefixContextOp.REPLACE])
def test_pi05_prefix_building_operations_require_state(op):
    config = _Config(is_pi05=True)
    ref = _context(op)

    ActionRequest(state=[0.25, -0.5], prefix_context=ref).validate(config)
    with pytest.raises(ValueError, match=f"op={op.value} requires state"):
        ActionRequest(state=None, prefix_context=ref).validate(config)


def test_close_is_control_only():
    config = _Config(is_pi05=False)
    ref = _context(PrefixContextOp.CLOSE)
    request = ActionRequest(state=None, prefix_context=ref)

    assert request.validate(config) is config
    with pytest.raises(ValueError, match="must not carry action fields: state"):
        ActionRequest(state=[0.0], prefix_context=ref).validate(config)
    with pytest.raises(ValueError, match="must not carry action fields: noise"):
        ActionRequest(state=None, noise=[[0.0]], prefix_context=ref).validate(config)
