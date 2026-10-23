import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from lightllm.server.api_http_rl import router as rl_router
from lightllm.server.rl_auth import (
    RL_API_KEY_ENV,
    mount_rl_router,
    require_rl_auth,
    validate_rl_auth,
)


def _request(headers=None):
    raw_headers = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    return Request({"type": "http", "headers": raw_headers})


def _route_paths(routes):
    for route in routes:
        path = getattr(route, "path", None)
        if path is not None:
            yield path
        yield from _route_paths(getattr(route, "routes", []))
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            yield from _route_paths(original_router.routes)


def test_rl_routes_are_mounted_only_when_enabled(monkeypatch):
    app = FastAPI()

    mounted = mount_rl_router(app, rl_router, SimpleNamespace(enable_rl=False), False)
    assert "/pause_generation" not in set(_route_paths(app.routes))

    monkeypatch.setenv(RL_API_KEY_ENV, "control-secret")
    mounted = mount_rl_router(app, rl_router, SimpleNamespace(enable_rl=True), mounted)
    assert "/pause_generation" in set(_route_paths(app.routes))
    assert mount_rl_router(app, rl_router, SimpleNamespace(enable_rl=True), mounted) is True


def test_mounted_rl_routes_reject_unauthenticated_requests(monkeypatch):
    monkeypatch.setenv(RL_API_KEY_ENV, "control-secret")
    app = FastAPI()
    app.include_router(rl_router)

    response = TestClient(app).post("/pause_generation")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_every_rl_route_requires_auth():
    assert rl_router.routes
    assert all(
        any(dependency.call is require_rl_auth for dependency in route.dependant.dependencies)
        for route in rl_router.routes
    )


def test_mounted_rl_route_dispatches_with_valid_bearer_token(monkeypatch):
    monkeypatch.setenv(RL_API_KEY_ENV, "control-secret")
    calls = []

    async def pause_generation():
        calls.append("pause")

    api_http_module = ModuleType("lightllm.server.api_http")
    api_http_module.g_objs = SimpleNamespace(httpserver_manager=SimpleNamespace(pause_generation=pause_generation))
    monkeypatch.setitem(sys.modules, "lightllm.server.api_http", api_http_module)

    app = FastAPI()
    app.include_router(rl_router)
    response = TestClient(app).post(
        "/pause_generation",
        headers={"Authorization": "Bearer control-secret"},
    )

    assert response.status_code == 200
    assert calls == ["pause"]


def test_rl_auth_accepts_only_the_configured_bearer_token(monkeypatch):
    monkeypatch.setenv(RL_API_KEY_ENV, "control-secret")

    assert require_rl_auth(_request({"authorization": "Bearer control-secret"})) is None
    assert require_rl_auth(_request({"authorization": "bEaReR control-secret"})) is None

    with pytest.raises(HTTPException) as error:
        require_rl_auth(_request({"authorization": "Bearer wrong-secret"}))
    assert error.value.status_code == 401

    with pytest.raises(HTTPException) as error:
        require_rl_auth(_request({"authorization": "Bearer incorrect-" + chr(233)}))
    assert error.value.status_code == 401


def test_rl_auth_rejects_missing_configuration(monkeypatch):
    monkeypatch.delenv(RL_API_KEY_ENV, raising=False)

    with pytest.raises(HTTPException) as error:
        require_rl_auth(_request({"authorization": "Bearer control-secret"}))
    assert error.value.status_code == 503


def test_rl_startup_requires_a_control_plane_key(monkeypatch):
    monkeypatch.delenv(RL_API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=RL_API_KEY_ENV):
        validate_rl_auth(SimpleNamespace(enable_rl=True))

    monkeypatch.setenv(RL_API_KEY_ENV, "control-secret")
    validate_rl_auth(SimpleNamespace(enable_rl=True))
