"""Authentication helpers for the RL control-plane HTTP routes."""

import os
import secrets
from typing import Protocol

from fastapi import APIRouter, FastAPI, HTTPException, Request

RL_API_KEY_ENV = "LIGHTLLM_RL_API_KEY"


class RLStartArgs(Protocol):
    enable_rl: bool


def get_rl_api_key() -> str | None:
    """Return the configured RL control-plane key without logging it."""
    key = os.getenv(RL_API_KEY_ENV)
    return key if key else None


def validate_rl_auth(args: RLStartArgs) -> None:
    """Reject an RL-enabled server that has no control-plane credential."""
    if args.enable_rl and get_rl_api_key() is None:
        raise ValueError(f"{RL_API_KEY_ENV} must be set when --enable_rl is enabled")


def mount_rl_router(app: FastAPI, router: APIRouter, args: RLStartArgs, mounted: bool) -> bool:
    """Mount the RL router once and only for an RL-enabled server."""
    if args.enable_rl and not mounted:
        app.include_router(router)
        return True
    return mounted


def require_rl_auth(request: Request) -> None:
    """Require a valid bearer token for an RL control-plane request."""
    configured_key = get_rl_api_key()
    if configured_key is None:
        raise HTTPException(status_code=503, detail="RL control-plane authentication is not configured")

    authorization_parts = request.headers.get("authorization", "").split()
    if (
        len(authorization_parts) != 2
        or authorization_parts[0].lower() != "bearer"
        or not secrets.compare_digest(authorization_parts[1].encode("utf-8"), configured_key.encode("utf-8"))
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid RL control-plane credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
