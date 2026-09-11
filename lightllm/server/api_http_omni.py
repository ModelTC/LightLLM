"""Public HTTP routes that extend LightLLM beyond text generation.

The handlers intentionally contain no model preprocessing.  They select the
configured modality, map transport errors to HTTP status codes, and delegate
request translation to the lazily imported Omni adapter.
"""

from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import Response

from lightllm.utils.error_utils import ClientDisconnected
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
router = APIRouter()


def _omni_manager(expected_modality: str):
    from .api_http import g_objs

    manager = g_objs.httpserver_manager
    if (
        manager is None
        or getattr(manager, "output_modality", None) != expected_modality
        or getattr(manager, "output_client", None) is None
    ):
        return None
    return manager


def _error(status: HTTPStatus, message: str) -> Response:
    from .api_http import create_error_response

    return create_error_response(status, message)


@router.post("/v1/vla/actions")
async def vla_actions(request: Request) -> Response:
    manager = _omni_manager("action")
    if manager is None:
        return _error(
            HTTPStatus.NOT_FOUND, "this server was not started with a VLA model"
        )
    try:
        from .httpserver.omni_adapter import generate_actions

        return await generate_actions(request, manager)
    except (TypeError, ValueError) as exc:
        return _error(HTTPStatus.BAD_REQUEST, str(exc))
    except ClientDisconnected as exc:
        logger.warning(str(exc))
        return Response(status_code=499)
    except TimeoutError as exc:
        return _error(HTTPStatus.GATEWAY_TIMEOUT, str(exc))
    except Exception as exc:
        logger.error("VLA action request failed: %s", str(exc), exc_info=True)
        return _error(HTTPStatus.EXPECTATION_FAILED, str(exc))


@router.delete("/v1/vla/contexts/{context_id}")
async def vla_close_context(context_id: str, request: Request) -> Response:
    manager = _omni_manager("action")
    if manager is None:
        return _error(
            HTTPStatus.NOT_FOUND, "this server was not started with a VLA model"
        )
    try:
        from .httpserver.omni_adapter import close_prefix_context

        return await close_prefix_context(request, manager, context_id)
    except (TypeError, ValueError) as exc:
        return _error(HTTPStatus.BAD_REQUEST, str(exc))
    except ClientDisconnected as exc:
        logger.warning(str(exc))
        return Response(status_code=499)
    except TimeoutError as exc:
        return _error(HTTPStatus.GATEWAY_TIMEOUT, str(exc))
    except Exception as exc:
        logger.error("VLA context close failed: %s", str(exc), exc_info=True)
        return _error(HTTPStatus.EXPECTATION_FAILED, str(exc))


@router.post("/v1/images/generations")
async def image_generations(request: Request) -> Response:
    manager = _omni_manager("image")
    if manager is None:
        return _error(
            HTTPStatus.NOT_FOUND,
            "this server was not started with a generation backend",
        )
    try:
        from .httpserver.omni_adapter import generate_images

        return await generate_images(request, manager)
    except (TypeError, ValueError) as exc:
        return _error(HTTPStatus.BAD_REQUEST, str(exc))
    except ClientDisconnected as exc:
        logger.warning(str(exc))
        return Response(status_code=499)
    except TimeoutError as exc:
        return _error(HTTPStatus.GATEWAY_TIMEOUT, str(exc))
    except Exception as exc:
        logger.error("image generation request failed: %s", str(exc), exc_info=True)
        return _error(HTTPStatus.EXPECTATION_FAILED, str(exc))
