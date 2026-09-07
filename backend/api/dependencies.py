"""Shared application state and FastAPI dependency providers."""
from __future__ import annotations

from backend.services.engine_service import EngineService

_service: EngineService | None = None


def init_service(service: EngineService) -> EngineService:
    global _service
    _service = service
    return _service


def get_service() -> EngineService:
    if _service is None:  # pragma: no cover - wired at application startup
        raise RuntimeError('EngineService has not been initialised')
    return _service
