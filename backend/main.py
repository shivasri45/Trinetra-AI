"""Trinetra AI application entry point.

Route groups live in ``backend/api``; this module only wires them together,
configures CORS, manages the service lifecycle and hosts the telemetry
WebSocket.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

from backend.api import control, missions, system, telemetry
from backend.api.dependencies import init_service
from backend.config.settings import settings
from backend.security.telemetry_security import check_api_key, startup_warnings
from backend.services.engine_service import EngineService

logger = logging.getLogger('trinetra')

DESCRIPTION = """
Digital twin for a MALE-UAV aero piston engine.

* Thermodynamic engine model solved from first principles, synchronised to
  telemetry by a per-channel Kalman filter.
* Fault detection fusing a statistical residual test, an unsupervised novelty
  detector and a supervised classifier, with a transparent physics prior.
* Remaining useful life from degradation-trend extrapolation with a confidence
  interval, referenced to published TBO.
* J1939-style CAN ingestion, per-frame HMAC integrity, API-key control and an
  audit trail.

All telemetry is synthetic. Nothing here is calibrated or certified for flight,
maintenance or airworthiness decisions.
"""

service = EngineService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_service(service)
    for warning in startup_warnings():
        logger.warning('SECURITY: %s', warning)
    service.db.audit('system', 'service.start', {'ingestion': service.ingestion_mode})
    yield
    service.stop(actor='system')
    service.close()


app = FastAPI(
    title='Trinetra AI Digital Twin',
    version='2.0.0',
    description=DESCRIPTION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'DELETE'],
    allow_headers=['*'],
)

app.include_router(system.router)
app.include_router(telemetry.router)
app.include_router(control.router)
app.include_router(missions.router)


@app.websocket('/ws/telemetry')
async def stream(websocket: WebSocket, api_key: str | None = Query(default=None)) -> None:
    """Live snapshot stream.

    Authentication uses the same key as the REST API, supplied either as the
    ``api_key`` query parameter or an ``X-API-Key`` header, because browsers
    cannot set headers on a WebSocket handshake.
    """
    supplied = api_key or websocket.headers.get('x-api-key')
    if not check_api_key(supplied):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason='invalid api key')
        return

    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    service.subscribers.add(queue)
    try:
        await websocket.send_json(service.snapshot())
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - transport level failures
        logger.info('telemetry stream closed: %s', exc)
    finally:
        service.subscribers.discard(queue)
