"""Control endpoints. Every route here changes state, so all require the API key."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import get_service
from backend.config.settings import FAULTS, MISSION_PROFILES
from backend.security.telemetry_security import require_api_key
from backend.services.engine_service import EngineService

router = APIRouter(tags=['control'], dependencies=[Depends(require_api_key)])


class FaultRequest(BaseModel):
    fault: str = Field(description=f'One of: {", ".join(FAULTS)}')


class MissionRequest(BaseModel):
    mission: str = Field(description=f'One of: {", ".join(MISSION_PROFILES)}')


class StartRequest(BaseModel):
    name: str | None = Field(default=None, description='Optional mission label')


class IngestionRequest(BaseModel):
    mode: str = Field(description='"direct" for in-process, "can" for the J1939-style bus')


@router.post('/simulation/start')
async def start(body: StartRequest | None = None,
                actor: str = Depends(require_api_key),
                service: EngineService = Depends(get_service)) -> dict:
    return service.start(actor=actor, name=body.name if body else None)


@router.post('/simulation/stop')
def stop(actor: str = Depends(require_api_key),
         service: EngineService = Depends(get_service)) -> dict:
    return service.stop(actor=actor)


@router.post('/simulation/reset')
def reset(actor: str = Depends(require_api_key),
          service: EngineService = Depends(get_service)) -> dict:
    """Clear live state. Stored missions are kept; use DELETE /missions to purge."""
    return service.reset(actor=actor)


@router.post('/simulation/fault')
def inject_fault(body: FaultRequest, actor: str = Depends(require_api_key),
                 service: EngineService = Depends(get_service)) -> dict:
    try:
        fault = service.inject_fault(body.fault)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    service.db.audit(actor, 'fault.inject', {'fault': fault})
    return {'fault': fault, 'note': 'Demonstration fault injection into the simulated engine.'}


@router.post('/mission/configure')
def configure_mission(body: MissionRequest, actor: str = Depends(require_api_key),
                      service: EngineService = Depends(get_service)) -> dict:
    try:
        mission = service.configure_mission(body.mission)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    service.db.audit(actor, 'mission.configure', {'mission': mission})
    return {'mission': mission}


@router.post('/ingestion/mode')
def set_ingestion(body: IngestionRequest, actor: str = Depends(require_api_key),
                  service: EngineService = Depends(get_service)) -> dict:
    try:
        mode = service.set_ingestion(body.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    service.db.audit(actor, 'ingestion.mode', {'mode': mode})
    return {'mode': mode, 'transport': service.source.__class__.__name__}
