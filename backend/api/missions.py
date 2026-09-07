"""Mission history, replay and post-flight reporting."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.dependencies import get_service
from backend.security.telemetry_security import require_api_key
from backend.services.engine_service import EngineService

router = APIRouter(prefix='/missions', tags=['missions'])


@router.get('')
def list_missions(limit: int = Query(50, ge=1, le=200),
                  service: EngineService = Depends(get_service)) -> dict:
    missions = service.db.list_missions(limit)
    return {'active_mission_id': service.mission_id, 'count': len(missions), 'missions': missions}


@router.get('/{mission_id}')
def get_mission(mission_id: int, service: EngineService = Depends(get_service)) -> dict:
    mission = service.db.mission(mission_id)
    if mission is None:
        raise HTTPException(404, f'mission {mission_id} not found')
    mission['stored_samples'] = service.db.snapshot_count(mission_id)
    return mission


@router.get('/{mission_id}/timeline')
def timeline(mission_id: int, step: int = Query(1, ge=1, le=50),
             service: EngineService = Depends(get_service)) -> dict:
    """Lightweight scrubber index: sequence, time, hours, health and fault only."""
    if service.db.mission(mission_id) is None:
        raise HTTPException(404, f'mission {mission_id} not found')
    return {'mission_id': mission_id, 'step': step,
            'points': service.db.snapshot_index(mission_id, step)}


@router.get('/{mission_id}/snapshots')
def snapshots(mission_id: int, start: int = Query(0, ge=0),
              limit: int = Query(200, ge=1, le=900),
              service: EngineService = Depends(get_service)) -> dict:
    """Paged full snapshots. The dashboard drives replay from this endpoint."""
    if service.db.mission(mission_id) is None:
        raise HTTPException(404, f'mission {mission_id} not found')
    rows = service.db.snapshots(mission_id, start, limit)
    total = service.db.snapshot_count(mission_id)
    return {'mission_id': mission_id, 'start': start, 'returned': len(rows),
            'total': total, 'snapshots': rows}


@router.get('/{mission_id}/report')
def report(mission_id: int, service: EngineService = Depends(get_service)) -> dict:
    """Mission-wise health report for the maintenance record."""
    result = service.mission_report(mission_id)
    if result is None:
        raise HTTPException(404, f'mission {mission_id} not found')
    return result


@router.delete('/{mission_id}')
def delete_mission(mission_id: int, actor: str = Depends(require_api_key),
                   service: EngineService = Depends(get_service)) -> dict:
    if mission_id == service.mission_id:
        raise HTTPException(409, 'cannot delete the mission currently recording')
    if not service.db.delete_mission(mission_id):
        raise HTTPException(404, f'mission {mission_id} not found')
    service.db.audit(actor, 'mission.delete', {'mission_id': mission_id})
    return {'deleted': mission_id}


@router.delete('')
def delete_all(actor: str = Depends(require_api_key),
               service: EngineService = Depends(get_service)) -> dict:
    """Purge all stored missions. Destructive and audited."""
    if service.mission_id is not None:
        raise HTTPException(409, 'stop the active mission before purging history')
    removed = service.db.delete_all_missions()
    service.db.audit(actor, 'mission.purge', {'removed': removed}, outcome='destructive')
    return {'deleted_missions': removed}
