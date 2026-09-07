"""Read-only endpoints for live telemetry, twin state and analytics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import get_service
from backend.services.engine_service import EngineService

router = APIRouter(tags=['telemetry'])


@router.get('/telemetry/latest')
def latest(service: EngineService = Depends(get_service)) -> dict:
    return service.snapshot()['telemetry']


@router.get('/telemetry/snapshot')
def snapshot(service: EngineService = Depends(get_service)) -> dict:
    """The complete synchronised snapshot: telemetry, twin, diagnosis, RUL, advisory."""
    return service.snapshot()


@router.get('/telemetry/history')
def history(limit: int = Query(300, ge=1, le=900),
            service: EngineService = Depends(get_service)) -> list[dict]:
    return list(service.history)[-limit:]


@router.get('/telemetry/cylinders')
def cylinders(service: EngineService = Depends(get_service)) -> dict:
    snap = service.snapshot()
    return {
        'measured': snap['cylinders'],
        'expected': snap['digital_twin']['cylinders']['expected'],
        'egt_spread': snap['telemetry'].get('egt_spread'),
        'cht_spread': snap['telemetry'].get('cht_spread'),
        'hottest_cylinder': snap['telemetry'].get('hottest_cylinder'),
        'coldest_cylinder': snap['telemetry'].get('coldest_cylinder'),
    }


@router.get('/digital-twin/state')
def twin_state(service: EngineService = Depends(get_service)) -> dict:
    return service.snapshot()['digital_twin']


@router.get('/digital-twin/subsystems')
def subsystems(service: EngineService = Depends(get_service)) -> dict:
    return service.snapshot()['digital_twin']['subsystems']


@router.get('/digital-twin/sensor-validation')
def sensor_validation(service: EngineService = Depends(get_service)) -> dict:
    """Analytical-redundancy check across physically coupled channels."""
    return service.snapshot()['digital_twin']['sensor_validation']


@router.get('/prediction')
def prediction(service: EngineService = Depends(get_service)) -> dict:
    return service.snapshot()['prediction']


@router.get('/fault')
def fault(service: EngineService = Depends(get_service)) -> dict:
    snap = service.snapshot()
    return {
        'predicted_fault': snap['prediction']['predicted_fault'],
        'confidence': snap['prediction']['confidence'],
        'anomaly': snap['prediction']['anomaly'],
        'detection_triggers': snap['prediction']['detection_triggers'],
        'class_probabilities': snap['prediction']['class_probabilities'],
        'injected_fault_for_demo': (snap.get('simulated_truth') or {}).get('fault'),
    }


@router.get('/rul')
def rul(service: EngineService = Depends(get_service)) -> dict:
    return service.snapshot()['rul']


@router.get('/explanation')
def explanation(service: EngineService = Depends(get_service)) -> dict:
    return service.snapshot()['explanation']


@router.get('/advisory')
def advisory(service: EngineService = Depends(get_service)) -> dict:
    return service.snapshot()['advisory']


@router.get('/vibration/spectrum')
def spectrum(service: EngineService = Depends(get_service)) -> dict:
    snap = service.snapshot()
    return {
        'spectrum': service.spectrum,
        'indicators': snap['vibration'],
        'rotational_hz': snap['vibration'].get('rotational_hz'),
        'firing_hz': snap['vibration'].get('firing_hz'),
        'sample_rate_hz': 2048,
        'note': 'Order-tracked on the edge node; only indicators cross the CAN bus.',
    }


@router.get('/alarms')
def alarms(service: EngineService = Depends(get_service)) -> list[dict]:
    return list(service.alarm_log)[::-1]


@router.get('/ingestion/status')
def ingestion_status(service: EngineService = Depends(get_service)) -> dict:
    """Configured mode, plus the transport detail of the last acquired frame.

    The two can differ for one sample after a mode change, which is why the
    configured value is reported separately rather than merged.
    """
    snap = service.snapshot()
    return {'mode': service.ingestion_mode, 'last_frame_transport': snap['ingestion']}


@router.get('/ingestion/frames')
def ingestion_frames(service: EngineService = Depends(get_service)) -> dict:
    """Raw CAN frames from the most recent acquisition cycle."""
    snap = service.snapshot()
    return {
        'mode': service.ingestion_mode,
        'frames': service.raw_frames,
        'transport': snap['ingestion'],
        'note': (None if service.ingestion_mode == 'can'
                 else 'Switch to CAN ingestion with POST /ingestion/mode to populate this.'),
    }
