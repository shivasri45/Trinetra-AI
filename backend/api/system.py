"""System, model provenance, interface control and security endpoints."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from backend.api.dependencies import get_service
from backend.config.settings import FAULTS, MISSION_PROFILES, SUBSYSTEMS, settings
from backend.ingestion import can_frames
from backend.ingestion.sources import HAS_PYTHON_CAN
from backend.security import telemetry_security as security
from backend.services.engine_service import EngineService

router = APIRouter(tags=['system'])


class VerifyRequest(BaseModel):
    telemetry: dict
    signature: str


@router.get('/health')
def health(service: EngineService = Depends(get_service)) -> dict:
    return {
        'service': 'online',
        'simulation_running': service.sim.running,
        'active_mission_id': service.mission_id,
        'samples_this_mission': service.sequence,
        'ingestion_mode': service.ingestion_mode,
        'diagnostics_backend': service.ml.backend,
        'python_can_available': HAS_PYTHON_CAN,
        'data_mode': 'SIMULATED PROTOTYPE',
    }


@router.get('/system/capabilities')
def capabilities() -> dict:
    """Everything a client needs to render controls without hardcoding lists."""
    return {
        'engine': {
            'designation': settings.engine.name,
            'cylinders': settings.engine.cylinders,
            'displacement_l': settings.engine.displacement_l,
            'compression_ratio': settings.engine.compression_ratio,
            'rated_power_kw': settings.engine.rated_power_kw,
            'max_rpm': settings.engine.max_rpm,
            'tbo_hours': settings.engine.tbo_hours,
        },
        'faults': list(FAULTS),
        'mission_profiles': list(MISSION_PROFILES),
        'subsystems': list(SUBSYSTEMS),
        'ingestion_modes': ['direct', 'can'],
        'sample_period_s': settings.sample_period_s,
        'thresholds': {'warning_health': settings.warning_health,
                       'critical_health': settings.critical_health,
                       'failure_health': settings.failure_health},
    }


@router.get('/model/metrics')
def model_metrics(service: EngineService = Depends(get_service)) -> dict:
    """Held-out performance of the deployed models, for provenance."""
    path = Path(settings.model_dir) / 'metrics.json'
    if not path.exists():
        return {'available': False, 'backend': service.ml.backend,
                'note': 'Run "python -m backend.ml.train" to produce metrics.'}
    return {'available': True, 'backend': service.ml.backend,
            **json.loads(path.read_text(encoding='utf-8'))}


@router.get('/model/features')
def model_features() -> dict:
    from backend.preprocessing.pipeline import FEATURE_NAMES
    return {'count': len(FEATURE_NAMES), 'features': list(FEATURE_NAMES),
            'note': 'No label, fault identifier, wear state or severity is a feature.'}


@router.get('/interface-control')
def interface_control() -> dict:
    """Machine-readable ICD for the engine data bus."""
    return {
        'bus': {'protocol': 'CAN 2.0B extended, J1939-style addressing',
                'bitrate': 250000, 'source_address': '0x00',
                'frames_per_sample': len(can_frames.MESSAGES),
                'bytes_per_sample': len(can_frames.MESSAGES) * 8,
                'sample_rate_hz': 1.0 / settings.sample_period_s},
        'messages': can_frames.interface_control_rows(),
        'note': ('The raw accelerometer stream is not transmitted. Vibration is '
                 'order-tracked on the edge node and only condition indicators '
                 'cross the bus.'),
    }


@router.get('/security/posture')
def posture() -> dict:
    return security.security_posture()


@router.post('/security/verify')
def verify(body: VerifyRequest) -> dict:
    """Verify an HMAC signature over a telemetry frame."""
    valid = security.verify(body.telemetry, body.signature)
    return {'valid': valid, 'key_id': settings.hmac_key_id,
            'algorithm': 'HMAC-SHA256', 'signed_fields': security.SIGNED_FIELDS_NOTE}


@router.get('/security/audit')
def audit(limit: int = Query(100, ge=1, le=500),
          actor: str = Depends(security.require_api_key),
          service: EngineService = Depends(get_service)) -> dict:
    return {'count': limit, 'entries': service.db.audit_trail(limit)}
