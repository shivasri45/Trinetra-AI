"""Central configuration for the Trinetra AI digital-twin prototype."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEV_CORS_ORIGINS: tuple[str, ...] = ('http://localhost:5173', 'http://127.0.0.1:5173')


def _cors_origins_from_env() -> tuple[str, ...]:
    """Browser origins permitted to call the API.

    Comma-separated in ``TRINETRA_CORS_ORIGINS``; defaults to the Vite dev
    server. Trailing slashes are stripped because the browser's ``Origin``
    header never carries one, and an entry written as ``https://host/`` would
    otherwise never match.

    A deployment that serves the dashboard and the API from the same origin does
    not need this at all. Avoid ``*``: these requests are credentialed, and
    browsers reject a wildcard on a credentialed response.
    """
    raw = os.getenv('TRINETRA_CORS_ORIGINS', '')
    origins = tuple(item.strip().rstrip('/') for item in raw.split(',') if item.strip())
    return origins or DEV_CORS_ORIGINS


@dataclass(frozen=True)
class EngineSpec:
    """Reference geometry for a MALE-UAV-class four-stroke aero piston engine.

    Values are representative of a ~1.2 L four-cylinder liquid/air-cooled unit
    in the Rotax 912/914 class. They are public specification-level figures used
    only to keep the synthetic model dimensionally realistic.
    """

    name: str = 'Generic 1.35L 4-cyl 4-stroke aero piston'
    cylinders: int = 4
    displacement_l: float = 1.352
    compression_ratio: float = 10.5
    strokes: int = 4
    idle_rpm: float = 1400.0
    max_rpm: float = 5800.0
    rated_power_kw: float = 73.5
    tbo_hours: float = 2000.0
    fuel_lhv_mj_kg: float = 43.5
    fuel_density_kg_l: float = 0.72
    stoich_afr: float = 14.7

    @property
    def displacement_m3(self) -> float:
        return self.displacement_l / 1000.0


@dataclass(frozen=True)
class Settings:
    # --- simulation ---
    seed: int = 42
    sample_period_s: float = 1.0
    history_limit: int = 900
    engine: EngineSpec = field(default_factory=EngineSpec)

    # --- health thresholds ---
    warning_health: float = 80.0
    critical_health: float = 50.0
    failure_health: float = 45.0  # RUL extrapolation target

    # --- anomaly detection ---
    # Aggregate test across channels, plus a per-channel test. The aggregate
    # dilutes a fault confined to one channel (one bad channel out of twenty-odd
    # barely moves the mean), so a single channel beyond 4 sigma is treated as
    # significant on its own.
    nis_warning: float = 9.0        # mean normalised innovation squared
    nis_peak_warning: float = 16.0  # worst single channel, 4 sigma
    nis_critical: float = 25.0

    # --- vibration / signal processing ---
    vib_sample_rate_hz: int = 2048
    vib_block_samples: int = 1024

    # --- machine learning ---
    model_dir: Path = Path('models')
    dataset_path: Path = Path('data/synthetic/engine_faults.csv')

    # --- persistence ---
    db_path: Path = Path(os.getenv('TRINETRA_DB', 'data/trinetra.db'))

    # --- ingestion ---
    ingestion_mode: str = os.getenv('TRINETRA_INGESTION', 'direct')  # direct | can
    can_channel: str = os.getenv('TRINETRA_CAN_CHANNEL', 'trinetra')
    can_interface: str = os.getenv('TRINETRA_CAN_INTERFACE', 'virtual')

    # --- security ---
    auth_enabled: bool = os.getenv('TRINETRA_AUTH', '1') != '0'
    api_key: str = os.getenv('TRINETRA_API_KEY', 'trinetra-dev-key')
    hmac_key: str = os.getenv('TRINETRA_HMAC_KEY', 'trinetra-dev-hmac')
    hmac_key_id: str = 'gcs-dev-01'
    cors_origins: tuple[str, ...] = field(default_factory=_cors_origins_from_env)


settings = Settings()

FAULTS: tuple[str, ...] = (
    'normal',
    'misfire',
    'injector_abnormality',
    'overheating',
    'lubrication_issue',
    'excessive_vibration',
    'sensor_drift',
    'combustion_instability',
    'coking_degradation',
    'alternator_failure',
)

MISSION_PROFILES: tuple[str, ...] = (
    'normal_cruise',
    'high_altitude',
    'hot_weather',
    'long_endurance',
    'rapid_throttle_transition',
)

SUBSYSTEMS: tuple[str, ...] = (
    'thermal',
    'lubrication',
    'combustion',
    'mechanical',
    'electrical',
    'fuel_delivery',
)
