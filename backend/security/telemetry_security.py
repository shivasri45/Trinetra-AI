"""Authentication, telemetry integrity and audit logging.

Three controls, all of which a defence-grade ground station would be expected to
have and none of which are free:

* **Authentication.** Control endpoints require an API key supplied in the
  ``X-API-Key`` header, compared in constant time. Read endpoints can be opened
  or closed with one setting. Without this, anyone able to reach the port could
  inject a fault or wipe the mission history.
* **Integrity.** Every telemetry snapshot is signed with HMAC-SHA256 over a
  canonical JSON serialisation of the telemetry frame. A recipient can therefore
  detect a modified or spoofed frame. Signing covers integrity and authenticity,
  not confidentiality: transport encryption is still TLS's job.
* **Audit.** Every state-changing action is recorded with actor, action, detail
  and timestamp, so post-incident reconstruction is possible.

The default development key is deliberately obvious. In deployment both keys come
from the environment, and the service refuses to start with defaults when
``TRINETRA_ENV=production``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone

from fastapi import Header, HTTPException, status

from backend.config.settings import settings

DEV_API_KEY = 'trinetra-dev-key'
DEV_HMAC_KEY = 'trinetra-dev-hmac'
SIGNED_FIELDS_NOTE = 'HMAC-SHA256 over canonical JSON of the telemetry frame'


def is_production() -> bool:
    return os.getenv('TRINETRA_ENV', 'development').lower() == 'production'


def startup_warnings() -> list[str]:
    warnings: list[str] = []
    if not settings.auth_enabled:
        warnings.append('Authentication is DISABLED; control endpoints are open to any caller.')
    if settings.api_key == DEV_API_KEY:
        warnings.append('Using the default development API key; set TRINETRA_API_KEY.')
    if settings.hmac_key == DEV_HMAC_KEY:
        warnings.append('Using the default development HMAC key; set TRINETRA_HMAC_KEY.')
    if is_production() and (settings.api_key == DEV_API_KEY or settings.hmac_key == DEV_HMAC_KEY
                            or not settings.auth_enabled):
        raise RuntimeError('Refusing to start in production with default or disabled credentials. '
                           'Set TRINETRA_API_KEY, TRINETRA_HMAC_KEY and TRINETRA_AUTH=1.')
    return warnings


# ------------------------------------------------------------------ auth guard
def require_api_key(x_api_key: str | None = Header(default=None, alias='X-API-Key')) -> str:
    """FastAPI dependency guarding state-changing endpoints."""
    if not settings.auth_enabled:
        return 'anonymous'
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Valid X-API-Key header required',
                            headers={'WWW-Authenticate': 'ApiKey'})
    return f'key:{hashlib.sha256(x_api_key.encode()).hexdigest()[:12]}'


def check_api_key(value: str | None) -> bool:
    """Non-raising variant, used by the WebSocket handshake."""
    if not settings.auth_enabled:
        return True
    return bool(value) and hmac.compare_digest(value, settings.api_key)


# ------------------------------------------------------------------- integrity
def canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      default=str).encode('utf-8')


def sign(payload: dict) -> dict:
    digest = hmac.new(settings.hmac_key.encode(), canonical(payload), hashlib.sha256).hexdigest()
    return {
        'algorithm': 'HMAC-SHA256',
        'key_id': settings.hmac_key_id,
        'signature': digest,
        'signed_fields': SIGNED_FIELDS_NOTE,
        'signed_at': datetime.now(timezone.utc).isoformat(),
    }


def verify(payload: dict, signature: str) -> bool:
    expected = hmac.new(settings.hmac_key.encode(), canonical(payload), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or '')


def security_posture() -> dict:
    return {
        'authentication': 'api-key' if settings.auth_enabled else 'disabled',
        'telemetry_integrity': 'HMAC-SHA256 per frame',
        'hmac_key_id': settings.hmac_key_id,
        'transport_encryption': 'not terminated here; deploy behind TLS',
        'audit_logging': 'enabled',
        'environment': 'production' if is_production() else 'development',
        'using_default_keys': settings.api_key == DEV_API_KEY or settings.hmac_key == DEV_HMAC_KEY,
    }
