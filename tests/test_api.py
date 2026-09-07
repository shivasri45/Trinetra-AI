"""API surface: authentication, snapshot contract, mission lifecycle and replay."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.config.settings import FAULTS, MISSION_PROFILES
from conftest import API_KEY, AUTH


def wait_for(fetch, predicate, timeout: float = 8.0, interval: float = 0.4):
    """Poll an endpoint until a condition holds; the sampling loop runs at 1 Hz."""
    deadline = time.monotonic() + timeout
    body = fetch()
    while not predicate(body) and time.monotonic() < deadline:
        time.sleep(interval)
        body = fetch()
    assert predicate(body), f'condition not met within {timeout}s: {body}'
    return body


@pytest.fixture(scope='module')
def client():
    from backend.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope='module')
def running(client):
    """A short recorded mission, used by the read and replay tests."""
    client.post('/mission/configure', json={'mission': 'normal_cruise'}, headers=AUTH)
    client.post('/simulation/start', headers=AUTH)
    for _ in range(6):
        client.get('/telemetry/snapshot')
    yield client
    client.post('/simulation/stop', headers=AUTH)


# ------------------------------------------------------------------------ auth
CONTROL_ROUTES = [
    ('post', '/simulation/start', None),
    ('post', '/simulation/stop', None),
    ('post', '/simulation/reset', None),
    ('post', '/simulation/fault', {'fault': 'misfire'}),
    ('post', '/mission/configure', {'mission': 'hot_weather'}),
    ('post', '/ingestion/mode', {'mode': 'direct'}),
    ('delete', '/missions/1', None),
    ('get', '/security/audit', None),
]


@pytest.mark.parametrize('method,path,body', CONTROL_ROUTES)
def test_state_changing_routes_require_a_key(client, method, path, body):
    response = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
    assert response.status_code == 401, path


@pytest.mark.parametrize('method,path,body', CONTROL_ROUTES)
def test_a_wrong_key_is_rejected(client, method, path, body):
    headers = {'X-API-Key': 'not-the-key'}
    response = (getattr(client, method)(path, json=body, headers=headers) if body
                else getattr(client, method)(path, headers=headers))
    assert response.status_code == 401, path


def test_read_routes_are_open(client):
    for path in ('/health', '/system/capabilities', '/interface-control',
                 '/security/posture', '/model/features'):
        assert client.get(path).status_code == 200, path


def test_websocket_requires_a_key(client):
    with pytest.raises(Exception):
        with client.websocket_connect('/ws/telemetry'):
            pass


def test_websocket_streams_with_a_key(client):
    with client.websocket_connect(f'/ws/telemetry?api_key={API_KEY}') as socket:
        snapshot = socket.receive_json()
    assert 'telemetry' in snapshot and 'digital_twin' in snapshot


# -------------------------------------------------------------------- contract
def test_health_reports_service_state(client):
    body = client.get('/health').json()
    assert body['service'] == 'online'
    assert body['data_mode'] == 'SIMULATED PROTOTYPE'
    assert body['ingestion_mode'] in ('direct', 'can')
    assert body['diagnostics_backend'] in ('trained', 'physics_prior_fallback')


def test_capabilities_drive_the_client(client):
    body = client.get('/system/capabilities').json()
    assert set(body['faults']) == set(FAULTS)
    assert set(body['mission_profiles']) == set(MISSION_PROFILES)
    assert body['engine']['tbo_hours'] > 0
    assert body['thresholds']['warning_health'] > body['thresholds']['critical_health']


def test_snapshot_contract(running):
    snapshot = running.get('/telemetry/snapshot').json()
    for key in ('sequence', 'telemetry', 'cylinders', 'vibration', 'digital_twin',
                'prediction', 'rul', 'explanation', 'advisory', 'ingestion', 'integrity'):
        assert key in snapshot, key
    twin = snapshot['digital_twin']
    for key in ('expected', 'estimated', 'measured', 'residuals', 'normalised_residuals',
                'subsystems', 'sensor_validation', 'health_score', 'operating_state'):
        assert key in twin, key
    assert snapshot['rul']['tbo_hours'] > 0
    assert snapshot['advisory']['dispatch'] in ('GO', 'GO_WITH_LIMITATION', 'NO_GO')


def test_ground_truth_is_labelled_as_demonstration_only(running):
    body = running.get('/fault').json()
    assert 'injected_fault_for_demo' in body
    assert 'predicted_fault' in body and 'class_probabilities' in body


def test_signed_frames_verify_and_detect_tampering(running):
    snapshot = running.get('/telemetry/snapshot').json()
    signature = snapshot['integrity']['signature']
    good = running.post('/security/verify',
                        json={'telemetry': snapshot['telemetry'], 'signature': signature}).json()
    assert good['valid'] is True
    tampered = dict(snapshot['telemetry'], rpm=1.0)
    bad = running.post('/security/verify',
                       json={'telemetry': tampered, 'signature': signature}).json()
    assert bad['valid'] is False


def test_history_and_read_endpoints(running):
    assert running.get('/telemetry/latest').json()['rpm'] > 0
    assert len(running.get('/telemetry/history?limit=5').json()) <= 5
    cylinders = running.get('/telemetry/cylinders').json()
    assert len(cylinders['measured']['egt']) == 4
    assert len(cylinders['expected']['egt']) == 4
    assert running.get('/digital-twin/subsystems').json()
    assert 'plausibility' in running.get('/digital-twin/sensor-validation').json()
    assert 'summary' in running.get('/explanation').json()
    assert 'dispatch' in running.get('/advisory').json()
    spectrum = running.get('/vibration/spectrum').json()
    assert spectrum['spectrum'] and 'indicators' in spectrum


def test_model_metrics_expose_provenance(client):
    body = client.get('/model/metrics').json()
    if not body.get('available'):
        pytest.skip('no trained model present')
    assert 0.0 <= body['classifier']['accuracy'] <= 1.0
    assert 'GroupShuffleSplit' in body['split']['strategy']
    assert body['novelty']['threshold_sweep']


def test_feature_list_declares_no_leakage(client):
    body = client.get('/model/features').json()
    assert body['count'] > 50
    assert 'No label' in body['note']
    assert not [name for name in body['features'] if 'fault' in name or 'severity' in name]


def test_interface_control_document_is_served(client):
    body = client.get('/interface-control').json()
    assert body['bus']['protocol'].startswith('CAN')
    assert len(body['messages']) >= 10
    assert 'not transmitted' in body['note']


# --------------------------------------------------------------------- control
def test_unsupported_values_are_rejected(client):
    assert client.post('/simulation/fault', json={'fault': 'nope'},
                       headers=AUTH).status_code == 422
    assert client.post('/mission/configure', json={'mission': 'nope'},
                       headers=AUTH).status_code == 422
    assert client.post('/ingestion/mode', json={'mode': 'nope'},
                       headers=AUTH).status_code == 422


def test_fault_injection_and_mission_change_are_accepted(running):
    assert running.post('/simulation/fault', json={'fault': 'misfire'},
                        headers=AUTH).json()['fault'] == 'misfire'
    assert running.post('/mission/configure', json={'mission': 'high_altitude'},
                        headers=AUTH).json()['mission'] == 'high_altitude'
    running.post('/simulation/fault', json={'fault': 'normal'}, headers=AUTH)
    running.post('/mission/configure', json={'mission': 'normal_cruise'}, headers=AUTH)


def test_switching_to_can_ingestion_keeps_the_pipeline_working(running):
    assert running.post('/ingestion/mode', json={'mode': 'can'},
                        headers=AUTH).json()['mode'] == 'can'
    assert running.get('/ingestion/status').json()['mode'] == 'can'
    try:
        # The change takes effect on the next acquisition cycle.
        frames = wait_for(lambda: running.get('/ingestion/frames').json(),
                          lambda body: bool(body['frames']))
        assert frames['frames'][0]['can_id'].startswith('0x')
        assert len(frames['frames'][0]['data'].split()) == 8
        snapshot = running.get('/telemetry/snapshot').json()
        assert snapshot['ingestion']['mode'] == 'can'
        assert snapshot['telemetry']['rpm'] > 0
        assert snapshot['digital_twin']['health_score'] > 0
    finally:
        running.post('/ingestion/mode', json={'mode': 'direct'}, headers=AUTH)


def test_audit_trail_records_control_actions(client):
    client.post('/simulation/fault', json={'fault': 'normal'}, headers=AUTH)
    entries = client.get('/security/audit', headers=AUTH).json()['entries']
    actions = {entry['action'] for entry in entries}
    assert 'fault.inject' in actions


# ---------------------------------------------------------------- missions
def test_mission_is_recorded_and_can_be_replayed(client):
    client.post('/mission/configure', json={'mission': 'long_endurance'}, headers=AUTH)
    started = client.post('/simulation/start', headers=AUTH).json()
    mission_id = started['mission_id']
    assert mission_id is not None
    for _ in range(5):
        client.get('/telemetry/snapshot')
    client.post('/simulation/stop', headers=AUTH)

    listing = client.get('/missions').json()
    assert listing['active_mission_id'] is None
    assert any(m['id'] == mission_id for m in listing['missions'])

    mission = client.get(f'/missions/{mission_id}').json()
    assert mission['ended_at'] is not None
    assert mission['stored_samples'] >= 1

    timeline = client.get(f'/missions/{mission_id}/timeline').json()['points']
    assert timeline and 'health' in timeline[0] and 'seq' in timeline[0]

    page = client.get(f'/missions/{mission_id}/snapshots?start=0&limit=3').json()
    assert page['returned'] >= 1
    assert 'digital_twin' in page['snapshots'][0]

    report = client.get(f'/missions/{mission_id}/report').json()
    assert report['summary']['samples'] >= 1
    assert 'advisory' in report['summary']


def test_reset_preserves_stored_missions(client):
    before = client.get('/missions').json()['count']
    body = client.post('/simulation/reset', headers=AUTH).json()
    assert body['missions_preserved'] is True
    assert client.get('/missions').json()['count'] == before


def test_missing_mission_returns_not_found(client):
    assert client.get('/missions/999999').status_code == 404
    assert client.get('/missions/999999/report').status_code == 404
    assert client.delete('/missions/999999', headers=AUTH).status_code == 404


def test_active_mission_cannot_be_deleted(client):
    started = client.post('/simulation/start', headers=AUTH).json()
    mission_id = started['mission_id']
    try:
        assert client.delete(f'/missions/{mission_id}', headers=AUTH).status_code == 409
        assert client.delete('/missions', headers=AUTH).status_code == 409
    finally:
        client.post('/simulation/stop', headers=AUTH)


def test_mission_can_be_deleted_after_it_ends(client):
    started = client.post('/simulation/start', headers=AUTH).json()
    mission_id = started['mission_id']
    client.get('/telemetry/snapshot')
    client.post('/simulation/stop', headers=AUTH)
    assert client.delete(f'/missions/{mission_id}', headers=AUTH).json()['deleted'] == mission_id
    assert client.get(f'/missions/{mission_id}').status_code == 404
