"""End-to-end smoke test against a running server.

    python -m uvicorn backend.main:app --port 8010
    python -m tools.smoke_test --base http://localhost:8010

Exercises the demonstration path a reviewer would follow: healthy operation
through a throttle-transition profile, a misfire with its evidence, the CAN
transport, and the filed mission report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

KEY = 'trinetra-dev-key'


def call(base: str, path: str, method: str = 'GET', body: dict | None = None,
         auth: bool = False):
    data = json.dumps(body).encode() if body is not None else (b'{}' if method == 'POST' else None)
    request = urllib.request.Request(base + path, data=data, method=method)
    request.add_header('Content-Type', 'application/json')
    if auth:
        request.add_header('X-API-Key', KEY)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read() or b'null')
    except urllib.error.HTTPError as exc:
        return exc.code, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://localhost:8010')
    base = parser.parse_args().base
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = '') -> None:
        print(f'{"PASS" if condition else "FAIL"}  {label}{f"  {detail}" if detail else ""}')
        if not condition:
            failures.append(label)

    status, health = call(base, '/health')
    check('service online', status == 200 and health['service'] == 'online',
          f"backend={health['diagnostics_backend']}" if health else '')

    status, _ = call(base, '/simulation/start', 'POST')
    check('unauthenticated control rejected', status == 401, f'status={status}')

    # Start from a clean engine so the script is idempotent against a
    # long-running server. Stopping a mission deliberately does not clear an
    # injected fault, so a previous run would otherwise leave one active.
    call(base, '/simulation/reset', 'POST', auth=True)
    call(base, '/simulation/fault', 'POST', {'fault': 'normal'}, auth=True)
    call(base, '/mission/configure', 'POST', {'mission': 'rapid_throttle_transition'}, auth=True)

    status, started = call(base, '/simulation/start', 'POST', auth=True)
    mission_id = started['mission_id'] if started else None
    check('authenticated start', status == 200 and mission_id is not None,
          f'mission={mission_id}')

    # The engine starts cold, so poll for thermal settling rather than guessing.
    deadline = time.monotonic() + 90
    snap = None
    while time.monotonic() < deadline:
        time.sleep(3)
        _, snap = call(base, '/telemetry/snapshot')
        twin = snap['digital_twin']
        if twin['operating_state'] == 'NORMAL' and not snap['prediction']['anomaly']:
            break
    twin = snap['digital_twin']
    check('healthy through throttle transients',
          twin['operating_state'] == 'NORMAL' and not snap['prediction']['anomaly'],
          f"health={twin['health_score']} nis={twin['mean_nis']} "
          f"peak={twin['peak_nis']} after {snap['sequence']} samples")

    call(base, '/simulation/fault', 'POST', {'fault': 'misfire'}, auth=True)
    time.sleep(15)
    _, snap = call(base, '/telemetry/snapshot')
    prediction, twin = snap['prediction'], snap['digital_twin']
    check('misfire diagnosed', prediction['predicted_fault'] == 'misfire',
          f"conf={prediction['confidence']} health={twin['health_score']}")
    check('misfire localised to a cylinder',
          snap['telemetry']['egt_spread'] > 150,
          f"spread={snap['telemetry']['egt_spread']} coldest=#{snap['telemetry']['coldest_cylinder']}")
    evidence = ', '.join(f"{item['feature']} {item['contribution']}%"
                         for item in snap['explanation']['evidence'][:3])
    check('evidence attributed', bool(snap['explanation']['evidence']), evidence)
    check('advisory issued', snap['advisory']['dispatch'] in ('GO_WITH_LIMITATION', 'NO_GO'),
          f"{snap['advisory']['dispatch']} / {snap['advisory']['maintenance_priority']} "
          f"/ limiting={snap['rul']['limiting_subsystem']}")

    signature = snap['integrity']['signature']
    _, good = call(base, '/security/verify', 'POST',
                   {'telemetry': snap['telemetry'], 'signature': signature})
    _, bad = call(base, '/security/verify', 'POST',
                  {'telemetry': dict(snap['telemetry'], rpm=1.0), 'signature': signature})
    check('frame integrity verifies and detects tampering',
          good['valid'] and not bad['valid'])

    call(base, '/ingestion/mode', 'POST', {'mode': 'can'}, auth=True)
    time.sleep(4)
    _, frames = call(base, '/ingestion/frames')
    first = frames['frames'][0] if frames['frames'] else {}
    check('can transport carries the frame set', len(frames['frames']) >= 10,
          f"{len(frames['frames'])} frames, first {first.get('can_id')} {first.get('data')}")
    _, snap = call(base, '/telemetry/snapshot')
    check('analytics intact over can', snap['telemetry']['rpm'] > 1000
          and snap['digital_twin']['health_score'] >= 0,
          f"rpm={snap['telemetry']['rpm']} health={snap['digital_twin']['health_score']}")
    call(base, '/ingestion/mode', 'POST', {'mode': 'direct'}, auth=True)

    _, stopped = call(base, '/simulation/stop', 'POST', auth=True)
    _, report = call(base, f'/missions/{mission_id}/report')
    check('mission filed with a report', report['summary']['samples'] > 0,
          f"samples={report['summary']['samples']} health_min={report['summary'].get('health_min')} "
          f"alarms={len(report['summary'].get('alarms', []))}")
    _, timeline = call(base, f'/missions/{mission_id}/timeline')
    check('replay timeline available', len(timeline['points']) > 0,
          f"{len(timeline['points'])} points")

    _, icd = call(base, '/interface-control')
    _, metrics = call(base, '/model/metrics')
    check('icd and model provenance served',
          len(icd['messages']) >= 10 and metrics.get('available'),
          f"{len(icd['messages'])} messages, accuracy="
          f"{metrics.get('classifier', {}).get('accuracy')}")

    _, audit = call(base, '/security/audit', auth=True)
    check('audit trail populated', len(audit['entries']) > 0,
          f"{len(audit['entries'])} entries")

    print()
    if failures:
        print(f'{len(failures)} check(s) failed: {", ".join(failures)}')
        return 1
    print('all smoke checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
