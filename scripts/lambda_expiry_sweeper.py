#!/usr/bin/env python3
"""Terminate only instances launched with this project's expiry marker."""
import base64
import json
import os
import re
import sys
import time
import urllib.request

API = "https://cloud.lambda.ai/api/v1"
NAME = re.compile(r"tmm-exp-([0-9]{10})-([0-9]{10})\Z")
LIFETIME = 12 * 60 * 60


def request(key, path, payload=None):
    token = base64.b64encode((key + ':').encode()).decode()
    req = urllib.request.Request(
        API + path, data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization': 'Basic ' + token, 'Content-Type': 'application/json'},
        method='GET' if payload is None else 'POST')
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)['data']


def sweep(key, now):
    instances = request(key, '/instances')
    if not isinstance(instances, list):
        raise ValueError('Unexpected instance list response')
    failures = False
    for instance in instances:
        match = NAME.fullmatch(instance.get('name') or '')
        if not match:
            continue
        deadline, launched = map(int, match.groups())
        if not 1 <= deadline - launched <= LIFETIME or now < deadline:
            continue
        if instance.get('status') in ('terminated', 'terminating', 'preempted'):
            continue
        instance_id = instance['id']
        try:
            time.sleep(1)  # Lambda Cloud generally limits requests to one per second.
            result = request(key, '/instance-operations/terminate', {'instance_ids': [instance_id]})
            if not any(item.get('id') == instance_id for item in result.get('terminated_instances', [])):
                raise ValueError('Termination not confirmed')
            print(f'Termination accepted for {instance_id}', flush=True)
        except Exception:
            print(f'Termination failed for {instance_id}; next scheduled run will retry', file=sys.stderr)
            failures = True
    if failures:
        raise RuntimeError('One or more termination requests failed')


if __name__ == '__main__':
    key = os.environ.get('LAMBDA_API_KEY')
    if not key:
        sys.exit('LAMBDA_API_KEY is missing')
    try:
        sweep(key, time.time())
    except Exception:
        sys.exit('Lambda expiry sweep failed; check API access and workflow configuration')
