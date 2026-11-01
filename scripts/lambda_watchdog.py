#!/usr/bin/env python3
"""Instance-side Lambda Cloud termination watchdog (standard library only)."""
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

TERMINATE_URL = "https://cloud.lambda.ai/api/v1/instance-operations/terminate"


def terminate(config):
    """Use the provider API: OS shutdown alone does not stop Lambda billing."""
    token = base64.b64encode((config["api_key"] + ":").encode()).decode()
    request = urllib.request.Request(
        TERMINATE_URL,
        data=json.dumps({"instance_ids": [config["instance_id"]]}).encode(),
        headers={"Authorization": "Basic " + token, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    terminated = result.get("data", {}).get("terminated_instances", [])
    if not any(item.get("id") == config["instance_id"] for item in terminated):
        raise RuntimeError("API did not confirm termination")


def main(config_path):
    config = json.loads(Path(config_path).read_text())
    print("Termination watchdog armed", flush=True)
    while True:
        remaining = config["deadline"] - time.time()
        if remaining > 0:
            time.sleep(min(remaining, 60))
            continue
        try:
            terminate(config)
            print("Termination accepted by Lambda Cloud", flush=True)
        except Exception:
            # Never log response bodies or exceptions that could expose secrets.
            print("Termination not confirmed; retrying in 60 seconds", flush=True)
        # Retry even after acceptance until the instance actually disappears.
        time.sleep(60)


if __name__ == "__main__":
    main(sys.argv[1])
