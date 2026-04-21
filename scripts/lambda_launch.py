#!/usr/bin/env python3
"""
Orchestrate a training job on Lambda Cloud.

Flow:
  1. Pick an available region for the requested instance type via Lambda API.
  2. Launch a new instance, wait until it is active and reachable via SSH.
  3. scp bootstrap + push_to_hf scripts + secrets onto the instance.
  4. Start training inside tmux; stream logs back to local via `tail -f` over SSH.
  5. Poll for a sentinel file (DONE_SUCCESS / DONE_FAILED) in the remote home dir.
  6. Terminate the instance (unless --keep-alive is passed).

Usage:
  # Pure SFT or CPT-SFT (main.py decides based on config["model"]["name"]):
  python scripts/lambda_launch.py --task sft \
      --config configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml

  # CPT (runs CPT/CPT.py):
  python scripts/lambda_launch.py --task cpt

  # Batch (same instance runs multiple configs sequentially):
  python scripts/lambda_launch.py --task sft \
      --config-list scripts/lambda_batch.txt
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

LAMBDA_API = "https://cloud.lambda.ai/api/v1"
DEFAULT_INSTANCE_TYPE = "gpu_1x_a100_sxm4"  # 1x A100 40GB SXM4
REMOTE_REPO_DIR = "Traditional-Medical-Model"
REMOTE_USER = "ubuntu"
POLL_INTERVAL_LAUNCH = 15  # seconds
POLL_INTERVAL_TRAIN = 60   # seconds


def log(msg: str) -> None:
    print(f"[lambda_launch] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[lambda_launch] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


class LambdaAPI:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.auth = (api_key, "")

    def _req(self, method: str, path: str, **kw: Any) -> dict:
        url = f"{LAMBDA_API}{path}"
        r = self.session.request(method, url, timeout=30, **kw)
        if r.status_code >= 400:
            die(f"{method} {path} -> {r.status_code}: {r.text}")
        return r.json().get("data", r.json())

    def instance_types(self) -> dict:
        return self._req("GET", "/instance-types")

    def list_instances(self) -> list[dict]:
        return self._req("GET", "/instances")

    def get_instance(self, instance_id: str) -> dict:
        return self._req("GET", f"/instances/{instance_id}")

    def launch(self, region: str, instance_type: str, ssh_key_names: list[str],
               name: str | None = None) -> str:
        body = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": ssh_key_names,
            "quantity": 1,
        }
        if name:
            body["name"] = name
        data = self._req("POST", "/instance-operations/launch", json=body)
        ids = data.get("instance_ids", [])
        if not ids:
            die(f"launch returned no instance_ids: {data}")
        return ids[0]

    def terminate(self, instance_id: str) -> None:
        self._req("POST", "/instance-operations/terminate",
                  json={"instance_ids": [instance_id]})


def find_region_for(api: LambdaAPI, instance_type: str) -> str:
    types = api.instance_types()
    entry = types.get(instance_type)
    if not entry:
        available = ", ".join(sorted(types.keys()))
        die(f"instance type {instance_type} not found. Available: {available}")
    regions = entry.get("regions_with_capacity_available", [])
    if not regions:
        die(f"no region has capacity for {instance_type} right now. "
            f"Retry later or pick another type.")
    return regions[0]["name"]


def wait_for_active(api: LambdaAPI, instance_id: str, timeout: int = 600) -> dict:
    log(f"waiting for instance {instance_id} to become active...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        inst = api.get_instance(instance_id)
        status = inst.get("status")
        ip = inst.get("ip")
        log(f"  status={status} ip={ip}")
        if status == "active" and ip:
            return inst
        if status == "terminated":
            die(f"instance {instance_id} was terminated before becoming active")
        time.sleep(POLL_INTERVAL_LAUNCH)
    die(f"instance {instance_id} did not become active within {timeout}s")


def ssh_cmd(key: str, ip: str, remote_cmd: str, *, check: bool = True,
            capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ServerAliveInterval=30",
        "-i", key,
        f"{REMOTE_USER}@{ip}",
        remote_cmd,
    ]
    # errors='replace' so `tail -c +N` mid-multibyte-sequence (Vietnamese
    # text in logs) doesn't crash the orchestrator and terminate the
    # instance mid-eval.
    return subprocess.run(cmd, check=check,
                          capture_output=capture, text=True,
                          encoding="utf-8", errors="replace")


def scp_to(key: str, ip: str, local_path: str, remote_path: str,
           *, retries: int = 4, delay: int = 10) -> None:
    cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
        "-i", key,
        local_path,
        f"{REMOTE_USER}@{ip}:{remote_path}",
    ]
    last_err: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            if attempt < retries:
                log(f"  scp attempt {attempt}/{retries} failed, retrying in {delay}s...")
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def wait_for_ssh(key: str, ip: str, timeout: int = 300,
                 stable_successes: int = 3, stable_interval: int = 5) -> None:
    """
    Wait until SSH is not only reachable but *stable* — require N consecutive
    successful echoes spaced stable_interval seconds apart. Lambda's sshd is
    flaky in the first ~30s after the instance goes active (cloud-init can
    regen host keys), so a single success doesn't mean the next scp will
    succeed.
    """
    log(f"waiting for SSH to come up at {ip}...")
    deadline = time.time() + timeout
    successes = 0
    while time.time() < deadline:
        try:
            ssh_cmd(key, ip, "echo ok", check=True, capture=True)
            successes += 1
            if successes >= stable_successes:
                log(f"  SSH ready ({successes} consecutive successes)")
                return
            time.sleep(stable_interval)
        except subprocess.CalledProcessError:
            successes = 0
            time.sleep(10)
    die(f"SSH did not come up on {ip} within {timeout}s")


def render_env_file(env_vars: dict[str, str]) -> str:
    lines = []
    for k, v in env_vars.items():
        if v is None:
            continue
        lines.append(f"{k}={shlex.quote(str(v))}")
    return "\n".join(lines) + "\n"


def run_job(
    api: LambdaAPI,
    task: str,
    configs: list[str],
    instance_type: str,
    ssh_key_name: str,
    ssh_private_key: str,
    env_vars: dict[str, str],
    git_repo: str,
    git_branch: str,
    keep_alive: bool,
) -> int:
    region = find_region_for(api, instance_type)
    log(f"launching {instance_type} in region {region}")
    job_tag = f"tm-{task}-{int(time.time())}"
    instance_id = api.launch(region, instance_type, [ssh_key_name], name=job_tag)
    log(f"instance_id={instance_id}")

    exit_code = 1
    try:
        inst = wait_for_active(api, instance_id)
        ip = inst["ip"]
        wait_for_ssh(ssh_private_key, ip)

        # Push scripts + env to the instance.
        scripts_dir = Path(__file__).resolve().parent
        scp_to(ssh_private_key, ip, str(scripts_dir / "lambda_bootstrap.sh"),
               "/home/ubuntu/lambda_bootstrap.sh")
        scp_to(ssh_private_key, ip, str(scripts_dir / "lambda_push_to_hf.py"),
               "/home/ubuntu/lambda_push_to_hf.py")

        # Write remote .env with secrets + job params.
        remote_env = {
            **env_vars,
            "JOB_TAG": job_tag,
            "TASK_TYPE": task,
            "CONFIG_LIST": "\n".join(configs),
            "GIT_REPO": git_repo,
            "GIT_BRANCH": git_branch,
        }
        env_body = render_env_file(remote_env)
        env_path = Path("/tmp") / f"{job_tag}.env"
        env_path.write_text(env_body)
        try:
            os.chmod(env_path, 0o600)
            scp_to(ssh_private_key, ip, str(env_path), "/home/ubuntu/job.env")
        finally:
            env_path.unlink(missing_ok=True)

        ssh_cmd(ssh_private_key, ip,
                "chmod 600 /home/ubuntu/job.env && "
                "chmod +x /home/ubuntu/lambda_bootstrap.sh")

        # Kick off training in tmux so the session survives if our ssh drops.
        log("starting training in remote tmux session 'train'")
        ssh_cmd(
            ssh_private_key, ip,
            "tmux new-session -d -s train "
            "'bash /home/ubuntu/lambda_bootstrap.sh 2>&1 | tee /home/ubuntu/train.log'",
        )

        # Follow logs until sentinel file appears. We use a non-fatal tail:
        # if it disconnects, we reconnect and keep polling status.
        log("streaming logs (Ctrl+C to stop streaming — the remote job will "
            "keep running; the instance will still auto-terminate)")
        last_size = 0
        while True:
            # Check sentinels.
            r = ssh_cmd(
                ssh_private_key, ip,
                "ls /home/ubuntu/DONE_SUCCESS /home/ubuntu/DONE_FAILED 2>/dev/null || true",
                check=False, capture=True,
            )
            sentinel = (r.stdout or "").strip()

            # Tail new log bytes since last_size.
            r2 = ssh_cmd(
                ssh_private_key, ip,
                f"wc -c < /home/ubuntu/train.log 2>/dev/null || echo 0",
                check=False, capture=True,
            )
            try:
                new_size = int((r2.stdout or "0").strip())
            except ValueError:
                new_size = last_size
            if new_size > last_size:
                r3 = ssh_cmd(
                    ssh_private_key, ip,
                    f"tail -c +{last_size + 1} /home/ubuntu/train.log",
                    check=False, capture=True,
                )
                sys.stdout.write(r3.stdout or "")
                sys.stdout.flush()
                last_size = new_size

            if "DONE_SUCCESS" in sentinel:
                log("job finished: SUCCESS")
                exit_code = 0
                break
            if "DONE_FAILED" in sentinel:
                log("job finished: FAILED")
                exit_code = 2
                break

            time.sleep(POLL_INTERVAL_TRAIN)

    except KeyboardInterrupt:
        log("interrupted locally — remote job continues; proceeding to terminate.")
    finally:
        if keep_alive:
            log(f"--keep-alive set; leaving instance {instance_id} running")
            log(f"  ssh -i {ssh_private_key} ubuntu@{ip}  # if still reachable")
        else:
            log(f"terminating instance {instance_id}")
            try:
                api.terminate(instance_id)
            except SystemExit:
                raise
            except Exception as e:
                log(f"  WARNING: terminate failed: {e} — terminate manually!")

    return exit_code


def parse_configs(args: argparse.Namespace) -> list[str]:
    if args.config and args.config_list:
        die("pass --config OR --config-list, not both")
    if args.config:
        return [args.config]
    if args.config_list:
        p = Path(args.config_list)
        if not p.exists():
            die(f"config list file not found: {p}")
        configs = [ln.strip() for ln in p.read_text().splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
        if not configs:
            die(f"config list file is empty: {p}")
        return configs
    if args.task == "cpt":
        return []  # CPT.py has no config arg
    die("--config or --config-list required for SFT task")
    return []  # unreachable


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["cpt", "sft"], required=True,
                        help="cpt -> runs CPT/CPT.py; sft -> runs main.py --config <...>")
    parser.add_argument("--config", help="Path to a single SFT config yaml")
    parser.add_argument("--config-list",
                        help="Path to a text file with one config path per line")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"Lambda instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--git-repo",
                        default="https://github.com/Venera-AI/Traditional-Medical-Model.git")
    parser.add_argument("--git-branch", default="main2")
    parser.add_argument("--keep-alive", action="store_true",
                        help="Do not terminate the instance after the job ends")
    parser.add_argument("--list-types", action="store_true",
                        help="List all instance types + availability and exit")
    args = parser.parse_args()

    lambda_key = os.environ.get("LAMBDA_API_KEY")
    if not lambda_key:
        die("LAMBDA_API_KEY missing (set in .env)")

    api = LambdaAPI(lambda_key)

    if args.list_types:
        types = api.instance_types()
        for name, info in sorted(types.items()):
            regions = [r["name"] for r in info.get("regions_with_capacity_available", [])]
            spec = info.get("instance_type", {}).get("specs", {})
            gpus = spec.get("gpus") or info.get("instance_type", {}).get("gpu_description", "")
            print(f"{name:30s} gpus={gpus}  available={regions}")
        return 0

    configs = parse_configs(args)
    ssh_key_name = os.environ.get("LAMBDA_SSH_KEY_NAME")
    if not ssh_key_name:
        die("LAMBDA_SSH_KEY_NAME missing (name of a key already registered on Lambda)")
    ssh_private_key = os.path.expanduser(
        os.environ.get("LAMBDA_SSH_PRIVATE_KEY", "~/.ssh/id_ed25519")
    )
    if not Path(ssh_private_key).exists():
        die(f"private key not found: {ssh_private_key}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        die("HF_TOKEN missing (set in .env)")

    env_vars = {
        "HF_TOKEN": hf_token,
        "WANDB_KEY": os.environ.get("WANDB_KEY", ""),
        "GH_TOKEN": os.environ.get("GH_TOKEN", ""),
        "HF_CPT_REPO": os.environ.get("HF_CPT_REPO", "huyennkai/Venera-CPT-TMModel"),
        "HF_SFT_REPO": os.environ.get("HF_SFT_REPO", "huyennkai/Venera-Traditional-Model"),
        # Docker image overrides (all optional; bootstrap falls back to
        # ghcr.io/venera-ai/traditional-medical-model:<GIT_BRANCH>).
        "DOCKER_IMAGE": os.environ.get("DOCKER_IMAGE", ""),
        "GHCR_LOGIN":   os.environ.get("GHCR_LOGIN", ""),
        "GHCR_USER":    os.environ.get("GHCR_USER", ""),
    }

    return run_job(
        api=api,
        task=args.task,
        configs=configs,
        instance_type=args.instance_type,
        ssh_key_name=ssh_key_name,
        ssh_private_key=ssh_private_key,
        env_vars=env_vars,
        git_repo=args.git_repo,
        git_branch=args.git_branch,
        keep_alive=args.keep_alive,
    )


if __name__ == "__main__":
    sys.exit(main())
