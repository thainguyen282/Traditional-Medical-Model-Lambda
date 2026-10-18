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
import base64
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
        # /instance-operations/launch can take 30-60s server-side. A 30s
        # client timeout there leaves an orphan instance: the launch
        # succeeds on Lambda's side but our requests.post raises ReadTimeout,
        # we never persist the instance_id, never scp bootstrap, never
        # terminate. Give launch endpoints ample time.
        timeout = kw.pop("timeout", 90 if "launch" in path else 30)
        r = self.session.request(method, url, timeout=timeout, **kw)
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


def find_region_for(api: LambdaAPI, instance_type: str, region: str | None = None) -> str:
    types = api.instance_types()
    entry = types.get(instance_type)
    if not entry:
        available = ", ".join(sorted(types.keys()))
        die(f"instance type {instance_type} not found. Available: {available}")
    regions = entry.get("regions_with_capacity_available", [])
    if region is not None:
        if any(item["name"] == region for item in regions):
            return region
        available = ", ".join(item["name"] for item in regions) or "none"
        die(f"{instance_type} has no available capacity in requested region {region}. "
            f"Available regions: {available}. No instance was launched.")
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


def check_git_access(repo: str, branch: str, token: str | None) -> None:
    """Check HTTPS clone access without prompts or credentials in command arguments."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    url = urlsplit(repo)
    if token and url.scheme == "https" and url.hostname == "github.com":
        index = int(env.get("GIT_CONFIG_COUNT", "0"))
        env["GIT_CONFIG_COUNT"] = str(index + 1)
        env[f"GIT_CONFIG_KEY_{index}"] = "http.https://github.com/.extraheader"
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env[f"GIT_CONFIG_VALUE_{index}"] = f"Authorization: Basic {auth}"
    try:
        result = subprocess.run(
            ["git", "-c", "credential.helper=", "ls-remote", "--exit-code", "--heads",
             "--", repo, f"refs/heads/{branch}"],
            env=env, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("Git repository preflight could not complete; check connectivity and Git installation") from None
    if result.returncode or not any(
            line.split()[-1:] == [f"refs/heads/{branch}"] for line in result.stdout.splitlines()):
        # Do not echo git stderr: it can include credential-bearing URLs.
        raise RuntimeError("Git repository/branch is inaccessible. Check --git-repo, --git-branch, "
                           "and GH_TOKEN access before retrying.")


def copy_medical_eval_bundle(key: str, ip: str, datasets: list[str], general_vi: bool = False) -> None:
    from eval_medical_baselines import DATASETS

    root = Path(__file__).resolve().parent.parent
    files = {"scripts/eval_medical_baselines.py", "scripts/setup_baseline_env.sh",
             "requirements-baselines.txt"}
    if general_vi:
        files.add("scripts/eval_general_vi.py")
        files.update({"scripts/eval_general_vi_vllm.py", "scripts/setup_vllm_env.sh"})
    else:
        for dataset in datasets:
            filename, notebook = DATASETS[dataset]
            files.update({f"EVAL/Data/{filename}", f"EVAL/{notebook}"})
        if "vm14k-full" in datasets:
            files.add(f"EVAL/Data/{DATASETS['vm14k'][0]}")
    with tempfile.TemporaryDirectory(prefix="medical-eval-") as directory:
        bundle = Path(directory) / "medical_eval_bundle.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for name in sorted(files):
                archive.add(root / name, arcname=name)
        scp_to(key, ip, str(bundle), "/home/ubuntu/medical_eval_bundle.tar.gz")


def copy_new_cpt_bundle(key: str, ip: str) -> None:
    root = Path(__file__).resolve().parent.parent
    files = ("CPT/train_new_bases.py", "requirements-cpt-new.txt",
             "requirements-baselines.txt")
    with tempfile.TemporaryDirectory(prefix="new-cpt-") as directory:
        bundle = Path(directory) / "new_cpt_bundle.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for name in files:
                archive.add(root / name, arcname=name)
        scp_to(key, ip, str(bundle), "/home/ubuntu/new_cpt_bundle.tar.gz")


def download_medical_eval(key: str, ip: str, job_tag: str) -> Path | None:
    remote_archive = f"/home/ubuntu/{job_tag}-medical-eval.tar.gz"
    remote_outputs = f"/home/ubuntu/{REMOTE_REPO_DIR}/EVAL/outputs/medical_baselines/{job_tag}"
    exists = ssh_cmd(key, ip, f"test -d {shlex.quote(remote_outputs)}", check=False, capture=True)
    if exists.returncode == 1:
        log("No baseline output directory: setup failed before evaluation; skipping result download.")
        return None
    exists.check_returncode()
    ssh_cmd(key, ip, f"sudo tar -czf {shlex.quote(remote_archive)} "
            f"-C {shlex.quote(remote_outputs)} . && sudo chmod 644 {shlex.quote(remote_archive)}")
    destination = Path("logs") / job_tag
    destination.mkdir(parents=True, exist_ok=True)
    local_archive = destination / "medical-eval.tar.gz"
    subprocess.run([
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15", "-i", key,
        f"{REMOTE_USER}@{ip}:{remote_archive}", str(local_archive),
    ], check=True, timeout=120)
    with tarfile.open(local_archive) as archive:
        if not archive.getmembers():
            raise RuntimeError("Downloaded evaluation archive is empty")
    log(f"evaluation backup saved locally: {local_archive.resolve()}")
    return local_archive


def download_sft_backup(key: str, ip: str, job_tag: str) -> Path:
    """Preserve outputs and diagnostics before destroying the training host."""
    remote_archive = f"/home/ubuntu/{job_tag}-sft-backup.tar.gz"
    repo = f"/home/ubuntu/{REMOTE_REPO_DIR}"
    command = (
        f"cd {shlex.quote(repo)} && "
        "paths=(); "
        "for p in group_outputs outputs EVAL/outputs app.log; do "
        "if [ -e \"$p\" ]; then paths+=(\"$p\"); fi; done; "
        f"sudo tar -czf {shlex.quote(remote_archive)} "
        "\"${paths[@]}\" -C /home/ubuntu train.log && "
        f"sudo chmod 644 {shlex.quote(remote_archive)}"
    )
    ssh_cmd(key, ip, "bash -lc " + shlex.quote(command))
    destination = Path("logs") / job_tag
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "sft-backup.tar.gz"
    subprocess.run([
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15", "-i", key,
        f"{REMOTE_USER}@{ip}:{remote_archive}", str(archive_path),
    ], check=True, timeout=1800)
    with tarfile.open(archive_path) as archive:
        if not any(member.name == "train.log" for member in archive.getmembers()):
            raise RuntimeError("SFT backup is missing train.log")
    log(f"SFT backup saved locally: {archive_path.resolve()}")
    return archive_path


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
    region: str | None = None,
    boot_timeout: int = 600,
) -> int:
    region = find_region_for(api, instance_type, region)
    log(f"launching {instance_type} in region {region}")
    job_tag = f"tm-{task}-{int(time.time())}"
    instance_id = api.launch(region, instance_type, [ssh_key_name], name=job_tag)
    log(f"instance_id={instance_id}")

    exit_code = 1
    baseline_started = False
    job_finished = False
    ip = None
    try:
        inst = wait_for_active(api, instance_id, timeout=boot_timeout)
        ip = inst["ip"]
        wait_for_ssh(ssh_private_key, ip)

        # Push scripts + env to the instance.
        scripts_dir = Path(__file__).resolve().parent
        scp_to(ssh_private_key, ip, str(scripts_dir / "lambda_bootstrap.sh"),
               "/home/ubuntu/lambda_bootstrap.sh")
        scp_to(ssh_private_key, ip, str(scripts_dir / "lambda_push_to_hf.py"),
               "/home/ubuntu/lambda_push_to_hf.py")
        if task == "sft":
            root = scripts_dir.parent
            with tempfile.TemporaryDirectory(prefix="sft-source-") as directory:
                bundle = Path(directory) / "sft_source.tar.gz"
                with tarfile.open(bundle, "w:gz") as archive:
                    for name in ("main.py", "src", "configs"):
                        archive.add(root / name, arcname=name,
                                    filter=lambda item: None if "__pycache__" in item.name else item)
                scp_to(ssh_private_key, ip, str(bundle), "/home/ubuntu/sft_source.tar.gz")
        if env_vars.get("EVAL_MEDICAL_MODELS"):
            copy_medical_eval_bundle(ssh_private_key, ip, env_vars["EVAL_DATASETS"].split(),
                                     bool(env_vars.get("EVAL_GENERAL_VI")))
        if env_vars.get("CPT_NEW_MODEL"):
            copy_new_cpt_bundle(ssh_private_key, ip)

        # Optionally scp a custom inference script (basename copied).
        inf_script = env_vars.get("INFERENCE_SCRIPT", "")
        if inf_script:
            local_inf = scripts_dir / Path(inf_script).name
            if local_inf.exists():
                log(f"scp custom inference script: {local_inf.name}")
                scp_to(ssh_private_key, ip, str(local_inf), f"/home/ubuntu/{local_inf.name}")
            else:
                log(f"INFERENCE_SCRIPT set but local file not found: {local_inf} — proceeding without scp")

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
        baseline_started = bool(env_vars.get("EVAL_MEDICAL_MODELS"))
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
                job_finished = True
                log("job finished: SUCCESS")
                exit_code = 0
                break
            if "DONE_FAILED" in sentinel:
                job_finished = True
                log("job finished: FAILED")
                exit_code = 2
                break

            time.sleep(POLL_INTERVAL_TRAIN)

    except KeyboardInterrupt:
        log("interrupted locally — preserving the SFT instance while its job may still be running.")
        if task == "sft":
            keep_alive = True
    finally:
        if task == "sft" and ip and not job_finished:
            log("SFT completion is unconfirmed; preserving the instance. Billing continues.")
            keep_alive = True
        if task == "sft" and ip and not keep_alive:
            try:
                download_sft_backup(ssh_private_key, ip, job_tag)
            except (Exception, KeyboardInterrupt) as error:
                log(f"WARNING: SFT backup failed: {error}")
                log("Leaving the instance running to preserve outputs; billing continues.")
                keep_alive = True
                exit_code = 2
        if baseline_started:
            try:
                download_medical_eval(ssh_private_key, ip, job_tag)
            except (Exception, KeyboardInterrupt) as error:
                log(f"WARNING: baseline result download failed: {error}")
                log("Check the Hugging Face upload in the job log. "
                    "Instance cleanup still follows --keep-alive.")
                exit_code = 2
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
    if args.task in ("cpt", "inference", "eval"):
        return []  # CPT.py / inference.py / eval loop take no config arg
    die("--config or --config-list required for SFT task")
    return []  # unreachable


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["cpt", "sft", "inference", "eval"],
                        required=True,
                        help="cpt -> CPT/CPT.py; sft -> main.py --config; "
                             "inference -> inference.py (compare models from README); "
                             "eval -> re-run Unsloth_QA_*.ipynb on all 6 README "
                             "models, push to each model's HF branch")
    parser.add_argument("--eval-only",
                        help="(eval task only) Override the 6-model loop with a "
                             "single model. Format: 'name|repo|revision|variant' "
                             "where variant ∈ {goc,cpt,sft,cpt_sft}. Example: "
                             "'SFT-exp0002|huyennkai/Venera-Traditional-Model|"
                             "sft-unsloth-qwen3-4B-exp0002-tm-sft-1777957925|sft'")
    parser.add_argument("--config", help="Path to a single SFT or CPT config yaml")
    parser.add_argument("--cpt-new-model", choices=["qwen3-4b-thinking-2507", "gemma4-e2b"],
                        help="Train a separate CPT LoRA on this base instead of legacy CPT.py")
    parser.add_argument("--cpt-max-steps", type=int, default=-1,
                        help="Optional short new-base CPT trial; positive step count")
    parser.add_argument("--baselines", "--medical-baselines", nargs="+",
                        help="(eval only) Baseline aliases, e.g. medreason; skips the README model loop")
    parser.add_argument("--general-vi", action="store_true",
                        help="Run Vietnamese general benchmarks; paper-style compares Qwen3-4B, 2507, and CPT")
    parser.add_argument("--eval-backend", choices=["transformers", "vllm"],
                        default="transformers", help="General Vietnamese inference backend")
    parser.add_argument("--eval-max-new-tokens", type=int, default=4096,
                        help="vLLM output token cap per answer")
    parser.add_argument("--eval-gpu-memory-utilization", type=float, default=0.70,
                        help="Fraction of GPU memory vLLM may reserve (default: 0.70)")
    parser.add_argument("--eval-enforce-eager", action="store_true",
                        help="Disable vLLM compilation and CUDA graphs to reduce startup memory")
    parser.add_argument("--eval-models", nargs="+", default=None,
                        help="General Vietnamese model aliases; e.g. qwen3-4b-paper gemma4-e2b")
    parser.add_argument("--eval-protocol", choices=["legacy", "paper-style", "thinking", "instruct"], default=None,
                        help="General Vietnamese evaluation profile; paper-style mirrors published Qwen3 settings")
    parser.add_argument("--eval-datasets", nargs="+", default=None,
                        help="Baseline datasets: vm14k (default, 564 rows), new, vm14k-full")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Baseline questions per dataset; 0 uses every row")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Baseline batch size; automatically reduced on GPU OOM")
    parser.add_argument("--eval-generation-mode", choices=["direct", "notebook"], default=None,
                        help="Baseline decoding: answer-only or notebook free generation")
    parser.add_argument("--config-list",
                        help="Path to a text file with one config path per line")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"Lambda instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--region", default=None,
                        help="Require a region, e.g. us-east-1 (Virginia); default: first available")
    parser.add_argument("--boot-timeout", type=int, default=600,
                        help="Seconds to wait for Lambda active status before cleanup (default: 600)")
    parser.add_argument("--git-repo",
                        default="https://github.com/Venera-AI/Traditional-Medical-Model.git")
    parser.add_argument("--git-branch", default="main2")
    parser.add_argument("--keep-alive", action="store_true",
                        help="Do not terminate the instance after the job ends")
    parser.add_argument("--list-types", action="store_true",
                        help="List all instance types + availability and exit")
    parser.add_argument("--inference-script", default=None,
                        help="(inference task) path to a custom inference script inside the repo (e.g. scripts/test_inference_max_tokens.py)")
    parser.add_argument("--inference-revision", default=None,
                        help="(inference task) HF revision to pass as --revision to the inference script")
    args = parser.parse_args()
    if args.cpt_new_model and args.task != "cpt":
        parser.error("--cpt-new-model requires --task cpt")
    if args.cpt_max_steps != -1 and (not args.cpt_new_model or args.cpt_max_steps < 1):
        parser.error("--cpt-max-steps requires --cpt-new-model and a positive value")
    if args.boot_timeout < 1:
        parser.error("--boot-timeout must be a positive number of seconds")
    if not (args.baselines or args.general_vi) and any(value is not None for value in (
            args.eval_datasets, args.eval_limit, args.eval_batch_size, args.eval_generation_mode,
            args.eval_protocol)):
        parser.error("--eval-datasets/limit/batch-size/generation-mode/protocol require --baselines or --general-vi")
    if args.eval_protocol and not args.general_vi:
        parser.error("--eval-protocol requires --general-vi")
    if args.eval_backend == "vllm" and not args.general_vi:
        parser.error("--eval-backend vllm requires --general-vi")
    if args.eval_protocol == "instruct" and args.eval_backend != "vllm":
        parser.error("--eval-protocol instruct requires --eval-backend vllm")
    if args.eval_max_new_tokens < 1:
        parser.error("--eval-max-new-tokens must be positive")
    if not 0 < args.eval_gpu_memory_utilization < 1:
        parser.error("--eval-gpu-memory-utilization must be between 0 and 1")
    if args.eval_models and not args.general_vi:
        parser.error("--eval-models requires --general-vi")
    if args.general_vi:
        if args.baselines or args.eval_generation_mode or args.eval_only or args.config or args.config_list:
            parser.error("--general-vi cannot be combined with medical baselines, generation mode, or training config")
        from eval_general_vi import BENCHMARKS, MODEL_KEYS, PAPER_STYLE_MODELS, ALL_MODEL_KEYS
        default_models = (("qwen3-4b-thinking-2507", "gemma4-e2b")
                          if args.eval_backend == "vllm" else
                          PAPER_STYLE_MODELS if args.eval_protocol == "paper-style" else MODEL_KEYS)
        args.baselines = list(args.eval_models or default_models)
        if any(model not in ALL_MODEL_KEYS for model in args.baselines):
            parser.error(f"Unknown general model; choose from: {', '.join(ALL_MODEL_KEYS)}")
        args.eval_datasets = args.eval_datasets or list(BENCHMARKS)
        if any(dataset not in BENCHMARKS for dataset in args.eval_datasets):
            parser.error(f"Unknown general benchmark; choose from: {', '.join(BENCHMARKS)}")
        if args.eval_backend == "vllm":
            allowed = {"thinking": ("qwen3-4b-thinking-2507", "gemma4-e2b"),
                       "instruct": ("qwen3-4b-cpt",)}.get(args.eval_protocol, ())
            if any(m not in allowed for m in args.baselines):
                parser.error("vLLM requires thinking baselines with --eval-protocol thinking, "
                             "or qwen3-4b-cpt with --eval-protocol instruct")
    args.eval_datasets = args.eval_datasets or ["vm14k"]
    args.eval_limit = 0 if args.eval_limit is None else args.eval_limit
    args.eval_batch_size = 4 if args.eval_batch_size is None else args.eval_batch_size
    args.eval_generation_mode = args.eval_generation_mode or "direct"
    if args.eval_limit < 0 or args.eval_batch_size < 1:
        parser.error("--eval-limit must be nonnegative and --eval-batch-size must be positive")
    if args.baselines:
        if args.task != "eval" or args.eval_only or args.config or args.config_list:
            parser.error("--baselines requires --task eval without --eval-only or training configs")
        # Ordinary training and notebook runs do not import the baseline evaluator.
        from eval_medical_baselines import MODELS, DATASETS, check_access, prepare_dataset
        if any(model not in MODELS for model in args.baselines):
            parser.error(f"Unknown baseline; choose from: {', '.join(MODELS)}")
        if not args.general_vi and any(dataset not in DATASETS for dataset in args.eval_datasets):
            parser.error(f"Unknown dataset; choose from: {', '.join(DATASETS)}")

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

    if args.baselines:
        try:
            check_git_access(args.git_repo, args.git_branch, os.environ.get("GH_TOKEN"))
            if args.general_vi:
                from eval_general_vi import source_rows
                for dataset in dict.fromkeys(args.eval_datasets):
                    source_rows(dataset, hf_token)
            else:
                for dataset in dict.fromkeys(args.eval_datasets):
                    prepare_dataset(dataset, args.eval_limit, args.eval_generation_mode)
            if args.general_vi:
                from eval_general_vi import resolve_models
                resolve_models(list(dict.fromkeys(args.baselines)), hf_token)
            else:
                check_access(list(dict.fromkeys(args.baselines)), hf_token)
        except Exception as error:
            die(f"baseline preflight failed before launching a GPU: {error}")

    if args.cpt_new_model:
        try:
            from huggingface_hub import HfApi
            from eval_medical_baselines import MODELS
            api_hf = HfApi(token=hf_token)
            check_git_access(args.git_repo, args.git_branch, os.environ.get("GH_TOKEN"))
            api_hf.model_info(MODELS[args.cpt_new_model])
            api_hf.repo_info("venera-ai/TradMed-CPT-53k-2048", repo_type="dataset")
            api_hf.repo_info(os.environ.get("HF_CPT_REPO", "Maverfrick/Venera-CPT-TMModel"))
        except Exception as error:
            die(f"new CPT preflight failed before launching a GPU: {error}")

    env_vars = {
        "HF_TOKEN": hf_token,
        "WANDB_KEY": os.environ.get("WANDB_KEY", ""),
        "GH_TOKEN": os.environ.get("GH_TOKEN", ""),
        "HF_CPT_REPO": os.environ.get("HF_CPT_REPO", "Maverfrick/Venera-CPT-TMModel"),
        "HF_SFT_REPO": os.environ.get("HF_SFT_REPO", "Maverfrick/Venera-Traditional-Model"),
        "CPT_NEW_MODEL": args.cpt_new_model or "",
        "CPT_MAX_STEPS": str(args.cpt_max_steps),
        "EVAL_BACKEND": args.eval_backend,
        "EVAL_MAX_NEW_TOKENS": str(args.eval_max_new_tokens),
        "EVAL_GPU_MEMORY_UTILIZATION": str(args.eval_gpu_memory_utilization),
        "EVAL_ENFORCE_EAGER": "1" if args.eval_enforce_eager else "",
        # Docker image overrides (all optional; bootstrap falls back to
        # ghcr.io/venera-ai/traditional-medical-model:<GIT_BRANCH>).
        "DOCKER_IMAGE": os.environ.get("DOCKER_IMAGE", ""),
        "GHCR_LOGIN":   os.environ.get("GHCR_LOGIN", ""),
        "GHCR_USER":    os.environ.get("GHCR_USER", ""),
        # eval task: when set, run_eval() loops over JUST this one entry
        # instead of the hard-coded 6-model README list.
        "EVAL_ONLY":    args.eval_only or "",
        # Optional override: run a custom inference script instead of
        # the default `inference.py` inside the container.
        "INFERENCE_SCRIPT": args.inference_script or os.environ.get("INFERENCE_SCRIPT", ""),
        # Optional: pass a HF revision string to the inference script.
        "INFERENCE_REVISION": args.inference_revision or os.environ.get("INFERENCE_REVISION", ""),
    }

    if args.baselines:
        env_vars.update({
            "EVAL_MEDICAL_MODELS": " ".join(dict.fromkeys(args.baselines)),
            "EVAL_DATASETS": " ".join(dict.fromkeys(args.eval_datasets)),
            "EVAL_LIMIT": str(args.eval_limit),
            "EVAL_BATCH_SIZE": str(args.eval_batch_size),
            "EVAL_GENERATION_MODE": args.eval_generation_mode,
            "EVAL_GENERAL_VI": "1" if args.general_vi else "",
            "EVAL_PROTOCOL": args.eval_protocol or "legacy",
        })

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
        region=args.region,
        boot_timeout=args.boot_timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
