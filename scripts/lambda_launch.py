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
import tarfile
import tempfile
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


def copy_medical_eval_bundle(key: str, ip: str) -> None:
    root = Path(__file__).resolve().parent.parent
    files = [
        "scripts/eval_medical_baselines.py",
        "EVAL/Unsloth_QA_goc_new.ipynb", "EVAL/Unsloth_QA_goc_VM14K.ipynb",
        "EVAL/Data/tcm_mcq.jsonl", "EVAL/Data/eastern_medicine_VM14K.jsonl",
    ]
    with tempfile.TemporaryDirectory(prefix="medical-eval-") as directory:
        bundle = Path(directory) / "medical_eval_bundle.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for name in files:
                archive.add(root / name, arcname=name)
        scp_to(key, ip, str(bundle), "/home/ubuntu/medical_eval_bundle.tar.gz")


def download_medical_eval(key: str, ip: str, job_tag: str) -> Path:
    remote_archive = f"/home/ubuntu/{job_tag}-medical-eval.tar.gz"
    remote_outputs = f"/home/ubuntu/{REMOTE_REPO_DIR}/EVAL/outputs/medical_baselines/{job_tag}"
    ssh_cmd(key, ip, f"sudo tar -czf {shlex.quote(remote_archive)} "
            f"-C {shlex.quote(remote_outputs)} . && sudo chmod 644 {shlex.quote(remote_archive)}")
    destination = Path("logs") / job_tag
    destination.mkdir(parents=True, exist_ok=True)
    local_archive = destination / "medical-eval.tar.gz"
    subprocess.run([
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15", "-i", key,
        f"{REMOTE_USER}@{ip}:{remote_archive}", str(local_archive),
    ], check=True)
    with tarfile.open(local_archive) as archive:
        if not archive.getmembers():
            raise RuntimeError("Downloaded evaluation archive is empty")
    log(f"evaluation backup saved locally: {local_archive.resolve()}")
    return local_archive


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
    ip = None
    job_started = False
    job_finished = False
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
        if env_vars.get("EVAL_MEDICAL_MODELS"):
            copy_medical_eval_bundle(ssh_private_key, ip)

        # The dedicated VM14K test evaluator may contain local changes that
        # are not on GIT_BRANCH yet. Copy it explicitly so eval-only
        # cpt_sft_test always runs the same file as this launcher checkout.
        eval_test_script = scripts_dir.parent / "EVAL" / "Unsloth_QA_CPT_SFT_VM14K_TEST.py"
        eval_only_parts = env_vars.get("EVAL_ONLY", "").split("|")
        if len(eval_only_parts) >= 4 and eval_only_parts[3] in {
            "cpt_sft_test", "goc_test"
        }:
            if not eval_test_script.exists():
                die(f"eval test script not found: {eval_test_script}")
            scp_to(ssh_private_key, ip, str(eval_test_script),
                   "/home/ubuntu/Unsloth_QA_CPT_SFT_VM14K_TEST.py")

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
        ssh_cmd(
            ssh_private_key, ip,
            "tmux new-session -d -s train "
            "'bash /home/ubuntu/lambda_bootstrap.sh 2>&1 | tee /home/ubuntu/train.log'",
        )
        job_started = True

        # Follow logs until sentinel file appears. We use a non-fatal tail:
        # if it disconnects, we reconnect and keep polling status.
        if env_vars.get("EVAL_MEDICAL_MODELS"):
            log("streaming logs (Ctrl+C leaves the evaluation instance running and billable)")
        else:
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
        if env_vars.get("EVAL_MEDICAL_MODELS"):
            log("interrupted locally; preserving the remote evaluation instance")
        else:
            log("interrupted locally — remote job continues; proceeding to terminate.")
    finally:
        if env_vars.get("EVAL_MEDICAL_MODELS") and ip and job_started:
            if not job_finished:
                keep_alive = True
                log("Medical evaluation has not finished; leaving it running. GPU billing continues.")
            try:
                download_medical_eval(ssh_private_key, ip, job_tag)
            except Exception as error:
                # Retain results even if the Hub upload or local backup fails.
                keep_alive = True
                exit_code = 2
                log(f"evaluation backup failed: {error}")
                log("Leaving instance running to preserve results; GPU billing continues. "
                    "Download EVAL/outputs/medical_baselines, then terminate it manually.")
        if keep_alive:
            log(f"leaving instance {instance_id} running")
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
                             "single model. Format: "
                             "'name|source_repo|revision|variant[|output_repo]' "
                             "where variant ∈ "
                             "{goc,goc_test,cpt,sft,cpt_sft,cpt_sft_test}. "
                             "output_repo is optional and lets evaluation load "
                             "a model from a read-only repo but upload results "
                             "to a writable repo. "
                             "Use cpt_sft_test for an adapter or goc_test for "
                             "a clean base model on the official 29-row VM14K "
                             "test split. Example: "
                             "'SFT-exp0002|huyennkai/Venera-Traditional-Model|"
                             "sft-unsloth-qwen3-4B-exp0002-tm-sft-1777957925|sft'")
    parser.add_argument("--config", help="Path to a single SFT config yaml")
    parser.add_argument("--medical-baselines", nargs="+", choices=["medgemma", "huatuo"],
                        help="(eval task) Evaluate selected medical models on new + VM14K MCQs")
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="(medical baselines) Questions per dataset; 0 uses the full dataset")
    parser.add_argument("--eval-batch-size", type=int, default=4,
                        help="(medical baselines) Initial batch size, reduced on GPU OOM")
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
    parser.add_argument("--inference-script", default=None,
                        help="(inference task) path to a custom inference script inside the repo (e.g. scripts/test_inference_max_tokens.py)")
    parser.add_argument("--inference-revision", default=None,
                        help="(inference task) HF revision to pass as --revision to the inference script")
    args = parser.parse_args()
    if args.medical_baselines and (args.task != "eval" or args.eval_only):
        parser.error("--medical-baselines requires --task eval and cannot use --eval-only")
    if args.eval_limit < 0 or args.eval_batch_size < 1:
        parser.error("--eval-limit must be nonnegative and --eval-batch-size must be positive")

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
    if args.medical_baselines:
        from eval_medical_baselines import DATASETS, check_access, prepare_dataset
        try:
            for dataset in DATASETS:
                prepare_dataset(dataset, args.eval_limit)
            check_access(args.medical_baselines, hf_token)
        except Exception as error:
            die(f"medical evaluation preflight failed before launching a GPU: {error}")

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
        # eval task: when set, run_eval() loops over JUST this one entry
        # instead of the hard-coded 6-model README list.
        "EVAL_ONLY":    args.eval_only or "",
        "EVAL_MEDICAL_MODELS": " ".join(dict.fromkeys(args.medical_baselines or [])),
        "EVAL_LIMIT": str(args.eval_limit),
        "EVAL_BATCH_SIZE": str(args.eval_batch_size),
        # Optional override: run a custom inference script instead of
        # the default `inference.py` inside the container.
        "INFERENCE_SCRIPT": args.inference_script or os.environ.get("INFERENCE_SCRIPT", ""),
        # Optional: pass a HF revision string to the inference script.
        "INFERENCE_REVISION": args.inference_revision or os.environ.get("INFERENCE_REVISION", ""),
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
